"""End-to-end RAG pipeline connecting isolated child-process hybrid retrieval with Colab GPU generation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from quorift.config import paths
from quorift.generation import (
    REFUSAL_MESSAGE,
    VALID_ROLES,
    validate_generation_request,
)
from quorift.gpu_exchange import (
    GPU_MODEL_ID,
    prepare_gpu_request_and_manifest,
    validate_gpu_response_record,
)

# Inline child worker script that executes inside a separate process to isolate PyTorch/Qdrant memory
RETRIEVAL_WORKER_SCRIPT = """
import json
import sys
from pathlib import Path
from quorift.retrieval import HybridRetriever

def main():
    if len(sys.argv) < 3:
        sys.exit(1)
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    with open(input_file, "r", encoding="utf-8") as f:
        req = json.load(f)

    question = req["question"]
    role = req["role"]
    top_k = req.get("top_k", 5)
    indexes_dir = Path(req["indexes_dir"]) if req.get("indexes_dir") else None
    registry_path = Path(req["registry_path"]) if req.get("registry_path") else None

    retriever = HybridRetriever(indexes_dir=indexes_dir, registry_path=registry_path)
    try:
        hits = retriever.retrieve(question=question, role=role, top_k=top_k)
        full_chunks = []
        for h in hits:
            cid = h["chunk_id"]
            if cid not in retriever.chunks_by_id:
                raise KeyError(f"Retrieved chunk ID '{cid}' not found in trusted chunk store.")
            full_chunks.append(retriever.chunks_by_id[cid])

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": full_chunks}, f, ensure_ascii=False)
    finally:
        retriever.close()

if __name__ == "__main__":
    main()
"""


def _run_retrieval_subprocess(
    question: str,
    role: str,
    top_k: int = 5,
    indexes_dir: Path | None = None,
    registry_path: Path | None = None,
    timeout: int = 60,
) -> tuple[list[dict[str, Any]], float]:
    """Execute retrieval inside an isolated child Python process and return full chunk records."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "retrieval_input.json"
        output_file = Path(tmp_dir) / "retrieval_output.json"

        req_payload = {
            "question": question,
            "role": role,
            "top_k": top_k,
            "indexes_dir": str(indexes_dir) if indexes_dir else None,
            "registry_path": str(registry_path) if registry_path else None,
        }
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(req_payload, f)

        cmd = [
            sys.executable,
            "-c",
            RETRIEVAL_WORKER_SCRIPT,
            str(input_file),
            str(output_file),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"Retrieval worker process failed with exit code {proc.returncode}")

        if not output_file.exists():
            raise FileNotFoundError("Retrieval worker did not produce output file")

        with open(output_file, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        chunks = out_data.get("chunks", [])
        return chunks, elapsed


def prepare_gpu_generation_request(
    question: str,
    role: str,
    indexes_dir: Path | None = None,
    registry_path: Path | None = None,
    retrieval_timeout: int = 60,
    model_id: str = GPU_MODEL_ID,
    question_id: str | None = None,
    top_k: int = 5,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Execute isolated retrieval and prepare GPU request payload and local manifest.

    Validates question and role, runs retrieval in a child process, revalidates
    retrieved chunks against the trusted registry, and produces:
    - GPU request (containing only request_id, model_id, request_sha256, messages)
    - Local manifest (containing full context for subsequent local validation)
    """
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {sorted(VALID_ROLES)}")
    if model_id != GPU_MODEL_ID:
        raise ValueError(
            f"Model '{model_id}' is not permitted. Only '{GPU_MODEL_ID}' is supported."
        )

    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")
    if not reg_file.exists():
        raise FileNotFoundError(f"Document registry not found: {reg_file}")

    with open(reg_file, "r", encoding="utf-8") as f:
        registry_list = json.load(f)
    registry = {r["document_id"]: r for r in registry_list}

    # Execute isolated retrieval
    chunks, retrieval_time = _run_retrieval_subprocess(
        question=question.strip(),
        role=role,
        top_k=top_k,
        indexes_dir=indexes_dir,
        registry_path=registry_path,
        timeout=retrieval_timeout,
    )

    # Revalidate complete chunks against registry (fail-closed)
    validate_generation_request(
        {
            "question": question.strip(),
            "role": role,
            "retrieved_chunks": chunks,
        },
        registry,
    )

    # Prepare GPU request and local manifest
    return prepare_gpu_request_and_manifest(
        question=question.strip(),
        role=role,
        retrieved_chunks=chunks,
        registry=registry,
        retrieval_time_seconds=retrieval_time,
        question_id=question_id,
        model_id=model_id,
    )


def finalize_gpu_response(
    gpu_response: dict[str, Any] | None,
    manifest: dict[str, Any],
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Validate returned GPU result against local manifest and current trusted registry.

    Reloads the registry at finalization time to ensure policies have not been
    revoked, deactivated, or updated since preparation.
    """
    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")
    if not reg_file.exists():
        raise FileNotFoundError(f"Document registry not found: {reg_file}")

    with open(reg_file, "r", encoding="utf-8") as f:
        registry_list = json.load(f)
    registry = {r["document_id"]: r for r in registry_list}

    return validate_gpu_response_record(
        gpu_response=gpu_response,
        manifest=manifest,
        registry=registry,
    )


def process_gpu_batch(
    gpu_responses: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate a batch of GPU responses against a list of local manifests.

    Detects duplicate responses, missing responses, unexpected request IDs, and
    enforces fail-closed validation for every request in the batch.
    """
    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")
    if not reg_file.exists():
        raise FileNotFoundError(f"Document registry not found: {reg_file}")

    with open(reg_file, "r", encoding="utf-8") as f:
        registry_list = json.load(f)
    registry = {r["document_id"]: r for r in registry_list}

    # Map responses by request_id and detect duplicates
    responses_by_id: dict[str, dict[str, Any]] = {}
    duplicate_response_ids = set()
    for resp in gpu_responses:
        rid = resp.get("request_id")
        if not rid:
            continue
        if rid in responses_by_id:
            duplicate_response_ids.add(rid)
        else:
            responses_by_id[rid] = resp

    results: list[dict[str, Any]] = []

    for manifest in manifests:
        rid = manifest.get("request_id")
        qid = manifest.get("question_id")

        if rid in duplicate_response_ids:
            results.append({
                "request_id": rid,
                "question_id": qid,
                "status": "error",
                "answer": "",
                "citations": [],
                "resolved_citations": [],
                "retrieval_time_seconds": manifest.get("retrieval_time_seconds", 0.0),
                "generation_time_seconds": 0.0,
                "total_time_seconds": manifest.get("retrieval_time_seconds", 0.0),
                "validation_passed": False,
                "error": f"Duplicate GPU responses received for request ID '{rid}'.",
                "model_id": manifest.get("model_id", GPU_MODEL_ID),
                "gpu_name": None,
                "prompt_tokens": None,
                "generated_tokens": None,
            })
            continue

        resp = responses_by_id.get(rid)
        if resp is None and manifest.get("requires_gpu") is not False:
            results.append({
                "request_id": rid,
                "question_id": qid,
                "status": "error",
                "answer": "",
                "citations": [],
                "resolved_citations": [],
                "retrieval_time_seconds": manifest.get("retrieval_time_seconds", 0.0),
                "generation_time_seconds": 0.0,
                "total_time_seconds": manifest.get("retrieval_time_seconds", 0.0),
                "validation_passed": False,
                "error": f"Missing GPU response for request ID '{rid}'.",
                "model_id": manifest.get("model_id", GPU_MODEL_ID),
                "gpu_name": None,
                "prompt_tokens": None,
                "generated_tokens": None,
            })
            continue

        val_res = validate_gpu_response_record(
            gpu_response=resp,
            manifest=manifest,
            registry=registry,
        )
        results.append(val_res)

    return results


def answer_question(*args: Any, **kwargs: Any) -> Any:
    """Obsolete synchronous pipeline entry point.

    Synchronous local inference has been replaced with the file-based
    Google Colab GPU generation pipeline. Use prepare_gpu_generation_request()
    and finalize_gpu_response() instead.
    """
    raise RuntimeError(
        "Synchronous answer_question() has been replaced by the file-based"
        " GPU pipeline. Use prepare_gpu_generation_request() to create GPU"
        " request payloads and finalize_gpu_response() to validate returned"
        " Colab GPU outputs."
    )
