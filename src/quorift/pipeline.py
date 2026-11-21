"""End-to-end RAG pipeline connecting isolated child-process hybrid retrieval with Ollama generation."""

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
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
    OLLAMA_REQUEST_TIMEOUT,
    generate_grounded_answer_ollama,
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
            if cid in retriever.chunks_by_id:
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
            err_msg = proc.stderr.strip() if proc.stderr else f"Exit code {proc.returncode}"
            raise RuntimeError(f"Retrieval worker process failed: {err_msg}")

        if not output_file.exists():
            raise FileNotFoundError(f"Retrieval worker did not produce output file: {output_file}")

        with open(output_file, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        chunks = out_data.get("chunks", [])
        return chunks, elapsed


def answer_question(
    question: str,
    role: str,
    indexes_dir: Path | None = None,
    registry_path: Path | None = None,
    timeout: int = OLLAMA_REQUEST_TIMEOUT,
    retrieval_timeout: int = 60,
    api_url: str = OLLAMA_API_URL,
    model_id: str = OLLAMA_MODEL_ID,
) -> dict[str, Any]:
    """Execute complete end-to-end RAG pipeline with process-isolated retrieval and grounded generation."""
    t_total_start = time.perf_counter()

    # 1. Validate question and role inputs
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {sorted(VALID_ROLES)}")

    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")
    if not reg_file.exists():
        raise FileNotFoundError(f"Document registry not found: {reg_file}")

    with open(reg_file, "r", encoding="utf-8") as f:
        registry_list = json.load(f)
    registry = {r["document_id"]: r for r in registry_list}

    # 2. Execute retrieval inside separate child process
    retrieval_time = 0.0
    try:
        chunks, retrieval_time = _run_retrieval_subprocess(
            question=question.strip(),
            role=role,
            top_k=5,
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            timeout=retrieval_timeout,
        )
    except Exception as e:
        total_time = max(time.perf_counter() - t_total_start, retrieval_time)
        return {
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": 0.0,
            "total_time_seconds": total_time,
            "validation_passed": False,
            "error": f"Retrieval failed: {e}",
            "model_id": model_id,
            "retrieved_chunks_count": 0,
        }

    # 3. Revalidate retrieved evidence against trusted registry (fail-closed)
    try:
        validate_generation_request(
            {
                "question": question.strip(),
                "role": role,
                "retrieved_chunks": chunks,
            },
            registry,
        )
    except Exception as e:
        total_time = max(time.perf_counter() - t_total_start, retrieval_time)
        return {
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": 0.0,
            "total_time_seconds": total_time,
            "validation_passed": False,
            "error": "Retrieved evidence failed security or integrity validation.",
            "model_id": model_id,
            "retrieved_chunks_count": len(chunks),
        }

    # 4. Generate grounded answer via Ollama backend
    try:
        gen_res = generate_grounded_answer_ollama(
            question=question.strip(),
            role=role,
            retrieved_chunks=chunks,
            registry=registry,
            model_id=model_id,
            api_url=api_url,
            timeout=timeout,
        )
    except Exception as e:
        total_time = max(time.perf_counter() - t_total_start, retrieval_time)
        return {
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": 0.0,
            "total_time_seconds": total_time,
            "validation_passed": False,
            "error": f"Generation failed: {e}",
            "model_id": model_id,
            "retrieved_chunks_count": len(chunks),
        }

    total_time = max(
        time.perf_counter() - t_total_start,
        retrieval_time + gen_res.get("generation_time_seconds", 0.0),
    )

    # 5. Check validation results: never display raw unvalidated text on failure
    if not gen_res.get("validation_passed", False):
        return {
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": gen_res.get("generation_time_seconds", 0.0),
            "total_time_seconds": total_time,
            "validation_passed": False,
            "error": f"Model output validation failed: {gen_res.get('validation_error')}",
            "model_id": model_id,
            "retrieved_chunks_count": len(chunks),
        }

    return {
        "status": gen_res["parsed_status"],
        "answer": gen_res["parsed_answer"],
        "citations": gen_res["validated_citations"],
        "resolved_citations": gen_res["resolved_citations"],
        "retrieval_time_seconds": retrieval_time,
        "generation_time_seconds": gen_res["generation_time_seconds"],
        "total_time_seconds": total_time,
        "validation_passed": True,
        "error": None,
        "model_id": model_id,
        "retrieved_chunks_count": len(chunks),
    }
