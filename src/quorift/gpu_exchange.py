"""GPU batch exchange module for file-based Google Colab generation and local validation."""

import hashlib
import json
from pathlib import Path
import time
from typing import Any
import uuid

from quorift.generation import (
    REFUSAL_MESSAGE,
    VALID_ROLES,
    build_grounded_prompt,
    generate_empty_evidence_response,
    parse_generation_response,
    resolve_citation_metadata,
    validate_generation_output,
    validate_generation_request,
)

GPU_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def compute_request_fingerprint(
    request_id: str,
    model_id: str,
    messages: list[dict[str, str]],
) -> str:
    """Compute deterministic SHA-256 fingerprint over serialized request content.

    Note: This fingerprint is used to detect mismatched or accidentally modified
    files during exchange; it is not cryptographic proof of authenticity.
    """
    payload = {
        "messages": messages,
        "model_id": model_id,
        "request_id": request_id,
    }
    canonical_bytes = json.dumps(
        payload, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def prepare_gpu_request_and_manifest(
    question: str,
    role: str,
    retrieved_chunks: list[dict[str, Any]],
    registry: dict[str, Any],
    retrieval_time_seconds: float = 0.0,
    question_id: str | None = None,
    model_id: str = GPU_MODEL_ID,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Prepare a GPU-safe request and a trusted local manifest record.

    The exported GPU request contains only:
    - request_id
    - model_id
    - request_sha256 (fingerprint)
    - messages (grounded chat prompt)

    It NEVER contains:
    - Gold answers or labels
    - Private evaluation metrics
    - Fact ledger records
    - Full document registry
    - Chunks outside authorized evidence
    - Local filesystem paths
    - Secrets or credentials

    If no authorized chunks are retrieved, the manifest is flagged with
    requires_gpu=False, allowing immediate local refusal without GPU generation.
    """
    # 1. Validate inputs
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {sorted(VALID_ROLES)}")
    if model_id != GPU_MODEL_ID:
        raise ValueError(
            f"Model '{model_id}' is not permitted. Only '{GPU_MODEL_ID}' is supported."
        )

    # 2. Validate request evidence against trusted registry
    req_payload = {
        "question": question.strip(),
        "role": role,
        "retrieved_chunks": retrieved_chunks,
    }
    validate_generation_request(req_payload, registry)

    request_id = f"req_{uuid.uuid4().hex[:16]}"

    # 3. Check for empty evidence: safe refusal locally
    if len(retrieved_chunks) == 0:
        messages = build_grounded_prompt(question.strip(), [])
        fingerprint = compute_request_fingerprint(request_id, model_id, messages)

        gpu_request = {
            "request_id": request_id,
            "model_id": model_id,
            "request_sha256": fingerprint,
            "messages": messages,
        }
        manifest = {
            "request_id": request_id,
            "model_id": model_id,
            "request_sha256": fingerprint,
            "question": question.strip(),
            "role": role,
            "question_id": question_id,
            "retrieved_chunks": [],
            "retrieval_time_seconds": retrieval_time_seconds,
            "requires_gpu": False,
            "created_at": time.time(),
        }
        return gpu_request, manifest

    # 4. Standard evidence: construct grounded messages and fingerprint
    messages = build_grounded_prompt(question.strip(), retrieved_chunks)
    fingerprint = compute_request_fingerprint(request_id, model_id, messages)

    gpu_request = {
        "request_id": request_id,
        "model_id": model_id,
        "request_sha256": fingerprint,
        "messages": messages,
    }

    manifest = {
        "request_id": request_id,
        "model_id": model_id,
        "request_sha256": fingerprint,
        "question": question.strip(),
        "role": role,
        "question_id": question_id,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_time_seconds": retrieval_time_seconds,
        "requires_gpu": True,
        "created_at": time.time(),
    }

    return gpu_request, manifest


def validate_gpu_response_record(
    gpu_response: dict[str, Any] | None,
    manifest: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Validate a single GPU response against its local manifest and current trusted registry.

    Enforces fail-closed security:
    - Never displays unvalidated or raw model text as final answer.
    - Revalidates evidence permissions, active versions, and content hashes against registry.
    - Verifies request ID, model ID, and request fingerprint.
    - Validates strict JSON response schema, citation IDs, and exact refusal format.
    """
    req_id = manifest.get("request_id", "unknown")
    qid = manifest.get("question_id")
    role = manifest.get("role", "general")
    chunks = manifest.get("retrieved_chunks", [])
    retrieval_time = manifest.get("retrieval_time_seconds", 0.0)

    # 1. Local resolution for empty evidence
    if manifest.get("requires_gpu") is False or len(chunks) == 0:
        return {
            "request_id": req_id,
            "question_id": qid,
            "status": "insufficient_evidence",
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": 0.0,
            "total_time_seconds": retrieval_time,
            "validation_passed": True,
            "error": None,
            "model_id": manifest.get("model_id", GPU_MODEL_ID),
            "gpu_name": None,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        }

    # 2. Check presence of response
    if gpu_response is None or not isinstance(gpu_response, dict):
        return {
            "request_id": req_id,
            "question_id": qid,
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": 0.0,
            "total_time_seconds": retrieval_time,
            "validation_passed": False,
            "error": "Missing or non-dictionary GPU response record.",
            "model_id": manifest.get("model_id", GPU_MODEL_ID),
            "gpu_name": None,
            "prompt_tokens": None,
            "generated_tokens": None,
        }

    gen_time = gpu_response.get("generation_time_seconds", 0.0)
    total_time = round(retrieval_time + gen_time, 4)
    gpu_name = gpu_response.get("gpu_name")
    prompt_tokens = gpu_response.get("prompt_tokens")
    gen_tokens = gpu_response.get("generated_tokens")

    # Helper for sanitized error return
    def _fail(err_msg: str) -> dict[str, Any]:
        return {
            "request_id": req_id,
            "question_id": qid,
            "status": "error",
            "answer": "",
            "citations": [],
            "resolved_citations": [],
            "retrieval_time_seconds": retrieval_time,
            "generation_time_seconds": gen_time,
            "total_time_seconds": total_time,
            "validation_passed": False,
            "error": err_msg,
            "model_id": manifest.get("model_id", GPU_MODEL_ID),
            "gpu_name": gpu_name,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
        }

    # 3. Check request ID match
    resp_req_id = gpu_response.get("request_id")
    if not resp_req_id or resp_req_id != req_id:
        return _fail(f"Request ID mismatch: expected '{req_id}', got '{resp_req_id}'.")

    # 4. Check model ID match
    resp_model_id = gpu_response.get("model_id")
    if resp_model_id != GPU_MODEL_ID:
        return _fail(
            f"Model ID mismatch: expected '{GPU_MODEL_ID}', got '{resp_model_id}'."
        )

    # 5. Check request fingerprint match
    resp_sha256 = gpu_response.get("request_sha256")
    expected_sha256 = manifest.get("request_sha256")
    if not resp_sha256 or resp_sha256 != expected_sha256:
        return _fail(
            "Request fingerprint mismatch: response does not correspond to manifest payload."
        )

    # 6. Check GPU execution status
    if gpu_response.get("status") != "success":
        gpu_err = gpu_response.get("error") or "GPU generation failed."
        # Sanitize error to avoid leaking system paths or tracebacks
        sanitized = "GPU model generation failed."
        return _fail(sanitized)

    # 7. Revalidate evidence against CURRENT trusted registry (detect stale policies)
    try:
        validate_generation_request(
            {
                "question": manifest.get("question", ""),
                "role": role,
                "retrieved_chunks": chunks,
            },
            registry,
        )
    except Exception:
        return _fail(
            "Evidence revalidation failed against current registry (document revoked, inactive, or hash mismatch)."
        )

    # 8. Parse raw model text
    raw_text = gpu_response.get("raw_model_output")
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return _fail("Model response is empty or invalid.")

    try:
        parsed = parse_generation_response(raw_text)
    except Exception:
        return _fail("Model output validation failed: malformed JSON response.")

    # 9. Validate structured generation output against retrieved chunks
    try:
        validated = validate_generation_output(parsed, chunks, registry, role)
    except Exception:
        return _fail("Model output validation failed: invalid format, citations, or refusal text.")

    # 10. Resolve citation metadata
    resolved_citations = [
        resolve_citation_metadata(cid, chunks) for cid in validated["citations"]
    ]

    return {
        "request_id": req_id,
        "question_id": qid,
        "status": validated["status"],
        "answer": validated["answer"],
        "citations": validated["citations"],
        "resolved_citations": resolved_citations,
        "retrieval_time_seconds": retrieval_time,
        "generation_time_seconds": gen_time,
        "total_time_seconds": total_time,
        "validation_passed": True,
        "error": None,
        "model_id": GPU_MODEL_ID,
        "gpu_name": gpu_name,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
    }
