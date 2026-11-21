"""Offline unit tests for GPU batch exchange, fingerprinting, and response validation."""

import hashlib
import json
import pytest

from quorift.generation import REFUSAL_MESSAGE
from quorift.gpu_exchange import (
    GPU_MODEL_ID,
    compute_request_fingerprint,
    prepare_gpu_request_and_manifest,
    validate_gpu_response_record,
)


@pytest.fixture
def sample_registry():
    """Trusted document registry fixture."""
    return {
        "GEN-02": {
            "document_id": "GEN-02",
            "active_version": "1.0",
            "source_content_hash": "b1480820d8326b16486ad48b34c3d5d9206ad095a66541d57865d109484d8c71",
            "active_status": True,
            "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        },
        "LND-01": {
            "document_id": "LND-01",
            "active_version": "1.0",
            "source_content_hash": "b41de702abe0f4520e16190848c50e32e47c191b446488db3c73096298343ad3",
            "active_status": True,
            "allowed_roles": ["lending", "admin"],
        },
    }


@pytest.fixture
def sample_chunk_gen02():
    """Authorized chunk fixture for GEN-02."""
    return {
        "chunk_id": "GEN-02|1.0|2.0",
        "document_id": "GEN-02",
        "title": "Travel and Expense Policy",
        "version": "1.0",
        "section_number": "2.0",
        "section_heading": "Lodging Ceilings",
        "page_start": 2,
        "page_end": 2,
        "source_pdf": "data/documents/gen_02.pdf",
        "source_hash": "b1480820d8326b16486ad48b34c3d5d9206ad095a66541d57865d109484d8c71",
        "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        "text": "The lodging expense ceiling per night for Tier 1 metro cities is INR 6,500.",
    }


@pytest.fixture
def sample_chunk_lnd01():
    """Restricted chunk fixture for LND-01."""
    return {
        "chunk_id": "LND-01|1.0|3.0",
        "document_id": "LND-01",
        "title": "Credit Policy",
        "version": "1.0",
        "section_number": "3.0",
        "section_heading": "DTI Caps",
        "page_start": 3,
        "page_end": 4,
        "source_pdf": "data/documents/lnd_01.pdf",
        "source_hash": "b41de702abe0f4520e16190848c50e32e47c191b446488db3c73096298343ad3",
        "allowed_roles": ["lending", "admin"],
        "text": "Any personal loan where DTI exceeds 45% requires credit risk review.",
    }


# 1. Fingerprint calculation is deterministic and sensitive to changes
def test_compute_request_fingerprint_deterministic():
    messages = [
        {"role": "system", "content": "System text"},
        {"role": "user", "content": "Question text"},
    ]
    fp1 = compute_request_fingerprint("req_001", GPU_MODEL_ID, messages)
    fp2 = compute_request_fingerprint("req_001", GPU_MODEL_ID, messages)
    assert fp1 == fp2
    assert len(fp1) == 64

    # Different request ID produces different fingerprint
    fp3 = compute_request_fingerprint("req_002", GPU_MODEL_ID, messages)
    assert fp1 != fp3

    # Different message content produces different fingerprint
    messages_altered = [
        {"role": "system", "content": "System text"},
        {"role": "user", "content": "Altered text"},
    ]
    fp4 = compute_request_fingerprint("req_001", GPU_MODEL_ID, messages_altered)
    assert fp1 != fp4


# 2. Input validation for prepare_gpu_request_and_manifest
def test_prepare_gpu_request_input_validation(sample_registry):
    # Empty question
    with pytest.raises(ValueError, match="Question must be a nonempty string"):
        prepare_gpu_request_and_manifest("", "general", [], sample_registry)

    # Whitespace question
    with pytest.raises(ValueError, match="Question must be a nonempty string"):
        prepare_gpu_request_and_manifest("   ", "general", [], sample_registry)

    # Invalid role
    with pytest.raises(ValueError, match="Invalid role 'auditor'"):
        prepare_gpu_request_and_manifest("Valid question?", "auditor", [], sample_registry)

    # Invalid model
    with pytest.raises(ValueError, match="Model 'llama3' is not permitted"):
        prepare_gpu_request_and_manifest(
            "Valid question?", "general", [], sample_registry, model_id="llama3"
        )


# 3. GPU request export contains only permitted fields (no gold labels or private data)
def test_gpu_request_contains_no_sensitive_fields(sample_registry, sample_chunk_gen02):
    gpu_req, manifest = prepare_gpu_request_and_manifest(
        question="What is lodging limit?",
        role="general",
        retrieved_chunks=[sample_chunk_gen02],
        registry=sample_registry,
        question_id="Q-001",
    )

    # GPU request must contain ONLY request_id, model_id, request_sha256, and messages
    assert set(gpu_req.keys()) == {"request_id", "model_id", "request_sha256", "messages"}
    assert gpu_req["model_id"] == GPU_MODEL_ID
    assert "gold" not in str(gpu_req).lower()
    assert "evaluation" not in str(gpu_req).lower()
    assert "fact_ledger" not in str(gpu_req).lower()

    # Manifest retains local metadata
    assert manifest["question"] == "What is lodging limit?"
    assert manifest["role"] == "general"
    assert manifest["question_id"] == "Q-001"
    assert manifest["requires_gpu"] is True
    assert manifest["retrieved_chunks"] == [sample_chunk_gen02]


# 4. Safe empty evidence refusal handled locally without requiring GPU
def test_empty_evidence_safe_local_refusal(sample_registry):
    gpu_req, manifest = prepare_gpu_request_and_manifest(
        question="Unknown query?",
        role="general",
        retrieved_chunks=[],
        registry=sample_registry,
    )
    assert manifest["requires_gpu"] is False
    assert manifest["retrieved_chunks"] == []

    # Validation should resolve immediately to safe refusal without a GPU response
    res = validate_gpu_response_record(None, manifest, sample_registry)
    assert res["status"] == "insufficient_evidence"
    assert res["answer"] == REFUSAL_MESSAGE
    assert res["citations"] == []
    assert res["validation_passed"] is True
    assert res["error"] is None


# 5. Successful validation of valid GPU answer
def test_validate_gpu_response_success(sample_registry, sample_chunk_gen02):
    gpu_req, manifest = prepare_gpu_request_and_manifest(
        question="What is lodging limit?",
        role="general",
        retrieved_chunks=[sample_chunk_gen02],
        registry=sample_registry,
        retrieval_time_seconds=1.25,
    )

    model_json = json.dumps({
        "status": "answered",
        "answer": "The lodging expense ceiling is INR 6,500.",
        "citations": ["GEN-02|1.0|2.0"],
    })
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": model_json,
        "generation_time_seconds": 2.50,
        "gpu_name": "NVIDIA Tesla T4",
        "prompt_tokens": 1200,
        "generated_tokens": 40,
        "error": None,
    }

    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "answered"
    assert res["answer"] == "The lodging expense ceiling is INR 6,500."
    assert res["citations"] == ["GEN-02|1.0|2.0"]
    assert len(res["resolved_citations"]) == 1
    assert res["resolved_citations"][0]["title"] == "Travel and Expense Policy"
    assert res["validation_passed"] is True
    assert res["retrieval_time_seconds"] == 1.25
    assert res["generation_time_seconds"] == 2.50
    assert res["total_time_seconds"] == 3.75
    assert res["gpu_name"] == "NVIDIA Tesla T4"


# 6. Request ID mismatch rejection
def test_validate_gpu_response_request_id_mismatch(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    gpu_resp = {
        "request_id": "wrong_req_id",
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": '{"status": "answered", "answer": "text", "citations": ["GEN-02|1.0|2.0"]}',
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    assert "Request ID mismatch" in res["error"]
    assert res["answer"] == ""


# 7. Model ID mismatch rejection
def test_validate_gpu_response_model_id_mismatch(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": "gemma3:1b",
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": '{"status": "answered", "answer": "text", "citations": ["GEN-02|1.0|2.0"]}',
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    assert "Model ID mismatch" in res["error"]


# 8. Fingerprint mismatch rejection (tampered request)
def test_validate_gpu_response_fingerprint_mismatch(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": "0" * 64,
        "status": "success",
        "raw_model_output": '{"status": "answered", "answer": "text", "citations": ["GEN-02|1.0|2.0"]}',
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    assert "fingerprint mismatch" in res["error"]


# 9. GPU execution failure handling
def test_validate_gpu_response_gpu_execution_error(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "error",
        "raw_model_output": "",
        "error": "CUDA out of memory in tensor alloc at /usr/local/cuda/...",
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    # Raw CUDA tracebacks must be sanitized
    assert "GPU model generation failed" in res["error"]
    assert "/usr/local/cuda" not in res["error"]


# 10. Malformed JSON rejection (never displays unvalidated text)
def test_validate_gpu_response_malformed_json(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": "Not a valid json response at all!",
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    assert "malformed JSON" in res["error"]
    assert res["answer"] == ""


# 11. Fabricated citation rejection
def test_validate_gpu_response_fabricated_citation(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    # Model cites a chunk that was NOT in the retrieved evidence
    model_json = json.dumps({
        "status": "answered",
        "answer": "Answer text",
        "citations": ["FABRICATED|1.0|1.0"],
    })
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": model_json,
    }
    res = validate_gpu_response_record(gpu_resp, manifest, sample_registry)
    assert res["status"] == "error"
    assert res["validation_passed"] is False
    assert res["answer"] == ""


# 12. Invalid refusal format rejection (paraphrased refusal or non-empty citations)
def test_validate_gpu_response_invalid_refusal_format(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    # Paraphrased refusal
    bad_refusal_1 = json.dumps({
        "status": "insufficient_evidence",
        "answer": "Sorry, I could not find this in the documents.",
        "citations": [],
    })
    gpu_resp_1 = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": bad_refusal_1,
    }
    res_1 = validate_gpu_response_record(gpu_resp_1, manifest, sample_registry)
    assert res_1["status"] == "error"
    assert res_1["validation_passed"] is False

    # Refusal with non-empty citations
    bad_refusal_2 = json.dumps({
        "status": "insufficient_evidence",
        "answer": REFUSAL_MESSAGE,
        "citations": ["GEN-02|1.0|2.0"],
    })
    gpu_resp_2 = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": bad_refusal_2,
    }
    res_2 = validate_gpu_response_record(gpu_resp_2, manifest, sample_registry)
    assert res_2["status"] == "error"
    assert res_2["validation_passed"] is False


# 13. Current registry revalidation: stale hash, deactivation, or permission revocation
def test_validate_gpu_response_revalidation_against_updated_registry(sample_registry, sample_chunk_gen02):
    _, manifest = prepare_gpu_request_and_manifest(
        question="Query?", role="general", retrieved_chunks=[sample_chunk_gen02], registry=sample_registry
    )
    valid_model_json = json.dumps({
        "status": "answered",
        "answer": "Valid answer.",
        "citations": ["GEN-02|1.0|2.0"],
    })
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": valid_model_json,
    }

    # Case A: Document deactivated in registry since preparation
    updated_registry_deactivated = {
        "GEN-02": {**sample_registry["GEN-02"], "active_status": False}
    }
    res_a = validate_gpu_response_record(gpu_resp, manifest, updated_registry_deactivated)
    assert res_a["status"] == "error"
    assert "Evidence revalidation failed" in res_a["error"]

    # Case B: Role permission revoked
    updated_registry_revoked = {
        "GEN-02": {**sample_registry["GEN-02"], "allowed_roles": ["finance"]}
    }
    res_b = validate_gpu_response_record(gpu_resp, manifest, updated_registry_revoked)
    assert res_b["status"] == "error"
    assert "Evidence revalidation failed" in res_b["error"]
