"""Focused offline unit tests for the GPU-based RAG pipeline, subprocess isolation, batching, and CLIs."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch
import pytest

from quorift.config import paths
from quorift.generation import REFUSAL_MESSAGE
from quorift.gpu_exchange import GPU_MODEL_ID
from quorift.pipeline import (
    RETRIEVAL_WORKER_SCRIPT,
    _run_retrieval_subprocess,
    answer_question,
    finalize_gpu_response,
    prepare_gpu_generation_request,
    process_gpu_batch,
)
from scripts.prepare_gpu_batch import main as prepare_main
from scripts.validate_gpu_batch import main as validate_main


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
def mock_registry_file(tmp_path, sample_registry):
    """Temporary document registry JSON file."""
    reg_path = tmp_path / "document_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(list(sample_registry.values()), f)
    return reg_path


@pytest.fixture
def sample_chunk_gen02():
    """Authorized chunk fixture for GEN-02."""
    return {
        "chunk_id": "GEN-02|1.0|2.0",
        "document_id": "GEN-02",
        "title": "Travel and Expense Policy",
        "version": "1.0",
        "section_number": "2.0",
        "section_heading": "Standard Per Diem Rates",
        "page_start": 2,
        "page_end": 3,
        "source_pdf": "data/documents/gen_02.pdf",
        "source_hash": "b1480820d8326b16486ad48b34c3d5d9206ad095a66541d57865d109484d8c71",
        "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        "text": "The daily meal per diem for domestic travel is $75 per day.",
    }


@pytest.fixture
def sample_chunk_lnd01():
    """Restricted chunk fixture for LND-01 (lending/admin only)."""
    return {
        "chunk_id": "LND-01|1.0|1.0",
        "document_id": "LND-01",
        "title": "Commercial Lending Policy",
        "version": "1.0",
        "section_number": "1.0",
        "section_heading": "Approval Limits",
        "page_start": 1,
        "page_end": 2,
        "source_pdf": "data/documents/lnd_01.pdf",
        "source_hash": "b41de702abe0f4520e16190848c50e32e47c191b446488db3c73096298343ad3",
        "allowed_roles": ["lending", "admin"],
        "text": "Senior loan officers may approve commercial loans up to $500,000.",
    }


# ==============================================================================
# 1. Obsolete answer_question entry point
# ==============================================================================

def test_obsolete_answer_question_raises_clear_error():
    with pytest.raises(RuntimeError, match="Synchronous answer_question\\(\\) has been replaced"):
        answer_question("Any question?", "general")


# ==============================================================================
# 2. Subprocess Retrieval Isolation & Fail-Closed Checks
# ==============================================================================

def test_retrieval_subprocess_isolation(sample_chunk_gen02):
    """Test that _run_retrieval_subprocess spawns a separate python subprocess."""
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        def fake_subprocess_execution(cmd, **kwargs):
            out_file = Path(cmd[4])
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"chunks": [sample_chunk_gen02]}, f)
            return mock_proc

        mock_run.side_effect = fake_subprocess_execution

        chunks, elapsed = _run_retrieval_subprocess(
            question="What is per diem?",
            role="general",
            top_k=5,
        )

        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == sys.executable
        assert call_args[1] == "-c"
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "GEN-02|1.0|2.0"
        assert elapsed >= 0.0


def test_retrieval_worker_missing_chunk_raises_keyerror():
    """Verify RETRIEVAL_WORKER_SCRIPT strictly raises KeyError when chunk ID is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "in.json"
        output_file = Path(tmp_dir) / "out.json"

        with open(input_file, "w", encoding="utf-8") as f:
            json.dump({
                "question": "test",
                "role": "general",
                "top_k": 5,
                "indexes_dir": None,
                "registry_path": None,
            }, f)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [{"chunk_id": "MISSING_CHUNK_99"}]
        mock_retriever.chunks_by_id = {"OTHER_CHUNK": {}}

        indented_script = "\n".join("    " + line for line in RETRIEVAL_WORKER_SCRIPT.strip().splitlines())
        test_code = f"""
import sys
from unittest.mock import patch, MagicMock

mock_retriever = MagicMock()
mock_retriever.retrieve.return_value = [{{"chunk_id": "MISSING_CHUNK_99"}}]
mock_retriever.chunks_by_id = {{"OTHER_CHUNK": {{}}}}

with patch("quorift.retrieval.HybridRetriever", return_value=mock_retriever):
{indented_script}
"""
        proc = subprocess.run(
            [sys.executable, "-c", test_code, str(input_file), str(output_file)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "KeyError" in proc.stderr
        assert "not found in trusted chunk store" in proc.stderr


# ==============================================================================
# 3. Preparation Pipeline: prepare_gpu_generation_request
# ==============================================================================

def test_prepare_gpu_generation_request_success(mock_registry_file, sample_chunk_gen02):
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.45)):
        gpu_req, manifest = prepare_gpu_generation_request(
            question="What is the lodging rate?",
            role="general",
            registry_path=mock_registry_file,
            question_id="Q-001",
        )

        assert gpu_req is not None
        assert gpu_req["model_id"] == GPU_MODEL_ID
        assert len(gpu_req["request_sha256"]) == 64
        assert len(gpu_req["messages"]) == 2
        assert manifest["question_id"] == "Q-001"
        assert manifest["requires_gpu"] is True
        assert manifest["retrieval_time_seconds"] == 0.45


def test_prepare_gpu_generation_request_invalid_inputs(mock_registry_file):
    with pytest.raises(ValueError, match="Question must be a nonempty string"):
        prepare_gpu_generation_request("", "general", registry_path=mock_registry_file)

    with pytest.raises(ValueError, match="Invalid role 'invalid_role'"):
        prepare_gpu_generation_request("Query?", "invalid_role", registry_path=mock_registry_file)

    with pytest.raises(ValueError, match="Model 'wrong_model' is not permitted"):
        prepare_gpu_generation_request("Query?", "general", registry_path=mock_registry_file, model_id="wrong_model")


def test_prepare_gpu_generation_request_revalidates_registry(mock_registry_file, sample_chunk_lnd01):
    """Restricted chunks returned to unauthorized role fail closed in preparation."""
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_lnd01], 0.25)):
        with pytest.raises(ValueError, match="Unauthorized evidence"):
            prepare_gpu_generation_request(
                question="What are loan limits?",
                role="general",  # role general not allowed to access LND-01
                registry_path=mock_registry_file,
            )


# ==============================================================================
# 4. Finalization and Batch Processing
# ==============================================================================

def test_finalize_gpu_response_success(mock_registry_file, sample_chunk_gen02):
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.5)):
        gpu_req, manifest = prepare_gpu_generation_request(
            question="What is lodging rate?",
            role="general",
            registry_path=mock_registry_file,
        )

    model_json = json.dumps({
        "status": "answered",
        "answer": "The rate is $75.",
        "citations": ["GEN-02|1.0|2.0"],
    })
    gpu_resp = {
        "request_id": manifest["request_id"],
        "model_id": GPU_MODEL_ID,
        "request_sha256": manifest["request_sha256"],
        "status": "success",
        "raw_model_output": model_json,
        "generation_time_seconds": 1.5,
        "gpu_name": "Tesla T4",
    }

    res = finalize_gpu_response(gpu_resp, manifest, registry_path=mock_registry_file)
    assert res["status"] == "answered"
    assert res["answer"] == "The rate is $75."
    assert res["citations"] == ["GEN-02|1.0|2.0"]
    assert res["validation_passed"] is True
    assert res["total_time_seconds"] == 2.0


def test_process_gpu_batch_handles_missing_and_duplicates(mock_registry_file, sample_chunk_gen02):
    manifest_1 = {
        "request_id": "req_1",
        "model_id": GPU_MODEL_ID,
        "request_sha256": "hash1",
        "question": "Q1",
        "role": "general",
        "question_id": "Q-001",
        "retrieved_chunks": [sample_chunk_gen02],
        "retrieval_time_seconds": 0.5,
        "requires_gpu": True,
    }
    manifest_2 = {
        "request_id": "req_2",
        "model_id": GPU_MODEL_ID,
        "request_sha256": "hash2",
        "question": "Q2",
        "role": "general",
        "question_id": "Q-002",
        "retrieved_chunks": [sample_chunk_gen02],
        "retrieval_time_seconds": 0.5,
        "requires_gpu": True,
    }

    # Responses: req_1 has duplicate responses, req_2 is missing entirely
    gpu_responses = [
        {"request_id": "req_1", "model_id": GPU_MODEL_ID, "request_sha256": "hash1", "status": "success", "raw_model_output": "{}"},
        {"request_id": "req_1", "model_id": GPU_MODEL_ID, "request_sha256": "hash1", "status": "success", "raw_model_output": "{}"},
    ]

    results = process_gpu_batch(gpu_responses, [manifest_1, manifest_2], registry_path=mock_registry_file)
    assert len(results) == 2

    res_1 = results[0]
    assert res_1["request_id"] == "req_1"
    assert res_1["status"] == "error"
    assert "Duplicate GPU responses" in res_1["error"]

    res_2 = results[1]
    assert res_2["request_id"] == "req_2"
    assert res_2["status"] == "error"
    assert "Missing GPU response" in res_2["error"]


# ==============================================================================
# 5. CLI Tests: scripts/prepare_gpu_batch.py & scripts/validate_gpu_batch.py
# ==============================================================================

def test_prepare_cli_rejects_tracked_output_dir(tmp_path):
    repo_src = paths.project_root / "src"
    test_in = tmp_path / "questions.jsonl"
    with open(test_in, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question": "Rate?", "role": "general"}) + "\n")

    with pytest.raises(SystemExit) as exc:
        prepare_main(["--input", str(test_in), "--output-dir", str(repo_src)])
    assert exc.value.code != 0


def test_validate_cli_rejects_tracked_output_file(tmp_path):
    repo_out = paths.project_root / "src" / "forbidden.jsonl"
    test_man = tmp_path / "manifest.jsonl"
    test_resp = tmp_path / "responses.jsonl"
    with open(test_man, "w", encoding="utf-8") as f:
        f.write("{}\n")
    with open(test_resp, "w", encoding="utf-8") as f:
        f.write("{}\n")

    with pytest.raises(SystemExit) as exc:
        validate_main(["--manifest", str(test_man), "--gpu-responses", str(test_resp), "--output", str(repo_out)])
    assert exc.value.code != 0


def test_prepare_cli_end_to_end(tmp_path, mock_registry_file, sample_chunk_gen02):
    test_in = tmp_path / "questions.jsonl"
    out_dir = tmp_path / "external_out"
    with open(test_in, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": "Q-001", "question": "What is lodging rate?", "role": "general"}) + "\n")

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.2)):
        prepare_main([
            "--input", str(test_in),
            "--output-dir", str(out_dir),
            "--registry-path", str(mock_registry_file),
        ])

    req_file = out_dir / "gpu_requests.jsonl"
    man_file = out_dir / "gpu_manifest.jsonl"
    assert req_file.exists()
    assert man_file.exists()

    with open(req_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["model_id"] == GPU_MODEL_ID


def test_validate_cli_end_to_end(tmp_path, mock_registry_file, sample_chunk_gen02):
    man_file = tmp_path / "manifest.jsonl"
    resp_file = tmp_path / "responses.jsonl"
    out_file = tmp_path / "validated.jsonl"

    from quorift.gpu_exchange import compute_request_fingerprint, build_grounded_prompt
    messages = build_grounded_prompt("Query?", [sample_chunk_gen02])
    sha = compute_request_fingerprint("req_100", GPU_MODEL_ID, messages)

    manifest_record = {
        "request_id": "req_100",
        "model_id": GPU_MODEL_ID,
        "request_sha256": sha,
        "question": "Query?",
        "role": "general",
        "question_id": "Q-001",
        "retrieved_chunks": [sample_chunk_gen02],
        "retrieval_time_seconds": 0.3,
        "requires_gpu": True,
    }
    with open(man_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_record) + "\n")

    response_record = {
        "request_id": "req_100",
        "model_id": GPU_MODEL_ID,
        "request_sha256": sha,
        "status": "success",
        "raw_model_output": json.dumps({"status": "answered", "answer": "The rate is $75.", "citations": ["GEN-02|1.0|2.0"]}),
        "generation_time_seconds": 1.2,
        "gpu_name": "Tesla T4",
    }
    with open(resp_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(response_record) + "\n")

    validate_main([
        "--manifest", str(man_file),
        "--gpu-responses", str(resp_file),
        "--output", str(out_file),
        "--registry-path", str(mock_registry_file),
    ])

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        res = json.loads(lines[0])
        assert res["status"] == "answered"
        assert res["validation_passed"] is True
