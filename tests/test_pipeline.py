"""Focused offline unit tests for the end-to-end RAG pipeline, subprocess separation, and CLI."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from quorift.config import paths
from quorift.generation import REFUSAL_MESSAGE
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
    OLLAMA_REQUEST_TIMEOUT,
)
from quorift.pipeline import (
    RETRIEVAL_WORKER_SCRIPT,
    _run_retrieval_subprocess,
    answer_question,
)
from scripts.ask import main as ask_main


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


def make_mock_http_response(body_dict: dict, status: int = 200):
    """Create a mock HTTP response object for urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body_dict).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# 1. Valid question and role produces an answered result
def test_valid_question_and_role(mock_registry_file, sample_chunk_gen02):
    mock_http = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The daily meal per diem for domestic travel is $75 per day.",
            })
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.25)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question(
                question="What is the meal allowance?",
                role="general",
                registry_path=mock_registry_file,
            )

            assert res["status"] == "answered"
            assert res["validation_passed"] is True
            assert res["answer"] == "The daily meal per diem for domestic travel is $75 per day."
            assert res["citations"] == ["GEN-02|1.0|2.0"]
            assert len(res["resolved_citations"]) == 1
            assert res["resolved_citations"][0]["document_id"] == "GEN-02"
            assert res["error"] is None


# 2. Empty question is rejected
def test_empty_question_rejected():
    with pytest.raises(ValueError, match="Question must be a nonempty string"):
        answer_question("", "general")

    with pytest.raises(ValueError, match="Question must be a nonempty string"):
        answer_question("    ", "general")


# 3. Invalid role is rejected
def test_invalid_role_rejected():
    with pytest.raises(ValueError, match="Invalid role"):
        answer_question("What is the policy?", "superadmin")

    with pytest.raises(ValueError, match="Invalid role"):
        answer_question("What is the policy?", "guest")


# 4. Retrieval subprocess executes separately
def test_retrieval_subprocess_executes_separately(tmp_path):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""

    out_file = tmp_path / "retrieval_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"chunks": []}, f)

    def fake_subprocess_run(cmd, **kwargs):
        return mock_proc

    with patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run:
        with patch("tempfile.TemporaryDirectory") as mock_tmpdir:
            mock_tmpdir.return_value.__enter__.return_value = str(tmp_path)
            mock_tmpdir.return_value.__exit__.return_value = None

            chunks, elapsed = _run_retrieval_subprocess("Test question?", "general")

            assert mock_run.call_count == 1
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == sys.executable
            assert cmd[1] == "-c"
            assert cmd[2] == RETRIEVAL_WORKER_SCRIPT
            assert isinstance(chunks, list)
            assert elapsed >= 0.0


# 5. Parent process does not import or load BGE / Qdrant
def test_parent_process_does_not_import_bge():
    check_code = (
        "import sys; "
        "import quorift.pipeline; "
        "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers imported in pipeline!'; "
        "assert 'qdrant_client' not in sys.modules, 'qdrant_client imported in pipeline!'; "
        "assert 'quorift.retrieval' not in sys.modules, 'quorift.retrieval imported in pipeline!'; "
        "print('ISOLATION_VERIFIED')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ISOLATION_VERIFIED" in proc.stdout


# 6. Correct top-5 authorized evidence transfer
def test_correct_top5_authorized_evidence_transfer(mock_registry_file, sample_chunk_gen02):
    five_chunks = []
    for i in range(1, 6):
        c = dict(sample_chunk_gen02)
        c["chunk_id"] = f"GEN-02|1.0|{i}.0"
        c["section_number"] = f"{i}.0"
        five_chunks.append(c)

    mock_http = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|1.0"],
                "answer": "Answer based on top chunks.",
            })
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=(five_chunks, 0.40)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question("Test question?", "general", registry_path=mock_registry_file)
            assert res["status"] == "answered"
            assert res["retrieved_chunks_count"] == 5


# 7. Full chunk metadata is preserved
def test_full_chunk_metadata_is_preserved(sample_chunk_gen02):
    required_keys = (
        "chunk_id",
        "document_id",
        "title",
        "section_number",
        "section_heading",
        "version",
        "source_hash",
        "page_start",
        "page_end",
        "source_pdf",
        "allowed_roles",
        "text",
    )
    for k in required_keys:
        assert k in sample_chunk_gen02, f"Missing required key '{k}' in chunk record"


# 8. Executable test verifying retrieval worker closes resources on both success and failure
def test_retrieval_worker_closes_retriever_on_success_and_failure(tmp_path):
    input_file = tmp_path / "in.json"
    output_file = tmp_path / "out.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump({"question": "test", "role": "general", "top_k": 5}, f)

    # Success execution: closes retriever cleanly
    mock_retriever_ok = MagicMock()
    mock_retriever_ok.retrieve.return_value = [{"chunk_id": "c1"}]
    mock_retriever_ok.chunks_by_id = {"c1": {"chunk_id": "c1", "text": "sample"}}

    with patch("quorift.retrieval.HybridRetriever", return_value=mock_retriever_ok):
        worker_scope = {}
        exec(compile(RETRIEVAL_WORKER_SCRIPT, "<worker>", "exec"), worker_scope)
        worker_main = worker_scope["main"]
        with patch("sys.argv", ["worker.py", str(input_file), str(output_file)]):
            worker_main()

    assert mock_retriever_ok.close.call_count == 1

    # Failure execution: closes retriever cleanly when retrieve raises
    mock_retriever_fail = MagicMock()
    mock_retriever_fail.retrieve.side_effect = RuntimeError("Index search failed")

    with patch("quorift.retrieval.HybridRetriever", return_value=mock_retriever_fail):
        worker_scope = {}
        exec(compile(RETRIEVAL_WORKER_SCRIPT, "<worker>", "exec"), worker_scope)
        worker_main = worker_scope["main"]
        with patch("sys.argv", ["worker.py", str(input_file), str(output_file)]):
            with pytest.raises(RuntimeError, match="Index search failed"):
                worker_main()

    assert mock_retriever_fail.close.call_count == 1


# 9. Missing retrieved chunk IDs cause failure and close retriever
def test_missing_retrieved_chunk_id_causes_failure(tmp_path):
    input_file = tmp_path / "in_missing.json"
    output_file = tmp_path / "out_missing.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump({"question": "test", "role": "general", "top_k": 5}, f)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"chunk_id": "missing_chunk_999"}]
    mock_retriever.chunks_by_id = {}  # Chunk ID is missing!

    with patch("quorift.retrieval.HybridRetriever", return_value=mock_retriever):
        worker_scope = {}
        exec(compile(RETRIEVAL_WORKER_SCRIPT, "<worker>", "exec"), worker_scope)
        worker_main = worker_scope["main"]
        with patch("sys.argv", ["worker.py", str(input_file), str(output_file)]):
            with pytest.raises(KeyError, match="Retrieved chunk ID 'missing_chunk_999' not found"):
                worker_main()

    # Must close retriever even on missing chunk ID error
    assert mock_retriever.close.call_count == 1


# 10. No unauthorized document reaches generation
def test_no_unauthorized_document_reaches_generation(mock_registry_file, sample_chunk_lnd01):
    # Role 'general' cannot access LND-01
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_lnd01], 0.20)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("What is commercial loan limit?", "general", registry_path=mock_registry_file)

            # Generation must NEVER be dispatched
            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert res["error"] == "Retrieved evidence failed security or integrity validation."
            assert res["answer"] == ""


# 11. Stale evidence is rejected
def test_stale_evidence_is_rejected(mock_registry_file, sample_chunk_gen02):
    stale_chunk = dict(sample_chunk_gen02)
    stale_chunk["source_hash"] = "altered_stale_hash_value"

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([stale_chunk], 0.20)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("What is the lodging limit?", "general", registry_path=mock_registry_file)

            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert res["error"] == "Retrieved evidence failed security or integrity validation."
            assert res["answer"] == ""


# 12. Empty evidence returns a safe refusal without an HTTP call
def test_empty_evidence_returns_safe_refusal_without_http(mock_registry_file):
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([], 0.15)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("What is the crypto loan policy?", "lending", registry_path=mock_registry_file)

            # No network call
            assert mock_urlopen.call_count == 0
            assert res["status"] == "insufficient_evidence"
            assert res["answer"] == REFUSAL_MESSAGE
            assert res["citations"] == []
            assert res["resolved_citations"] == []
            assert res["validation_passed"] is True
            assert res["error"] is None


# 13. Valid answer and citations are returned correctly
def test_valid_answer_and_citations_returned(mock_registry_file, sample_chunk_gen02):
    mock_http = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Tier 1 lodging is capped at INR 6,500.",
            })
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.30)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question("Lodging ceiling?", "general", registry_path=mock_registry_file)

            assert res["status"] == "answered"
            assert res["answer"] == "Tier 1 lodging is capped at INR 6,500."
            assert res["citations"] == ["GEN-02|1.0|2.0"]
            assert len(res["resolved_citations"]) == 1
            meta = res["resolved_citations"][0]
            assert meta["document_id"] == "GEN-02"
            assert meta["title"] == "Travel and Expense Policy"
            assert meta["version"] == "1.0"
            assert meta["section_number"] == "2.0"
            assert meta["page_start"] == 2
            assert meta["page_end"] == 3


# 14. Malformed model output is not displayed and does not leak raw model output in error
def test_malformed_model_output_not_displayed(mock_registry_file, sample_chunk_gen02):
    mock_http = make_mock_http_response({
        "message": {
            "content": "This response contains SECRET_PROMPT_INJECTION_OR_CONFIDENTIAL_TEXT."
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.20)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question("Lodging cap?", "general", registry_path=mock_registry_file)

            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert res["answer"] == ""
            assert res["error"] == "Model output validation failed."
            assert "SECRET" not in res["error"]
            assert "CONFIDENTIAL" not in res["error"]


# 15. Retrieval failure stops generation and does not leak raw subprocess stderr or tracebacks
def test_retrieval_failure_never_triggers_ollama(mock_registry_file):
    with patch("quorift.pipeline._run_retrieval_subprocess", side_effect=RuntimeError("Index failed with Traceback: /private/internal/db.sqlite")):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("Test query?", "general", registry_path=mock_registry_file)

            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["error"] == "Retrieval failed."
            assert "Traceback" not in res["error"]
            assert "/private/internal" not in res["error"]
            assert res["validation_passed"] is False
            assert res["answer"] == ""


# 16. Retrieval timeout error is sanitized
def test_retrieval_timeout_error_sanitized(mock_registry_file):
    with patch("quorift.pipeline._run_retrieval_subprocess", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("Test query?", "general", registry_path=mock_registry_file)

            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["error"] == "Retrieval timed out."
            assert res["generation_time_seconds"] == 0.0


# 17. Raw subprocess stderr is not exposed in public error
def test_raw_subprocess_errors_not_exposed(mock_registry_file):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Traceback (most recent call last):\nFile 'C:/internal/keys.py'\nDATABASE_PASSWORD=secret"

    with patch("subprocess.run", return_value=mock_proc):
        res = answer_question("Query?", "general", registry_path=mock_registry_file)

        assert res["status"] == "error"
        assert res["error"] == "Retrieval failed."
        assert "DATABASE_PASSWORD" not in res["error"]
        assert "secret" not in res["error"]
        assert "Traceback" not in res["error"]


# 18. Generation timeout records nonzero elapsed time and sanitized error
def test_generation_timeout_records_nonzero_elapsed_time(mock_registry_file, sample_chunk_gen02):
    def fake_ollama_call(*args, **kwargs):
        time.sleep(0.02)
        raise TimeoutError("Ollama generation request timed out after 300 seconds.")

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.20)):
        with patch("quorift.pipeline.generate_grounded_answer_ollama", side_effect=fake_ollama_call):
            res = answer_question("Lodging?", "general", registry_path=mock_registry_file)

            assert res["status"] == "error"
            assert res["error"] == "Generation timed out."
            assert res["generation_time_seconds"] > 0.0
            assert res["total_time_seconds"] >= res["retrieval_time_seconds"] + res["generation_time_seconds"]
            assert res["answer"] == ""


# 19. Generation service unavailable error is sanitized
def test_generation_service_unavailable_sanitized(mock_registry_file, sample_chunk_gen02):
    def fake_ollama_conn_err(*args, **kwargs):
        raise ConnectionError("Could not connect to local Ollama service at http://127.0.0.1:11434")

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.20)):
        with patch("quorift.pipeline.generate_grounded_answer_ollama", side_effect=fake_ollama_conn_err):
            res = answer_question("Lodging?", "general", registry_path=mock_registry_file)

            assert res["status"] == "error"
            assert res["error"] == "Generation service unavailable."
            assert "http://" not in res["error"]
            assert "127.0.0.1" not in res["error"]


# 20. Prevent accidental model substitution in public pipeline API
def test_unauthorized_model_substitution_rejected(mock_registry_file):
    with pytest.raises(ValueError, match="Model 'llama3:8b' is not permitted"):
        answer_question("Query?", "general", model_id="llama3:8b", registry_path=mock_registry_file)

    with pytest.raises(ValueError, match="Model 'gpt-4' is not permitted"):
        answer_question("Query?", "general", model_id="gpt-4", registry_path=mock_registry_file)


# 21. Temporary files are cleaned up on success and error
def test_temporary_files_cleaned_up_on_success_and_error(tmp_path):
    created_dirs = []
    real_tempdir = tempfile.TemporaryDirectory

    class TrackingTempDir:
        def __init__(self, *args, **kwargs):
            self.temp_dir = real_tempdir(*args, **kwargs)

        def __enter__(self):
            path = self.temp_dir.__enter__()
            created_dirs.append(path)
            return path

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.temp_dir.__exit__(exc_type, exc_val, exc_tb)

    # Success case
    mock_proc_ok = MagicMock()
    mock_proc_ok.returncode = 0

    def fake_run_ok(cmd, **kwargs):
        out_path = Path(cmd[4])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": []}, f)
        return mock_proc_ok

    with patch("tempfile.TemporaryDirectory", TrackingTempDir):
        with patch("subprocess.run", side_effect=fake_run_ok):
            _run_retrieval_subprocess("Test question?", "general")

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()

    # Error case (worker fails with exception)
    with patch("tempfile.TemporaryDirectory", TrackingTempDir):
        with patch("subprocess.run", side_effect=RuntimeError("Subprocess failed")):
            with pytest.raises(RuntimeError):
                _run_retrieval_subprocess("Test question?", "general")

    assert len(created_dirs) == 2
    assert not Path(created_dirs[1]).exists()


# 22. Timing fields are correct
def test_timing_fields_are_correct(mock_registry_file, sample_chunk_gen02):
    mock_http = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Sample answer.",
            })
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.35)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question("Lodging?", "general", registry_path=mock_registry_file)

            assert res["retrieval_time_seconds"] == 0.35
            assert res["generation_time_seconds"] >= 0.0
            assert res["total_time_seconds"] >= res["retrieval_time_seconds"]


# 23. Existing Qwen model configuration remains unchanged
def test_existing_qwen_model_configuration_unchanged():
    assert OLLAMA_MODEL_ID == "qwen3:4b-instruct-2507-q4_K_M"
    assert OLLAMA_API_URL == "http://127.0.0.1:11434/api/chat"
    assert OLLAMA_REQUEST_TIMEOUT == 300


# 24. CLI ask.py execution test
def test_ask_cli_smoke(sample_chunk_gen02, capsys):
    mock_result = {
        "status": "answered",
        "answer": "The overnight lodging ceiling for Tier 1 Metros is capped at INR 6,500 per night.",
        "citations": ["GEN-02|1.0|2.0"],
        "resolved_citations": [
            {
                "chunk_id": "GEN-02|1.0|2.0",
                "document_id": "GEN-02",
                "title": "Travel and Expense Policy",
                "version": "1.0",
                "section_number": "2.0",
                "page_start": 1,
                "page_end": 1,
            }
        ],
        "retrieval_time_seconds": 0.85,
        "generation_time_seconds": 15.20,
        "total_time_seconds": 16.05,
        "validation_passed": True,
        "error": None,
        "model_id": OLLAMA_MODEL_ID,
        "retrieved_chunks_count": 1,
    }

    with patch("scripts.ask.answer_question", return_value=mock_result):
        ask_main(["--role", "general", "--question", "What is Tier 1 lodging limit?"])
        captured = capsys.readouterr()
        assert "INR 6,500" in captured.out
        assert "GEN-02 — Travel and Expense Policy" in captured.out
        assert "Retrieval:  0.85s" in captured.out
        assert "Generation: 15.20s" in captured.out
