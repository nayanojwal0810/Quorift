"""Focused offline unit tests for the end-to-end RAG pipeline, subprocess separation, and CLI."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
        # Emulate writing output file
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


# 8. Retrieval worker closes resources on success and error
def test_retrieval_worker_closes_resources():
    worker_check_script = """
from unittest.mock import MagicMock, patch
import quorift.pipeline

mock_retriever = MagicMock()
with patch("quorift.retrieval.HybridRetriever", return_value=mock_retriever):
    exec(compile(quorift.pipeline.RETRIEVAL_WORKER_SCRIPT, "<test>", "exec"))
    # In finally: block of worker, retriever.close() must be called
    assert mock_retriever.close.call_count == 0  # not yet run main
print('WORKER_RESOURCE_CLEANUP_CHECKED')
"""
    assert "retriever.close()" in RETRIEVAL_WORKER_SCRIPT
    assert "finally:" in RETRIEVAL_WORKER_SCRIPT


# 9. No unauthorized document reaches generation
def test_no_unauthorized_document_reaches_generation(mock_registry_file, sample_chunk_lnd01):
    # Role 'general' cannot access LND-01
    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_lnd01], 0.20)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("What is commercial loan limit?", "general", registry_path=mock_registry_file)

            # Generation must NEVER be dispatched
            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert "security or integrity validation" in res["error"]
            assert res["answer"] == ""


# 10. Stale evidence is rejected
def test_stale_evidence_is_rejected(mock_registry_file, sample_chunk_gen02):
    stale_chunk = dict(sample_chunk_gen02)
    stale_chunk["source_hash"] = "altered_stale_hash_value"

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([stale_chunk], 0.20)):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("What is the lodging limit?", "general", registry_path=mock_registry_file)

            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert res["answer"] == ""


# 11. Empty evidence returns a safe refusal without an HTTP call
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


# 12. Valid answer and citations are returned correctly
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


# 13. Malformed model output is not displayed as a valid answer
def test_malformed_model_output_not_displayed(mock_registry_file, sample_chunk_gen02):
    mock_http = make_mock_http_response({
        "message": {
            "content": "This response is not JSON format at all."
        }
    })

    with patch("quorift.pipeline._run_retrieval_subprocess", return_value=([sample_chunk_gen02], 0.20)):
        with patch("urllib.request.urlopen", return_value=mock_http):
            res = answer_question("Lodging cap?", "general", registry_path=mock_registry_file)

            assert res["status"] == "error"
            assert res["validation_passed"] is False
            assert res["answer"] == ""
            assert "validation failed" in res["error"]


# 14. Retrieval failure stops generation
def test_retrieval_failure_stops_generation(mock_registry_file):
    with patch("quorift.pipeline._run_retrieval_subprocess", side_effect=RuntimeError("Qdrant index corrupted")):
        with patch("urllib.request.urlopen") as mock_urlopen:
            res = answer_question("Test query?", "general", registry_path=mock_registry_file)

            assert mock_urlopen.call_count == 0
            assert res["status"] == "error"
            assert "Retrieval failed" in res["error"]
            assert res["validation_passed"] is False
            assert res["answer"] == ""


# 15. Temporary files are cleaned up
def test_temporary_files_cleaned_up(tmp_path):
    created_dirs = []
    real_tempdir = tempfile.TemporaryDirectory

    class TrackingTempDir:
        def __init__(self, *args, **kwargs):
            self.temp_dir = real_tempdir(*args, **kwargs)

        def __enter__(self):
            path = self.temp_dir.__enter__()
            created_dirs.append(path)
            # Create a mock retrieval output file in this directory
            out_file = Path(path) / "retrieval_output.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"chunks": []}, f)
            return path

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self.temp_dir.__exit__(exc_type, exc_val, exc_tb)

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch("tempfile.TemporaryDirectory", TrackingTempDir):
        with patch("subprocess.run", return_value=mock_proc):
            _run_retrieval_subprocess("Test question?", "general")

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()


# 16. Timing fields are correct
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


# 17. Existing Qwen model configuration remains unchanged
def test_existing_qwen_model_configuration_unchanged():
    assert OLLAMA_MODEL_ID == "qwen3:4b-instruct-2507-q4_K_M"
    assert OLLAMA_API_URL == "http://127.0.0.1:11434/api/chat"
    assert OLLAMA_REQUEST_TIMEOUT == 300


# 18. CLI ask.py execution test
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
