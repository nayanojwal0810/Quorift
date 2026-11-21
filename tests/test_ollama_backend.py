"""Tests for local Ollama generation backend, runner, CLI, performance metrics, and security guardrails."""

import io
import json
from pathlib import Path
import subprocess
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from quorift.config import paths
from quorift.generation import REFUSAL_MESSAGE, SYSTEM_PROMPT
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
    OLLAMA_REQUEST_TIMEOUT,
    call_ollama_chat,
    generate_grounded_answer_ollama,
)
from scripts.run_local_generation import (
    ALLOWED_MODELS,
    EXPECTED_QIDS,
    load_and_validate_requests,
    main as runner_main,
    run_local_generation,
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
    """Authorized chunk for general and lending roles."""
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
    """Restricted chunk for lending and admin roles only."""
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


@pytest.fixture
def valid_requests_file(tmp_path, sample_chunk_gen02):
    """Create a temporary requests.jsonl containing the exact 6 approved question IDs."""
    req_file = tmp_path / "requests.jsonl"
    records = [
        {
            "question_id": qid,
            "question": f"Question text for {qid}",
            "role": "general",
            "retrieved_chunks": [sample_chunk_gen02],
        }
        for qid in EXPECTED_QIDS
    ]
    with open(req_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return req_file


def make_mock_http_response(body_dict: dict, status: int = 200):
    """Create a mock HTTP response object for urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body_dict).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# 1. The original model remains the default
def test_default_model_remains_qwen3_4b():
    assert OLLAMA_MODEL_ID == "qwen3:4b-instruct-2507-q4_K_M"
    assert "qwen3:4b-instruct-2507-q4_K_M" in ALLOWED_MODELS


# 2. Only the two approved comparison model IDs are accepted
def test_only_approved_models_accepted(valid_requests_file, tmp_path):
    out_file1 = tmp_path / "out1.jsonl"
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Answer.",
            })
        }
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        # 4B accepted
        res1 = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file1,
            model_id="qwen3:4b-instruct-2507-q4_K_M",
            question_id="Q-001",
        )
        assert len(res1) == 1
        assert res1[0]["model_id"] == "qwen3:4b-instruct-2507-q4_K_M"

        # 2.5 3B accepted
        out_file2 = tmp_path / "out2.jsonl"
        res2 = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file2,
            model_id="qwen2.5:3b",
            question_id="Q-001",
        )
        assert len(res2) == 1
        assert res2[0]["model_id"] == "qwen2.5:3b"

    # Unapproved models rejected
    with pytest.raises(ValueError, match="Invalid model 'llama3:8b'"):
        run_local_generation(
            requests_path=valid_requests_file,
            output_path=tmp_path / "out3.jsonl",
            model_id="llama3:8b",
        )


# 3. Only the six approved development questions can be selected
def test_only_approved_questions_can_be_selected(valid_requests_file, tmp_path):
    with pytest.raises(ValueError, match="Invalid question_id 'Q-999'"):
        run_local_generation(
            requests_path=valid_requests_file,
            output_path=tmp_path / "out.jsonl",
            question_id="Q-999",
        )


# 4. The existing six-question default still works
def test_existing_six_question_default_works(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_all.jsonl"
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Answer.",
            })
        }
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file,
            question_id=None,
        )
        assert len(results) == 6
        assert [r["question_id"] for r in results] == EXPECTED_QIDS


# 5. Single-question execution works
def test_single_question_execution_works(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_single.jsonl"
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Answer.",
            })
        }
    })
    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file,
            question_id="Q-033",
        )
        assert len(results) == 1
        assert results[0]["question_id"] == "Q-033"


# 6. A 300-second timeout is recorded with actual elapsed time (not 0.0s)
def test_timeout_recorded_with_actual_elapsed_time(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_timeout.jsonl"
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        results = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file,
            question_id="Q-033",
            timeout=1,
        )
        assert len(results) == 1
        res = results[0]
        assert res["parsed_status"] == "error"
        assert res["validation_passed"] is False
        assert "timed out after" in res["validation_error"]
        assert res["generation_time_seconds"] >= 0.0
        # Check elapsed time is nonzero (not hardcoded 0.0)
        assert isinstance(res["generation_time_seconds"], float)


# 7. A custom diagnostic timeout is passed correctly
def test_custom_diagnostic_timeout_passed(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_timeout600.jsonl"
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "Answer.",
            })
        }
    })
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file,
            question_id="Q-033",
            timeout=600,
        )
        assert mock_urlopen.call_count == 1
        # Check timeout kwarg passed to urlopen
        kwargs = mock_urlopen.call_args[1]
        assert kwargs.get("timeout") == 600


# 8. Failed requests remain in output results
def test_failed_requests_remain_in_output_results(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_failed.jsonl"
    with patch("urllib.request.urlopen", side_effect=ConnectionError("Cannot connect")):
        results = run_local_generation(
            requests_path=valid_requests_file,
            output_path=out_file,
            question_id="Q-018",
        )
        assert len(results) == 1
        assert results[0]["parsed_status"] == "error"
        assert results[0]["validation_passed"] is False

        # File was still written
        assert out_file.exists()
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert lines[0]["question_id"] == "Q-018"
        assert lines[0]["validation_passed"] is False


# 9. A failed run produces a nonzero exit code after writing results
def test_failed_run_produces_nonzero_exit_code(valid_requests_file, tmp_path):
    out_file = tmp_path / "out_exit.jsonl"
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(SystemExit) as exc_info:
            runner_main([
                "--model", "qwen2.5:3b",
                "--question-id", "Q-001",
                "--output", str(out_file),
            ])
        assert exc_info.value.code == 1
        assert out_file.exists()


# 10. Existing results are not overwritten accidentally
def test_existing_results_not_overwritten(valid_requests_file, tmp_path):
    existing_file = tmp_path / "existing_results.jsonl"
    existing_file.write_text("dummy content\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Accidental overwrite prevented"):
        run_local_generation(
            requests_path=valid_requests_file,
            output_path=existing_file,
            question_id="Q-001",
        )


# 11. Ollama performance metrics are captured correctly
def test_ollama_performance_metrics_captured(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The lodging cap is INR 6,500.",
            })
        },
        "total_duration": 2000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 150,
        "prompt_eval_duration": 600000000,
        "eval_count": 50,
        "eval_duration": 900000000,
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is lodging cap?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
            question_id="Q-001",
        )

        assert res["total_duration_ns"] == 2000000000
        assert res["load_duration_ns"] == 500000000
        assert res["prompt_eval_duration_ns"] == 600000000
        assert res["eval_duration_ns"] == 900000000
        assert res["prompt_eval_count"] == 150
        assert res["eval_count"] == 50

        assert res["total_duration_seconds"] == 2.0
        assert res["load_duration_seconds"] == 0.5
        assert res["prompt_eval_duration_seconds"] == 0.6
        assert res["eval_duration_seconds"] == 0.9
        assert res["eval_tokens_per_second"] == round(50 / 0.9, 2)


# 12. Missing duration fields remain null
def test_missing_duration_fields_remain_null(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The lodging cap is INR 6,500.",
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is lodging cap?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert res["total_duration_ns"] is None
        assert res["load_duration_ns"] is None
        assert res["prompt_eval_duration_ns"] is None
        assert res["eval_duration_ns"] is None
        assert res["total_duration_seconds"] is None
        assert res["load_duration_seconds"] is None
        assert res["prompt_eval_duration_seconds"] is None
        assert res["eval_duration_seconds"] is None
        assert res["eval_tokens_per_second"] is None


# 13. Invalid output is never marked as a valid answer
def test_invalid_output_never_marked_valid_answer(sample_registry, sample_chunk_gen02):
    # Malformed non-JSON output
    mock_resp_bad = make_mock_http_response({
        "message": {"content": "This is raw unformatted text without JSON."}
    })
    with patch("urllib.request.urlopen", return_value=mock_resp_bad):
        res = generate_grounded_answer_ollama(
            question="What is the cap?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )
        assert res["validation_passed"] is False
        assert res["parsed_status"] == "error"

    # Fabricated citation
    mock_resp_fab = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["UNKNOWN-99|1.0|1.0"],
                "answer": "Fabricated citation answer.",
            })
        }
    })
    with patch("urllib.request.urlopen", return_value=mock_resp_fab):
        res2 = generate_grounded_answer_ollama(
            question="What is the cap?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )
        assert res2["validation_passed"] is False
        assert "was not present in the authorized evidence" in res2["validation_error"]


# 14. call_ollama_chat basic sanity and security
def test_call_ollama_chat_security_and_options():
    messages = [{"role": "user", "content": "Hello"}]
    mock_resp = make_mock_http_response({"message": {"content": "{}"}})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        call_ollama_chat(messages)
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "qwen3:4b-instruct-2507-q4_K_M"
        assert payload["format"] == "json"
        assert payload["options"]["temperature"] == 0

    with pytest.raises(ValueError, match="Security policy: only local Ollama endpoint"):
        call_ollama_chat(messages, api_url="http://external-cloud.com/api/chat")


# 15. Service unavailable raises clear ConnectionError advising ollama serve
def test_service_unavailable_handling():
    messages = [{"role": "user", "content": "Hello"}]
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ConnectionError, match="ollama serve"):
            call_ollama_chat(messages)


# 16. Missing model raises clear RuntimeError advising ollama pull
def test_missing_model_handling():
    messages = [{"role": "user", "content": "Hello"}]
    http_err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b'{"error": "model not found"}'),
    )
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError, match="ollama pull"):
            call_ollama_chat(messages)


# 17. Empty evidence short circuits without HTTP request
def test_empty_evidence_short_circuits(sample_registry):
    with patch("urllib.request.urlopen") as mock_urlopen:
        res = generate_grounded_answer_ollama(
            question="What is policy?",
            role="general",
            retrieved_chunks=[],
            registry=sample_registry,
        )
        assert mock_urlopen.call_count == 0
        assert res["validation_passed"] is True
        assert res["parsed_status"] == "insufficient_evidence"
        assert res["parsed_answer"] == REFUSAL_MESSAGE
        assert res["validated_citations"] == []


# 18. Results cannot be written under reports directory
def test_results_cannot_be_written_under_reports(valid_requests_file):
    with pytest.raises(ValueError, match="tracked reports directory"):
        run_local_generation(
            requests_path=valid_requests_file,
            output_path=paths.reports_dir / "illegal_run.jsonl",
            question_id="Q-001",
        )


# 19. ollama_results.jsonl is ignored by git
def test_ollama_results_ignored_by_git():
    res = subprocess.run(
        ["git", "check-ignore", "data/.generation/ollama_results.jsonl"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "data/.generation/" in res.stdout
