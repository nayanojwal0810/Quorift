"""Tests for local Ollama generation backend, performance metrics, and security guardrails."""

import io
import json
from pathlib import Path
import subprocess
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from quorift.generation import REFUSAL_MESSAGE, SYSTEM_PROMPT
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
    OLLAMA_REQUEST_TIMEOUT,
    call_ollama_chat,
    generate_grounded_answer_ollama,
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


def make_mock_http_response(body_dict: dict, status: int = 200):
    """Create a mock HTTP response object for urllib.request.urlopen context manager."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(body_dict).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


# 1. Default model is qwen3:4b-instruct-2507-q4_K_M
def test_default_model_remains_qwen3_4b():
    assert OLLAMA_MODEL_ID == "qwen3:4b-instruct-2507-q4_K_M"


# 2. call_ollama_chat passes the correct model identifier
def test_call_ollama_chat_passes_correct_model_id():
    messages = [{"role": "user", "content": "Hello"}]
    mock_resp = make_mock_http_response({"message": {"content": "{}"}})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        call_ollama_chat(messages)

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "qwen3:4b-instruct-2507-q4_K_M"
        assert payload["model"] == OLLAMA_MODEL_ID


# 3. call_ollama_chat connects to http://127.0.0.1:11434/api/chat
def test_call_ollama_chat_connects_to_local_endpoint():
    messages = [{"role": "user", "content": "Hello"}]
    mock_resp = make_mock_http_response({"message": {"content": "{}"}})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        call_ollama_chat(messages)

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:11434/api/chat"
        assert req.get_method() == "POST"
        assert req.headers["Content-type"] == "application/json"


# 4. call_ollama_chat sets JSON format mode and correct generation options
def test_call_ollama_chat_payload_options():
    messages = [{"role": "user", "content": "Hello"}]
    mock_resp = make_mock_http_response({"message": {"content": "{}"}})

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        call_ollama_chat(messages)

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert payload["options"]["temperature"] == 0
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["num_predict"] == 320


# 5. call_ollama_chat enforces local endpoint security policy
def test_call_ollama_chat_enforces_local_endpoint():
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Security policy: only local Ollama endpoint"):
        call_ollama_chat(messages, api_url="http://external-cloud-api.com/api/chat")

    with pytest.raises(ValueError, match="Security policy: only local Ollama endpoint"):
        call_ollama_chat(messages, api_url="http://192.168.1.50:11434/api/chat")


# 6. Generation reuses SYSTEM_PROMPT from generation.py
def test_generation_reuses_system_prompt(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The meal per diem is $75 per day.",
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        generate_grounded_answer_ollama(
            question="What is the meal allowance?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT


# 7. Authorized evidence is formatted and sent to Ollama
def test_authorized_evidence_formatted_and_sent(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The meal per diem is $75 per day.",
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        generate_grounded_answer_ollama(
            question="What is the meal allowance?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        user_message = payload["messages"][1]["content"]
        assert "Chunk ID: GEN-02|1.0|2.0" in user_message
        assert "Travel and Expense Policy" in user_message
        assert "Section: 2.0 - Standard Per Diem Rates" in user_message
        assert "$75 per day" in user_message
        assert "What is the meal allowance?" in user_message


# 8. Unauthorized evidence is rejected before HTTP request is made (fails closed)
def test_unauthorized_evidence_rejected_before_http(sample_registry, sample_chunk_lnd01):
    with patch("urllib.request.urlopen") as mock_urlopen:
        # role "finance" is not authorized for LND-01
        with pytest.raises(ValueError, match="Unauthorized evidence"):
            generate_grounded_answer_ollama(
                question="What is the commercial loan limit?",
                role="finance",
                retrieved_chunks=[sample_chunk_lnd01],
                registry=sample_registry,
            )

        # HTTP request must never have been dispatched
        assert mock_urlopen.call_count == 0


# 9. Empty evidence short-circuits without an HTTP request and returns generate_empty_evidence_response()
def test_empty_evidence_short_circuits_without_http(sample_registry):
    with patch("urllib.request.urlopen") as mock_urlopen:
        res = generate_grounded_answer_ollama(
            question="What is policy for unauthorized user?",
            role="hr",
            retrieved_chunks=[],
            registry=sample_registry,
            question_id="Q-EMPTY",
        )

        # No HTTP call
        assert mock_urlopen.call_count == 0
        assert res["validation_passed"] is True
        assert res["parsed_status"] == "insufficient_evidence"
        assert res["parsed_answer"] == REFUSAL_MESSAGE
        assert res["validated_citations"] == []
        assert res["resolved_citations"] == []
        assert res["question_id"] == "Q-EMPTY"


# 10. Valid grounded answer JSON is parsed and accepted
def test_valid_grounded_answer_parsed_and_accepted(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["GEN-02|1.0|2.0"],
                "answer": "The daily meal per diem for domestic travel is $75 per day.",
            })
        },
        "prompt_eval_count": 120,
        "eval_count": 25,
        "total_duration": 1500000000,
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is the daily meal per diem?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
            question_id="Q-001",
        )

        assert res["validation_passed"] is True
        assert res["validation_error"] is None
        assert res["parsed_status"] == "answered"
        assert res["parsed_answer"] == "The daily meal per diem for domestic travel is $75 per day."
        assert res["validated_citations"] == ["GEN-02|1.0|2.0"]
        assert len(res["resolved_citations"]) == 1
        assert res["resolved_citations"][0]["document_id"] == "GEN-02"
        assert res["resolved_citations"][0]["section_number"] == "2.0"
        assert res["prompt_eval_count"] == 120
        assert res["eval_count"] == 25


# 11. Valid exact refusal JSON is parsed and accepted
def test_valid_exact_refusal_accepted(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "insufficient_evidence",
                "citations": [],
                "answer": REFUSAL_MESSAGE,
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is the bonus pool formula?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert res["validation_passed"] is True
        assert res["validation_error"] is None
        assert res["parsed_status"] == "insufficient_evidence"
        assert res["parsed_answer"] == REFUSAL_MESSAGE
        assert res["validated_citations"] == []
        assert res["resolved_citations"] == []


# 12. Refusal containing unsupported text is rejected
def test_refusal_with_unsupported_text_rejected(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "insufficient_evidence",
                "citations": [],
                "answer": "I do not have enough information to answer this question about bonuses.",
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is the bonus pool?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert res["validation_passed"] is False
        assert "must exactly match the approved refusal message" in res["validation_error"]


# 13. Fabricated citation in model response is rejected
def test_fabricated_citation_rejected(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": json.dumps({
                "status": "answered",
                "citations": ["FABRICATED-99|1.0|1.0"],
                "answer": "Here is an answer citing an unretrieved chunk.",
            })
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is the rule?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert res["validation_passed"] is False
        assert "was not present in the authorized evidence" in res["validation_error"]


# 14. Ollama service unavailable (URLError) raises clear connection error advising ollama serve
def test_service_unavailable_raises_connection_error():
    messages = [{"role": "user", "content": "Hello"}]
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(ConnectionError, match="ollama serve"):
            call_ollama_chat(messages)


# 15. Model missing (HTTP 404 or missing model message) raises clear missing-model error advising ollama pull
def test_missing_model_raises_runtime_error():
    messages = [{"role": "user", "content": "Hello"}]
    http_err = urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b'{"error": "model \'qwen3:4b-instruct-2507-q4_K_M\' not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError, match="ollama pull"):
            call_ollama_chat(messages)


# 16. Malformed HTTP response (non-JSON body, missing message object) raises ValueError
def test_malformed_http_response_handling(sample_registry, sample_chunk_gen02):
    # Non-JSON body
    mock_resp_bad_json = MagicMock()
    mock_resp_bad_json.status = 200
    mock_resp_bad_json.read.return_value = b"<html>Server Error</html>"
    mock_resp_bad_json.__enter__.return_value = mock_resp_bad_json
    mock_resp_bad_json.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp_bad_json):
        with pytest.raises(json.JSONDecodeError):
            call_ollama_chat([{"role": "user", "content": "test"}])

    # Missing message object in Ollama response
    mock_resp_missing_msg = make_mock_http_response({"done": True})
    with patch("urllib.request.urlopen", return_value=mock_resp_missing_msg):
        with pytest.raises(ValueError, match="missing 'message' object"):
            generate_grounded_answer_ollama(
                question="Test question",
                role="general",
                retrieved_chunks=[sample_chunk_gen02],
                registry=sample_registry,
            )

    # Empty content in assistant message
    mock_resp_empty_content = make_mock_http_response({"message": {"role": "assistant", "content": "   "}})
    with patch("urllib.request.urlopen", return_value=mock_resp_empty_content):
        with pytest.raises(ValueError, match="empty or non-string content"):
            generate_grounded_answer_ollama(
                question="Test question",
                role="general",
                retrieved_chunks=[sample_chunk_gen02],
                registry=sample_registry,
            )


# 17. Malformed model JSON (assistant message content is not valid JSON) sets validation_passed=False
def test_malformed_model_json_content(sample_registry, sample_chunk_gen02):
    mock_resp = make_mock_http_response({
        "message": {
            "content": "This response is plain English, not valid JSON at all."
        }
    })

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = generate_grounded_answer_ollama(
            question="What is the meal allowance?",
            role="general",
            retrieved_chunks=[sample_chunk_gen02],
            registry=sample_registry,
        )

        assert res["validation_passed"] is False
        assert "Malformed JSON in model output" in res["validation_error"]
        assert res["parsed_status"] == "error"


# 18. HTTP timeout raises TimeoutError
def test_http_timeout_raises_timeout_error():
    messages = [{"role": "user", "content": "Hello"}]
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(TimeoutError, match="timed out after"):
            call_ollama_chat(messages, timeout=10)


# 19. Ollama performance metrics are captured correctly
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


# 20. Missing duration fields remain null
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


# 21. Generation results are ignored by git
def test_generation_results_git_ignored():
    """Verify data/.generation/ollama_results.jsonl matches gitignore rules."""
    res = subprocess.run(
        ["git", "check-ignore", "data/.generation/ollama_results.jsonl"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "data/.generation/" in res.stdout
