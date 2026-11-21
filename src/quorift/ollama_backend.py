"""Local Ollama backend for grounded answer generation using Qwen."""

import json
from pathlib import Path
import socket
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

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

OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL_ID = "qwen3:4b-instruct-2507-q4_K_M"
OLLAMA_REQUEST_TIMEOUT = 300  # 5 minutes for CPU inference


def call_ollama_chat(
    messages: list[dict[str, str]],
    model_id: str = OLLAMA_MODEL_ID,
    api_url: str = OLLAMA_API_URL,
    timeout: int = OLLAMA_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Execute chat completion request against local Ollama service with fixed parameters."""
    parsed_url = urllib.parse.urlparse(api_url)
    if parsed_url.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(
            f"Security policy: only local Ollama endpoint (127.0.0.1) is permitted, got '{api_url}'"
        )

    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 320,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            if status_code != 200:
                raise RuntimeError(f"Ollama returned unexpected HTTP status {status_code}")
            resp_bytes = resp.read()
            return json.loads(resp_bytes.decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass

        if e.code == 404 or "model" in err_body.lower():
            raise RuntimeError(
                f"Ollama model '{model_id}' was not found. "
                f"Please pull the required model first using: ollama pull {model_id}"
            ) from e
        raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason}. Details: {err_body}") from e
    except (TimeoutError, socket.timeout) as e:
        raise TimeoutError(
            f"Ollama generation request timed out after {timeout} seconds. "
            f"Model '{model_id}' inference took longer than permitted."
        ) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e.reason).lower():
            raise TimeoutError(
                f"Ollama generation request timed out after {timeout} seconds. "
                f"Model '{model_id}' inference took longer than permitted."
            ) from e
        raise ConnectionError(
            f"Could not connect to local Ollama service at {api_url}. "
            f"Please ensure Ollama is installed and running (`ollama serve`). "
            f"Underlying error: {e.reason}"
        ) from e


def generate_grounded_answer_ollama(
    question: str,
    role: str,
    retrieved_chunks: list[dict[str, Any]],
    registry: dict[str, Any],
    model_id: str = OLLAMA_MODEL_ID,
    api_url: str = OLLAMA_API_URL,
    timeout: int = OLLAMA_REQUEST_TIMEOUT,
    question_id: str | None = None,
) -> dict[str, Any]:
    """Validate request, send prompt to local Ollama, parse and validate structured citations."""
    t0 = time.perf_counter()

    # 1. Validate incoming generation request
    request_data = {
        "question": question,
        "role": role,
        "retrieved_chunks": retrieved_chunks,
    }
    validate_generation_request(request_data, registry)

    # 2. Check empty evidence: refuse immediately without making an HTTP request
    if len(retrieved_chunks) == 0:
        empty_resp = generate_empty_evidence_response()
        elapsed = time.perf_counter() - t0
        return {
            "question_id": question_id,
            "model_id": model_id,
            "raw_model_output": json.dumps(empty_resp),
            "parsed_status": empty_resp["status"],
            "parsed_answer": empty_resp["answer"],
            "validated_citations": empty_resp["citations"],
            "resolved_citations": [],
            "validation_passed": True,
            "validation_error": None,
            "generation_time_seconds": elapsed,
            "total_duration_ns": None,
            "load_duration_ns": None,
            "prompt_eval_duration_ns": None,
            "eval_duration_ns": None,
            "total_duration_seconds": None,
            "load_duration_seconds": None,
            "prompt_eval_duration_seconds": None,
            "eval_duration_seconds": None,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "eval_tokens_per_second": None,
        }

    # 3. Construct chat messages using shared grounded prompt logic
    messages = build_grounded_prompt(question, retrieved_chunks)

    # 4. Execute HTTP request to local Ollama
    raw_response = call_ollama_chat(
        messages=messages,
        model_id=model_id,
        api_url=api_url,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0

    # 5. Extract assistant message content
    if not isinstance(raw_response, dict):
        raise ValueError(f"Malformed Ollama response, expected dict: {type(raw_response).__name__}")

    message_obj = raw_response.get("message")
    if not isinstance(message_obj, dict):
        raise ValueError("Ollama response missing 'message' object")

    raw_text = message_obj.get("content")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Ollama assistant message contains empty or non-string content")

    # 6. Parse and validate generation output
    validation_passed = False
    validation_error = None
    parsed_status = "error"
    parsed_answer = ""
    validated_citations: list[str] = []
    resolved_citations: list[dict[str, Any]] = []

    try:
        parsed = parse_generation_response(raw_text)
        validated = validate_generation_output(parsed, retrieved_chunks, registry, role)
        validation_passed = True
        parsed_status = validated["status"]
        parsed_answer = validated["answer"]
        validated_citations = validated["citations"]
        resolved_citations = [
            resolve_citation_metadata(cid, retrieved_chunks) for cid in validated_citations
        ]
    except Exception as e:
        validation_passed = False
        validation_error = str(e)

    total_duration_ns = raw_response.get("total_duration")
    load_duration_ns = raw_response.get("load_duration")
    prompt_eval_duration_ns = raw_response.get("prompt_eval_duration")
    eval_duration_ns = raw_response.get("eval_duration")
    prompt_eval_count = raw_response.get("prompt_eval_count")
    eval_count = raw_response.get("eval_count")

    total_duration_seconds = (
        round(total_duration_ns / 1e9, 4) if isinstance(total_duration_ns, (int, float)) else None
    )
    load_duration_seconds = (
        round(load_duration_ns / 1e9, 4) if isinstance(load_duration_ns, (int, float)) else None
    )
    prompt_eval_duration_seconds = (
        round(prompt_eval_duration_ns / 1e9, 4) if isinstance(prompt_eval_duration_ns, (int, float)) else None
    )
    eval_duration_seconds = (
        round(eval_duration_ns / 1e9, 4) if isinstance(eval_duration_ns, (int, float)) else None
    )

    eval_tokens_per_second = None
    if (
        isinstance(eval_count, (int, float))
        and isinstance(eval_duration_ns, (int, float))
        and eval_duration_ns > 0
    ):
        eval_tokens_per_second = round(eval_count / (eval_duration_ns / 1e9), 2)

    return {
        "question_id": question_id,
        "model_id": model_id,
        "raw_model_output": raw_text,
        "parsed_status": parsed_status,
        "parsed_answer": parsed_answer,
        "validated_citations": validated_citations,
        "resolved_citations": resolved_citations,
        "validation_passed": validation_passed,
        "validation_error": validation_error,
        "generation_time_seconds": elapsed,
        "total_duration_ns": total_duration_ns,
        "load_duration_ns": load_duration_ns,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "eval_duration_ns": eval_duration_ns,
        "total_duration_seconds": total_duration_seconds,
        "load_duration_seconds": load_duration_seconds,
        "prompt_eval_duration_seconds": prompt_eval_duration_seconds,
        "eval_duration_seconds": eval_duration_seconds,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "eval_tokens_per_second": eval_tokens_per_second,
    }
