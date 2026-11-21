"""Local generation runner executing Qwen inference via Ollama on prepared development requests."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from quorift.config import paths
from quorift.generation import validate_generation_request
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
    OLLAMA_REQUEST_TIMEOUT,
    generate_grounded_answer_ollama,
)

EXPECTED_QIDS = [
    "Q-001",
    "Q-018",
    "Q-025",
    "Q-033",
    "Q-037",
    "Q-041",
]

ALLOWED_MODELS = (
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen2.5:3b",
)


def load_and_validate_requests(
    requests_path: Path,
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load requests.jsonl and strictly validate question IDs, order, and evidence permissions."""
    if not requests_path.exists():
        raise FileNotFoundError(
            f"Generation requests file not found: {requests_path}\n"
            f"Please run 'python -m scripts.prepare_generation_inputs' first."
        )

    records = []
    with open(requests_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {requests_path} at line {line_no}: {e}") from e

    if len(records) != len(EXPECTED_QIDS):
        raise ValueError(
            f"Expected exactly {len(EXPECTED_QIDS)} question requests, found {len(records)}"
        )

    actual_qids = [r.get("question_id") for r in records]
    if actual_qids != EXPECTED_QIDS:
        raise ValueError(
            f"Question ID sequence mismatch:\nExpected: {EXPECTED_QIDS}\nActual:   {actual_qids}"
        )

    # Validate each request and its retrieved chunks against the trusted registry
    for r in records:
        validate_generation_request(r, registry)

    return records


def run_local_generation(
    requests_path: Path | None = None,
    output_path: Path | None = None,
    registry_path: Path | None = None,
    model_id: str = OLLAMA_MODEL_ID,
    api_url: str = OLLAMA_API_URL,
    question_id: str | None = None,
    timeout: int = OLLAMA_REQUEST_TIMEOUT,
) -> list[dict[str, Any]]:
    """Execute local Ollama inference on prepared requests without loading retrieval models."""
    if model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model '{model_id}'. Allowed models are: {list(ALLOWED_MODELS)}"
        )

    req_file = requests_path or (paths.data_dir / ".generation" / "requests.jsonl")
    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")

    if output_path is not None:
        out_file = Path(output_path)
        if not out_file.is_absolute():
            if out_file.parts and out_file.parts[0] == "data":
                out_file = paths.project_root / out_file
            else:
                out_file = paths.data_dir / ".generation" / out_file
    else:
        out_file = paths.data_dir / ".generation" / "ollama_results.jsonl"

    # Prevent writing under reports directory
    if paths.reports_dir in out_file.parents or "reports" in out_file.parts:
        raise ValueError(f"Results must not be created under tracked reports directory: {out_file}")

    # Prevent accidental overwriting of an existing experiment file
    if out_file.exists():
        raise FileExistsError(
            f"Output file already exists: {out_file}. "
            f"Accidental overwrite prevented. Please specify a new --output filename or remove the existing file."
        )

    with open(reg_file, "r", encoding="utf-8") as f:
        registry = {r["document_id"]: r for r in json.load(f)}

    all_requests = load_and_validate_requests(req_file, registry)

    if question_id is not None:
        if question_id not in EXPECTED_QIDS:
            raise ValueError(
                f"Invalid question_id '{question_id}'. Allowed questions: {EXPECTED_QIDS}"
            )
        requests = [r for r in all_requests if r.get("question_id") == question_id]
        if not requests:
            raise ValueError(f"Question '{question_id}' not found in {req_file}")
    else:
        requests = all_requests

    results = []
    print(
        f"Executing generation for {len(requests)} question(s) using model '{model_id}' "
        f"(timeout: {timeout}s)..."
    )

    for req in requests:
        qid = req["question_id"]
        q_text = req["question"]
        role = req["role"]
        chunks = req["retrieved_chunks"]

        print(f"Processing {qid} (role: {role})...", end=" ", flush=True)

        t_start = time.perf_counter()
        try:
            res = generate_grounded_answer_ollama(
                question=q_text,
                role=role,
                retrieved_chunks=chunks,
                registry=registry,
                model_id=model_id,
                api_url=api_url,
                timeout=timeout,
                question_id=qid,
            )
            results.append(res)
            status_desc = "PASS" if res["validation_passed"] else f"FAIL ({res['validation_error']})"
            print(f"done ({res['generation_time_seconds']:.2f}s) -> Status: {res['parsed_status']} [{status_desc}]")
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            err_record = {
                "question_id": qid,
                "model_id": model_id,
                "raw_model_output": None,
                "parsed_status": "error",
                "parsed_answer": "",
                "validated_citations": [],
                "resolved_citations": [],
                "validation_passed": False,
                "validation_error": str(e),
                "generation_time_seconds": elapsed,
                "total_duration_ns": None,
                "load_duration_ns": None,
                "prompt_eval_duration_ns": None,
                "eval_duration_ns": None,
                "total_duration_seconds": None,
                "load_duration_seconds": None,
                "prompt_eval_duration_seconds": None,
                "eval_duration_seconds": None,
                "prompt_eval_count": None,
                "eval_count": None,
                "eval_tokens_per_second": None,
            }
            results.append(err_record)
            print(f"FAILED after {elapsed:.2f}s: {e}")

    # Write output JSONL (even when failures occur)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quorift — Local Qwen Generation Runner")
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL_ID,
        choices=ALLOWED_MODELS,
        help="Ollama model identifier to execute.",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        choices=EXPECTED_QIDS,
        help="Run only one designated approved development question.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=OLLAMA_REQUEST_TIMEOUT,
        help="Per-request generation timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output filename or relative path inside data/.generation/.",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Quorift — Local Qwen Generation Runner (Ollama)")
    print("=" * 70)

    try:
        results = run_local_generation(
            model_id=args.model,
            question_id=args.question_id,
            timeout=args.timeout,
            output_path=Path(args.output) if args.output else None,
        )
        print(f"\nExecution finished. Results saved.")
        print(f"{'QID':<8} | {'Status':<22} | {'Valid':<7} | {'Time (s)':<10} | {'Tok/s':<8} | {'Citations':<25}")
        print("-" * 90)
        for r in results:
            cits = ", ".join(r["validated_citations"]) if r["validated_citations"] else "[]"
            tok_s = f"{r.get('eval_tokens_per_second') or 'N/A'}"
            print(f"{r['question_id']:<8} | {r['parsed_status']:<22} | {str(r['validation_passed']):<7} | {r['generation_time_seconds']:<10.2f} | {tok_s:<8} | {cits:<25}")

        has_failures = any(not r.get("validation_passed", False) for r in results)
        if has_failures:
            print("\n[WARNING] One or more generation requests failed validation or timed out.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Runner execution halted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
