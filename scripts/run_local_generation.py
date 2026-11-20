"""Local generation runner executing Qwen inference via Ollama on prepared development requests."""

import json
from pathlib import Path
import sys
from typing import Any

from quorift.config import paths
from quorift.generation import validate_generation_request
from quorift.ollama_backend import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_ID,
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
) -> list[dict[str, Any]]:
    """Execute local Ollama inference on prepared requests without loading retrieval models."""
    req_file = requests_path or (paths.data_dir / ".generation" / "requests.jsonl")
    out_file = output_path or (paths.data_dir / ".generation" / "ollama_results.jsonl")
    reg_file = registry_path or (paths.metadata_dir / "document_registry.json")

    with open(reg_file, "r", encoding="utf-8") as f:
        registry = {r["document_id"]: r for r in json.load(f)}

    requests = load_and_validate_requests(req_file, registry)
    results = []

    print(f"Executing generation for {len(requests)} questions using Ollama model '{model_id}'...")

    for req in requests:
        qid = req["question_id"]
        q_text = req["question"]
        role = req["role"]
        chunks = req["retrieved_chunks"]

        print(f"Processing {qid} (role: {role})...", end=" ", flush=True)

        try:
            res = generate_grounded_answer_ollama(
                question=q_text,
                role=role,
                retrieved_chunks=chunks,
                registry=registry,
                model_id=model_id,
                api_url=api_url,
                question_id=qid,
            )
            results.append(res)
            status_desc = "PASS" if res["validation_passed"] else f"FAIL ({res['validation_error']})"
            print(f"done ({res['generation_time_seconds']:.2f}s) -> Status: {res['parsed_status']} [{status_desc}]")
        except Exception as e:
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
                "generation_time_seconds": 0.0,
                "prompt_eval_count": 0,
                "eval_count": 0,
            }
            results.append(err_record)
            print(f"FAILED: {e}")

    # Write output JSONL
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return results


def main() -> None:
    print("=" * 70)
    print("Quorift — Local Qwen Generation Runner (Ollama)")
    print("=" * 70)

    req_file = paths.data_dir / ".generation" / "requests.jsonl"
    out_file = paths.data_dir / ".generation" / "ollama_results.jsonl"

    try:
        results = run_local_generation(requests_path=req_file, output_path=out_file)
        print(f"\nExecution finished. Results saved to: {out_file}\n")
        print(f"{'QID':<8} | {'Status':<22} | {'Valid':<7} | {'Time (s)':<10} | {'Citations':<25}")
        print("-" * 80)
        for r in results:
            cits = ", ".join(r["validated_citations"]) if r["validated_citations"] else "[]"
            status_str = f"{r['parsed_status']}"
            print(f"{r['question_id']:<8} | {status_str:<22} | {str(r['validation_passed']):<7} | {r['generation_time_seconds']:<10.2f} | {cits:<25}")

    except Exception as e:
        print(f"\n[ERROR] Runner execution halted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
