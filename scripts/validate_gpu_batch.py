"""CLI tool for validating returned Google Colab GPU responses against local manifests."""

import argparse
import json
from pathlib import Path
import sys

# Ensure quorift package is discoverable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from quorift.pipeline import process_gpu_batch


def _validate_output_file(output_file: Path) -> None:
    """Ensure output file is outside tracked repository directories."""
    resolved = output_file.resolve()
    tracked_forbidden = [
        repo_root,
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests",
        repo_root / "data" / "metadata",
    ]
    for forbidden in tracked_forbidden:
        if resolved == forbidden or (
            resolved.is_relative_to(forbidden)
            and not str(resolved).endswith(".generation")
        ):
            raise ValueError(
                f"Security check failed: output file '{output_file}' is inside tracked project directories. "
                r"Please specify an output path outside tracked folders (e.g., D:\Quorift_Internal\results.jsonl)."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quorift — Validate returned GPU generation responses against local manifest.",
        epilog=(
            "Windows PowerShell example:\n"
            r"  python scripts\validate_gpu_batch.py "
            r"--manifest D:\Quorift_Internal\gpu_manifest.jsonl "
            r"--gpu-responses D:\Quorift_Internal\gpu_responses.jsonl "
            r"--output D:\Quorift_Internal\gpu_validated_results.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        "-m",
        required=True,
        type=Path,
        help="Path to local trusted manifest JSONL file generated during preparation.",
    )
    parser.add_argument(
        "--gpu-responses",
        "-r",
        required=True,
        type=Path,
        help="Path to returned GPU response JSONL file downloaded from Google Colab.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path to write validated results JSONL file (must be outside tracked project dirs).",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional custom path to trusted document registry JSON.",
    )
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"[FATAL] Manifest file does not exist: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if not args.gpu_responses.exists():
        print(f"[FATAL] GPU responses file does not exist: {args.gpu_responses}", file=sys.stderr)
        sys.exit(1)

    try:
        _validate_output_file(args.output)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    # Load manifests
    manifests = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                manifests.append(json.loads(line_str))

    # Load GPU responses
    gpu_responses = []
    with open(args.gpu_responses, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                gpu_responses.append(json.loads(line_str))

    print("=" * 70)
    print("Quorift — Validate GPU Batch Responses")
    print("=" * 70)
    print(f"Manifest records:      {len(manifests)} from {args.manifest}")
    print(f"GPU response records:  {len(gpu_responses)} from {args.gpu_responses}")
    print(f"Output destination:    {args.output}")
    print("-" * 70)

    # Process and validate batch
    try:
        validated_results = process_gpu_batch(
            gpu_responses=gpu_responses,
            manifests=manifests,
            registry_path=args.registry_path,
        )
    except Exception as e:
        print(f"[FATAL] Batch processing failed: {e}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for res in validated_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # Display summary
    answered_count = sum(1 for r in validated_results if r["status"] == "answered")
    refused_count = sum(1 for r in validated_results if r["status"] == "insufficient_evidence")
    error_count = sum(1 for r in validated_results if r["status"] == "error")

    print(f"Validated records written: {len(validated_results)}")
    print(f"  - Answered:              {answered_count}")
    print(f"  - Safe Refusal:          {refused_count}")
    print(f"  - Errors / Failures:     {error_count}")

    if error_count > 0:
        print("\nErrors encountered:")
        for r in validated_results:
            if r["status"] == "error":
                print(f"  - Request {r.get('request_id')} ({r.get('question_id')}): {r.get('error')}")

    print("=" * 70)
    print(f"[SUCCESS] Validated results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
