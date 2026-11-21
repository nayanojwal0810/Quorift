"""CLI tool for preparing GPU generation requests and local manifests from question records."""

import argparse
import json
from pathlib import Path
import sys

# Ensure quorift package is discoverable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from quorift.generation import VALID_ROLES
from quorift.gpu_exchange import GPU_MODEL_ID
from quorift.pipeline import prepare_gpu_generation_request


def _validate_output_dir(output_dir: Path) -> None:
    """Ensure output directory is outside tracked repository directories."""
    resolved = output_dir.resolve()
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
                f"Security check failed: output directory '{output_dir}' is inside tracked project directories. "
                r"Please specify an output path outside tracked folders (e.g., D:\Quorift_Internal or a separate scratch folder)."
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quorift — Prepare GPU generation batch and local manifest from questions JSONL.",
        epilog=(
            "Windows PowerShell example:\n"
            r"  python scripts\prepare_gpu_batch.py "
            r"--input D:\Quorift_Internal\data\evaluation\questions.jsonl "
            r"--output-dir D:\Quorift_Internal"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input JSONL file containing question records with 'question' and 'role' fields.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Directory to write 'gpu_requests.jsonl' and 'gpu_manifest.jsonl' (must be outside tracked dirs).",
    )
    parser.add_argument(
        "--requests-filename",
        type=str,
        default="gpu_requests.jsonl",
        help="Filename for exported GPU request payloads (default: 'gpu_requests.jsonl').",
    )
    parser.add_argument(
        "--manifest-filename",
        type=str,
        default="gpu_manifest.jsonl",
        help="Filename for local trusted manifest records (default: 'gpu_manifest.jsonl').",
    )
    parser.add_argument(
        "--indexes-dir",
        type=Path,
        default=None,
        help="Optional custom path to local search indexes directory.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Optional custom path to trusted document registry JSON.",
    )
    parser.add_argument(
        "--retrieval-timeout",
        type=int,
        default=60,
        help="Subprocess retrieval timeout per question in seconds (default: 60).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"[FATAL] Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        _validate_output_dir(args.output_dir)
    except ValueError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.output_dir / args.requests_filename
    manifest_path = args.output_dir / args.manifest_filename

    # Read question items
    questions = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"[FATAL] Malformed JSON on line {line_no} of {args.input}: {e}", file=sys.stderr)
                sys.exit(1)

            if "question" not in item or "role" not in item:
                print(
                    f"[FATAL] Line {line_no} missing required 'question' or 'role' fields.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if item["role"] not in VALID_ROLES:
                print(
                    f"[FATAL] Line {line_no} has invalid role '{item['role']}'. Valid roles: {sorted(VALID_ROLES)}",
                    file=sys.stderr,
                )
                sys.exit(1)

            questions.append(item)

    print("=" * 70)
    print("Quorift — Prepare GPU Generation Batch")
    print("=" * 70)
    print(f"Target Model:  {GPU_MODEL_ID}")
    print(f"Input records: {len(questions)} from {args.input}")
    print(f"Requests file: {requests_path}")
    print(f"Manifest file: {manifest_path}")
    print("-" * 70)

    prepared_count = 0
    local_refusal_count = 0

    with open(requests_path, "w", encoding="utf-8") as f_req, open(manifest_path, "w", encoding="utf-8") as f_man:
        for idx, item in enumerate(questions, start=1):
            q_text = item["question"]
            role = item["role"]
            qid = item.get("question_id", f"Q_{idx:03d}")

            print(f"[{idx}/{len(questions)}] Preparing {qid} (role: {role})...")

            try:
                gpu_req, manifest = prepare_gpu_generation_request(
                    question=q_text,
                    role=role,
                    indexes_dir=args.indexes_dir,
                    registry_path=args.registry_path,
                    retrieval_timeout=args.retrieval_timeout,
                    model_id=GPU_MODEL_ID,
                    question_id=qid,
                )
            except Exception as e:
                print(f"  [ERROR] Retrieval/preparation failed for {qid}: {e}", file=sys.stderr)
                sys.exit(1)

            # Write manifest
            f_man.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            f_man.flush()

            # Write GPU request if GPU generation is needed
            if manifest.get("requires_gpu", True) and gpu_req is not None:
                f_req.write(json.dumps(gpu_req, ensure_ascii=False) + "\n")
                f_req.flush()
                prepared_count += 1
                print(f"  Exported GPU request: {gpu_req['request_id']} (sha256: {gpu_req['request_sha256'][:8]}...)")
            else:
                local_refusal_count += 1
                print("  Zero authorized chunks retrieved -> marked for local refusal (no GPU needed).")

    print("-" * 70)
    print(f"[SUCCESS] Batch prepared: {prepared_count} GPU requests exported, {local_refusal_count} local refusals.")
    print(f"Next step: Upload '{requests_path}' to Google Colab and run 'quorift_gpu_generation.ipynb'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
