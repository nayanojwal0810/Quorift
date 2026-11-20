"""Prepare portable generation input JSONL for the 6 approved development questions."""

import json
from pathlib import Path
import sys
from typing import Any

from quorift.config import paths
from quorift.generation import validate_generation_request
from quorift.retrieval import HybridRetriever

APPROVED_DEV_QIDS = [
    "Q-001",
    "Q-018",
    "Q-025",
    "Q-033",
    "Q-037",
    "Q-041",
]


def load_generation_questions(
    questions_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the 6 designated development questions in strict fixed order."""
    q_file = questions_path or (paths.evaluation_dir / "questions.jsonl")
    all_questions = []
    with open(q_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    selected = [q for q in all_questions if q["question_id"] in APPROVED_DEV_QIDS]
    if len(selected) != len(APPROVED_DEV_QIDS):
        raise ValueError(
            f"Expected exactly {len(APPROVED_DEV_QIDS)} questions, found {len(selected)}"
        )

    # Sort into deterministic approved order
    order_map = {qid: i for i, qid in enumerate(APPROVED_DEV_QIDS)}
    selected.sort(key=lambda q: order_map[q["question_id"]])

    for q in selected:
        qid = q["question_id"]
        if q.get("split") != "development":
            raise ValueError(f"{qid} must belong to development split")
        if q.get("corpus_state") != "v1_active":
            raise ValueError(f"{qid} must have corpus_state v1_active")

    return selected


def prepare_generation_inputs(
    output_path: Path | None = None,
    indexes_dir: Path | None = None,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Execute local RRF retrieval for each approved question and write validated generation inputs."""
    out_file = output_path or (paths.data_dir / ".generation" / "requests.jsonl")
    r_file = registry_path or (paths.metadata_dir / "document_registry.json")

    with open(r_file, encoding="utf-8") as f:
        registry = {reg["document_id"]: reg for reg in json.load(f)}

    questions = load_generation_questions()
    retriever = HybridRetriever(indexes_dir=indexes_dir, registry_path=r_file)

    requests: list[dict[str, Any]] = []
    try:
        for q in questions:
            qid = q["question_id"]
            q_text = q["question"]
            role = q["role"]

            # Perform top-5 RRF retrieval
            hits = retriever.retrieve(question=q_text, role=role, top_k=5)

            retrieved_chunks = []
            for h in hits:
                cid = h["chunk_id"]
                full_chunk = retriever.chunks_by_id.get(cid)
                if not full_chunk:
                    raise ValueError(f"Chunk record '{cid}' missing from loaded chunks store")

                retrieved_chunks.append(
                    {
                        "chunk_id": full_chunk["chunk_id"],
                        "document_id": full_chunk["document_id"],
                        "title": full_chunk["title"],
                        "section_number": full_chunk["section_number"],
                        "section_heading": full_chunk["section_heading"],
                        "version": full_chunk["version"],
                        "source_hash": full_chunk["source_hash"],
                        "page_start": full_chunk["page_start"],
                        "page_end": full_chunk["page_end"],
                        "source_pdf": full_chunk["source_pdf"],
                        "allowed_roles": full_chunk["allowed_roles"],
                        "text": full_chunk["text"],
                    }
                )

            request_record = {
                "schema_version": "1.0",
                "question_id": qid,
                "question": q_text,
                "role": role,
                "retrieved_chunks": retrieved_chunks,
            }

            # Security and consistency validation
            validate_generation_request(request_record, registry)
            requests.append(request_record)

        # Write to local temporary JSONL
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

    finally:
        retriever.close()

    return requests


def main() -> None:
    print("=" * 70)
    print("Quorift — Preparing Generation Inputs (6 Approved Development Questions)")
    print("=" * 70)

    try:
        requests = prepare_generation_inputs()
        out_file = paths.data_dir / ".generation" / "requests.jsonl"
        print(f"\nSuccessfully generated {len(requests)} input records at: {out_file}\n")
        print(f"{'QID':<8} | {'Role':<10} | {'Retrieved Chunks Count':<24} | {'Top Retrieved Chunk IDs'}")
        print("-" * 75)
        for req in requests:
            cids = [c["chunk_id"] for c in req["retrieved_chunks"]]
            cids_str = ", ".join(cids) if cids else "(empty - refusal expected)"
            print(f"{req['question_id']:<8} | {req['role']:<10} | {len(cids):<24} | {cids_str}")

        print("\n[SUCCESS] Input preparation complete and permission-safe.")
    except Exception as e:
        print(f"\n[ERROR] Preparation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
