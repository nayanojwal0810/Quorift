"""Command-line entry point for querying the Quorift RAG pipeline."""

import argparse
import sys

from quorift.generation import VALID_ROLES
from quorift.pipeline import answer_question


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quorift — Enterprise Banking Knowledge Assistant (RAG Pipeline)"
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=sorted(VALID_ROLES),
        help="Demonstration role for access-control simulation (not user authentication).",
    )
    parser.add_argument(
        "--question",
        required=True,
        type=str,
        help="Natural language banking policy question to answer.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request generation timeout in seconds (default: 300).",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("Quorift — Banking Knowledge Assistant")
    print("=" * 70)
    print(f"Role:     {args.role} (demonstration input)")
    print(f"Question: {args.question.strip()}")
    print("-" * 70)

    try:
        res = answer_question(
            question=args.question,
            role=args.role,
            timeout=args.timeout,
        )

        status = res["status"]
        if status == "error":
            print(f"\n[ERROR] Request failed: {res['error']}")
            print("\nTiming:")
            print(f"- Retrieval:  {res['retrieval_time_seconds']:.2f}s")
            print(f"- Generation: {res['generation_time_seconds']:.2f}s")
            print(f"- Total:      {res['total_time_seconds']:.2f}s")
            print("=" * 70)
            sys.exit(1)

        print("\nAnswer:")
        print(res["answer"])

        citations = res.get("resolved_citations", [])
        if citations:
            print("\nCitations:")
            for c in citations:
                doc_title = c.get("title", c.get("document_id", "Unknown"))
                ver = c.get("version", "1.0")
                sec = c.get("section_number", "")
                p_start = c.get("page_start", "")
                p_end = c.get("page_end", "")
                page_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
                cid = c.get("chunk_id", "")
                print(f"- {c['document_id']} — {doc_title} (Version {ver}, Section {sec}, {page_str}) [{cid}]")
        elif status == "insufficient_evidence":
            print("\nCitations:\n(None — insufficient policy evidence)")

        print("\nTiming:")
        print(f"- Retrieval:  {res['retrieval_time_seconds']:.2f}s")
        print(f"- Generation: {res['generation_time_seconds']:.2f}s")
        print(f"- Total:      {res['total_time_seconds']:.2f}s")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Pipeline halted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
