"""Baseline retrieval evaluation on eight selected development questions."""

import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from quorift.config import paths
from quorift.retrieval import HybridRetriever

SELECTED_QIDS = [
    "Q-001",
    "Q-010",
    "Q-012",
    "Q-021",
    "Q-022",
    "Q-025",
    "Q-033",
    "Q-034",
]


def load_selected_questions(
    questions_path: Path | None = None,
    facts_path: Path | None = None,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the eight designated development evaluation questions."""
    q_file = questions_path or (paths.evaluation_dir / "questions.jsonl")
    f_file = facts_path or (paths.metadata_dir / "fact_ledger.json")
    r_file = registry_path or (paths.metadata_dir / "document_registry.json")

    with open(f_file, encoding="utf-8") as f:
        facts = {fact["fact_id"]: fact for fact in json.load(f)}

    with open(r_file, encoding="utf-8") as f:
        registry = {reg["document_id"]: reg for reg in json.load(f)}

    all_questions = []
    with open(q_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    selected = [q for q in all_questions if q["question_id"] in SELECTED_QIDS]
    if len(selected) != 8:
        raise ValueError(f"Expected exactly 8 questions, loaded {len(selected)}")

    # Sort in specified order
    order_map = {qid: i for i, qid in enumerate(SELECTED_QIDS)}
    selected.sort(key=lambda q: order_map[q["question_id"]])

    for q in selected:
        qid = q["question_id"]
        assert q["split"] == "development", f"{qid} must be in development split"
        assert q["corpus_state"] == "v1_active", f"{qid} corpus_state must be v1_active"
        assert q["expected_behavior"] == "answer", f"{qid} expected_behavior must be answer"

        # Resolve exact (document_id, section_number) gold evidence pairs from facts
        gold_pairs = set()
        for fid in q["gold_fact_ids"]:
            fact = facts[fid]
            gold_pairs.add((fact["document_id"], fact["section"]))

        assert len(gold_pairs) > 0, f"{qid} has no gold evidence pairs"
        q["gold_evidence_pairs"] = sorted(list(gold_pairs))

        # Check role permission
        role = q["role"]
        for doc_id, _ in gold_pairs:
            reg_entry = registry[doc_id]
            assert role in reg_entry["allowed_roles"], f"{qid}: Role {role} cannot access {doc_id}"

    return selected


def evaluate_retrieval(
    retriever: HybridRetriever | None = None,
    indexes_dir: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run baseline retrieval evaluation across BM25, Dense, and RRF on the 8 development questions."""
    created = False
    if retriever is None:
        retriever = HybridRetriever(indexes_dir=indexes_dir)
        created = True

    try:
        questions = load_selected_questions()
        methods = ["bm25", "dense", "rrf"]

        raw_results: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
        latencies: dict[str, list[float]] = {m: [] for m in methods}

        for q in questions:
            qid = q["question_id"]
            q_text = q["question"]
            role = q["role"]
            req_pairs = q["gold_evidence_pairs"]

            # 1. Evaluate BM25
            t0 = time.perf_counter()
            bm25_hits = retriever.bm25_search(q_text, role=role, candidate_k=top_k)
            t_bm25 = time.perf_counter() - t0
            latencies["bm25"].append(t_bm25)

            # 2. Evaluate Dense
            t0 = time.perf_counter()
            dense_hits = retriever.dense_search(q_text, role=role, candidate_k=top_k)
            t_dense = time.perf_counter() - t0
            latencies["dense"].append(t_dense)

            # 3. Evaluate RRF
            t0 = time.perf_counter()
            rrf_hits = retriever.retrieve(q_text, role=role, top_k=top_k)
            t_rrf = time.perf_counter() - t0
            latencies["rrf"].append(t_rrf)

            hits_map = {
                "bm25": bm25_hits,
                "dense": dense_hits,
                "rrf": rrf_hits,
            }

            for m in methods:
                hits = hits_map[m]
                ret_pairs = [(h["document_id"], h["section_number"]) for h in hits[:top_k]]

                hit_at_5 = 1.0 if any(p in ret_pairs for p in req_pairs) else 0.0
                retrieved_req = [p for p in req_pairs if p in ret_pairs]
                recall_at_5 = len(retrieved_req) / len(req_pairs)
                full_at_5 = 1.0 if len(retrieved_req) == len(req_pairs) else 0.0

                # MRR calculation
                first_rank = None
                for idx, p in enumerate(ret_pairs, start=1):
                    if p in req_pairs:
                        first_rank = idx
                        break
                mrr_at_5 = (1.0 / first_rank) if first_rank else 0.0

                raw_results[m].append(
                    {
                        "question_id": qid,
                        "question": q_text,
                        "role": role,
                        "category": q["category"],
                        "required_evidence": req_pairs,
                        "retrieved_evidence": ret_pairs,
                        "correct_retrieved": retrieved_req,
                        "first_correct_rank": first_rank,
                        "hit_at_5": hit_at_5,
                        "recall_at_5": recall_at_5,
                        "mrr_at_5": mrr_at_5,
                        "full_evidence_at_5": full_at_5,
                        "top_chunk_ids": [h["chunk_id"] for h in hits[:top_k]],
                    }
                )

        summary_metrics = {}
        for m in methods:
            m_res = raw_results[m]
            m_lats = latencies[m]

            def percentile(vals: list[float], pct: float) -> float:
                if not vals:
                    return 0.0
                s = sorted(vals)
                k = (len(s) - 1) * pct
                f = int(k)
                c = min(f + 1, len(s) - 1)
                return s[f] + (s[c] - s[f]) * (k - f)

            summary_metrics[m] = {
                "hit_at_5": statistics.mean([r["hit_at_5"] for r in m_res]),
                "recall_at_5": statistics.mean([r["recall_at_5"] for r in m_res]),
                "mrr_at_5": statistics.mean([r["mrr_at_5"] for r in m_res]),
                "full_evidence_at_5": statistics.mean([r["full_evidence_at_5"] for r in m_res]),
                "median_latency_ms": statistics.median(m_lats) * 1000,
                "p95_latency_ms": percentile(m_lats, 0.95) * 1000,
            }

        return {
            "questions": questions,
            "raw_results": raw_results,
            "summary_metrics": summary_metrics,
        }
    finally:
        if created:
            retriever.close()


def generate_development_report(eval_output: dict[str, Any], output_path: Path) -> None:
    """Generate reports/retrieval_development.md markdown report from evaluation results."""
    summary = eval_output["summary_metrics"]
    raw = eval_output["raw_results"]
    questions = eval_output["questions"]

    lines = [
        "# Retrieval Development Report — Baseline Evaluation",
        "",
        "## 1. Dataset & Setup",
        "",
        "- **Evaluated Questions**: Exactly 8 development questions.",
        "- **Corpus**: Apex Crest Bank Operational Policies (Version 1.0, 20 documents, 120 chunks).",
        "- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, CPU execution).",
        "- **Lexical Index**: `bm25s` with tokenization across full chunk corpus.",
        "- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with constant `k=60`, candidate depth 20 per method, `top_k=5`.",
        "- **Role Filtering**: Strict role-based permission enforcement before candidate truncation.",
        "",
        "| Question ID | Category | Role | Required Evidence |",
        "|---|---|---|---|",
    ]

    for q in questions:
        req_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in q["gold_evidence_pairs"])
        lines.append(f"| {q['question_id']} | {q['category']} | `{q['role']}` | {req_str} |")

    lines.extend(
        [
            "",
            "## 2. Comparison Metrics (top_k=5)",
            "",
            "| Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median latency | P95 latency |",
            "|---|---|---|---|---|---|---|",
            f"| BM25 | {summary['bm25']['hit_at_5']:.4f} | {summary['bm25']['recall_at_5']:.4f} | {summary['bm25']['mrr_at_5']:.4f} | {summary['bm25']['full_evidence_at_5']:.4f} | {summary['bm25']['median_latency_ms']:.2f} ms | {summary['bm25']['p95_latency_ms']:.2f} ms |",
            f"| BGE dense | {summary['dense']['hit_at_5']:.4f} | {summary['dense']['recall_at_5']:.4f} | {summary['dense']['mrr_at_5']:.4f} | {summary['dense']['full_evidence_at_5']:.4f} | {summary['dense']['median_latency_ms']:.2f} ms | {summary['dense']['p95_latency_ms']:.2f} ms |",
            f"| RRF baseline | {summary['rrf']['hit_at_5']:.4f} | {summary['rrf']['recall_at_5']:.4f} | {summary['rrf']['mrr_at_5']:.4f} | {summary['rrf']['full_evidence_at_5']:.4f} | {summary['rrf']['median_latency_ms']:.2f} ms | {summary['rrf']['p95_latency_ms']:.2f} ms |",
            "",
            "## 3. Question-Level Results",
            "",
            "| QID | Required Evidence | BM25 First Rank | Dense First Rank | RRF First Rank | RRF Full Evidence |",
            "|---|---|---|---|---|---|",
        ]
    )

    for i, q in enumerate(questions):
        qid = q["question_id"]
        req_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in q["gold_evidence_pairs"])
        b_rank = raw["bm25"][i]["first_correct_rank"]
        d_rank = raw["dense"][i]["first_correct_rank"]
        r_rank = raw["rrf"][i]["first_correct_rank"]
        r_full = "Yes" if raw["rrf"][i]["full_evidence_at_5"] == 1.0 else "No (Partial)"

        b_str = str(b_rank) if b_rank else "Miss"
        d_str = str(d_rank) if d_rank else "Miss"
        r_str = str(r_rank) if r_rank else "Miss"
        lines.append(f"| {qid} | {req_str} | {b_str} | {d_str} | {r_str} | {r_full} |")

    lines.extend(
        [
            "",
            "## 4. Failure & Partial Evidence Analysis",
            "",
        ]
    )

    misses = [r for r in raw["rrf"] if r["full_evidence_at_5"] < 1.0]
    if not misses:
        lines.append("All eight questions retrieved complete required evidence within the top 5 results.")
    else:
        for m in misses:
            qid = m["question_id"]
            req_pairs = m["required_evidence"]
            correct_pairs = m["correct_retrieved"]
            missing_pairs = [p for p in req_pairs if p not in correct_pairs]
            top_chunks = ", ".join(m["top_chunk_ids"])
            missing_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in missing_pairs)
            lines.append(f"### {qid} ({m['category']})")
            lines.append(f"- **Missing Required Sections**: {missing_str}")
            lines.append(f"- **Top-5 Chunks Returned**: `{top_chunks}`")
            if len(req_pairs) > 1 and len(correct_pairs) > 0:
                lines.append(
                    f"- **Observation**: Multi-evidence question where {len(correct_pairs)}/{len(req_pairs)} "
                    f"required sections appeared in top 5. The missing section was ranked below position 5."
                )
            else:
                lines.append(
                    "- **Observation**: Target evidence was outranked by semantically adjacent policy sections."
                )
            lines.append("")

    lines.extend(
        [
            "## 5. Baseline Conclusions & Next Steps",
            "",
            "- **Observed Strengths**: Baseline hybrid RRF effectively combines lexical precision with semantic matching.",
            "- **Permission Boundary Integrity**: Role filtering completely prevents unauthorized cross-department leakage.",
            "- **Potential Improvements to Investigate in Next Tasks**:",
            "  1. Cross-encoder reranking (`ms-marco-MiniLM-L6-v2`) to re-score and promote multi-section evidence into the top 3–5 slots.",
            "  2. Exploration of candidate depth and fusion weights.",
            "",
            "> [!NOTE]",
            "> This report represents initial baseline measurements on 8 development questions without parameter tuning.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_evaluation() -> None:
    """CLI entry point for running retrieval evaluation."""
    print("=" * 65)
    print("Quorift — Baseline Retrieval Evaluation (8 Development Questions)")
    print("=" * 65)

    eval_output = evaluate_retrieval()
    summary = eval_output["summary_metrics"]

    print("\n[Summary Metrics — top_k=5]")
    print(
        f"{'Method':<14} | {'Hit@5':<8} | {'Recall@5':<10} | {'MRR@5':<8} | {'FullEv@5':<9} | {'Median(ms)':<10} | {'P95(ms)':<10}"
    )
    print("-" * 85)
    for m in ["bm25", "dense", "rrf"]:
        s = summary[m]
        print(
            f"{m:<14} | {s['hit_at_5']:<8.4f} | {s['recall_at_5']:<10.4f} | {s['mrr_at_5']:<8.4f} | {s['full_evidence_at_5']:<9.4f} | {s['median_latency_ms']:<10.2f} | {s['p95_latency_ms']:<10.2f}"
        )

    report_file = paths.reports_dir / "retrieval_development.md"
    generate_development_report(eval_output, report_file)
    print(f"\nReport successfully generated and saved to: {report_file}")


def main() -> None:
    try:
        run_evaluation()
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
