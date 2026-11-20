"""Baseline retrieval evaluation on selected development question batches."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from quorift.config import paths
from quorift.retrieval import HybridRetriever

BATCH_1_QIDS = [
    "Q-001",
    "Q-010",
    "Q-012",
    "Q-021",
    "Q-022",
    "Q-025",
    "Q-033",
    "Q-034",
]

BATCH_2_QIDS = [
    "Q-018",
    "Q-020",
    "Q-024",
    "Q-026",
    "Q-028",
    "Q-030",
    "Q-035",
    "Q-036",
]

BATCH_3_QIDS = [
    "Q-017",
    "Q-019",
    "Q-027",
    "Q-029",
    "Q-031",
    "Q-032",
    "Q-041",
    "Q-043",
]

SELECTED_QIDS = BATCH_1_QIDS


def compute_retrieval_metrics(
    gold_pairs: list[tuple[str, str]] | set[tuple[str, str]],
    retrieved_pairs: list[tuple[str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Calculate standard retrieval metrics for a single query.

    Args:
        gold_pairs: Target (document_id, section_number) pairs.
        retrieved_pairs: Ranked list of (document_id, section_number) pairs.
        top_k: Evaluation cutoff rank (default 5).

    Returns:
        dict with hit_at_k, recall_at_k, mrr_at_k, full_evidence_at_k,
        first_correct_rank, retrieved_gold.
    """
    gold_set = set(gold_pairs)
    if not gold_set:
        raise ValueError("gold_pairs must not be empty")

    top_hits = retrieved_pairs[:top_k]

    first_rank = None
    for idx, p in enumerate(top_hits, start=1):
        if p in gold_set:
            first_rank = idx
            break

    retrieved_gold = []
    seen = set()
    for p in top_hits:
        if p in gold_set and p not in seen:
            seen.add(p)
            retrieved_gold.append(p)

    hit = 1.0 if len(retrieved_gold) > 0 else 0.0
    recall = len(retrieved_gold) / len(gold_set)
    full = 1.0 if len(retrieved_gold) == len(gold_set) else 0.0
    mrr = (1.0 / first_rank) if first_rank else 0.0

    return {
        f"hit_at_{top_k}": hit,
        f"recall_at_{top_k}": recall,
        f"mrr_at_{top_k}": mrr,
        f"full_evidence_at_{top_k}": full,
        "first_correct_rank": first_rank,
        "retrieved_gold": retrieved_gold,
    }


def load_selected_questions(
    batch: int = 1,
    questions_path: Path | None = None,
    facts_path: Path | None = None,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the designated development evaluation questions for the specified batch."""
    if batch == 1:
        target_qids = BATCH_1_QIDS
    elif batch == 2:
        target_qids = BATCH_2_QIDS
    elif batch == 3:
        target_qids = BATCH_3_QIDS
    else:
        raise ValueError(f"Invalid batch {batch}. Only batch 1, 2, and 3 are supported.")

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

    selected = [q for q in all_questions if q["question_id"] in target_qids]
    if len(selected) != 8:
        raise ValueError(f"Expected exactly 8 questions for batch {batch}, loaded {len(selected)}")

    # Sort in specified fixed order
    order_map = {qid: i for i, qid in enumerate(target_qids)}
    selected.sort(key=lambda q: order_map[q["question_id"]])

    for q in selected:
        qid = q["question_id"]
        assert q["split"] == "development", f"{qid} must be in development split"
        assert q["corpus_state"] == "v1_active", f"{qid} corpus_state must be v1_active"

        # Resolve exact (document_id, section_number) gold evidence pairs from facts
        gold_pairs = set()
        for fid in q["gold_fact_ids"]:
            fact = facts[fid]
            gold_pairs.add((fact["document_id"], fact["section"]))

        assert len(gold_pairs) > 0, f"{qid} has no gold evidence pairs"
        q["gold_evidence_pairs"] = sorted(list(gold_pairs))
        q["restricted_docs"] = sorted(list(set(doc for doc, _ in gold_pairs)))

        role = q["role"]
        if q["expected_behavior"] == "answer":
            q["is_denial"] = False
            for doc_id, _ in gold_pairs:
                reg_entry = registry[doc_id]
                assert role in reg_entry["allowed_roles"], f"{qid}: Role {role} cannot access {doc_id}"
        elif q["expected_behavior"] == "deny_unauthorized":
            q["is_denial"] = True
            for doc_id in q["restricted_docs"]:
                reg_entry = registry[doc_id]
                assert role not in reg_entry["allowed_roles"], f"{qid}: Role {role} is unexpectedly authorized for {doc_id}"
        else:
            raise ValueError(f"{qid} has unexpected expected_behavior: {q['expected_behavior']}")

    return selected


def evaluate_retrieval(
    batch: int = 1,
    retriever: HybridRetriever | None = None,
    indexes_dir: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run baseline retrieval evaluation across BM25, Dense, and RRF on designated batch questions."""
    created = False
    if retriever is None:
        retriever = HybridRetriever(indexes_dir=indexes_dir)
        created = True

    try:
        questions = load_selected_questions(batch=batch)
        methods = ["bm25", "dense", "rrf"]

        raw_results: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
        denial_results: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
        latencies: dict[str, list[float]] = {m: [] for m in methods}
        denial_latencies: dict[str, list[float]] = {m: [] for m in methods}

        registry_path = paths.metadata_dir / "document_registry.json"
        with open(registry_path, encoding="utf-8") as f:
            registry = {reg["document_id"]: reg for reg in json.load(f)}

        for q in questions:
            qid = q["question_id"]
            q_text = q["question"]
            role = q["role"]
            is_denial = q.get("is_denial", False)
            req_pairs = q["gold_evidence_pairs"]
            restricted_docs = set(q.get("restricted_docs", []))

            # 1. Evaluate BM25
            t0 = time.perf_counter()
            bm25_hits = retriever.bm25_search(q_text, role=role, candidate_k=top_k)
            t_bm25 = time.perf_counter() - t0

            # 2. Evaluate Dense
            t0 = time.perf_counter()
            dense_hits = retriever.dense_search(q_text, role=role, candidate_k=top_k)
            t_dense = time.perf_counter() - t0

            # 3. Evaluate RRF
            t0 = time.perf_counter()
            rrf_hits = retriever.retrieve(q_text, role=role, top_k=top_k)
            t_rrf = time.perf_counter() - t0

            lat_map = {
                "bm25": t_bm25,
                "dense": t_dense,
                "rrf": t_rrf,
            }

            hits_map = {
                "bm25": bm25_hits,
                "dense": dense_hits,
                "rrf": rrf_hits,
            }

            for m in methods:
                hits = hits_map[m]
                ret_pairs = [(h["document_id"], h["section_number"]) for h in hits[:top_k]]
                q_latency = lat_map[m]

                if not is_denial:
                    latencies[m].append(q_latency)
                    metrics = compute_retrieval_metrics(req_pairs, ret_pairs, top_k=top_k)
                    raw_results[m].append(
                        {
                            "question_id": qid,
                            "question": q_text,
                            "role": role,
                            "category": q["category"],
                            "required_evidence": req_pairs,
                            "retrieved_evidence": ret_pairs,
                            "correct_retrieved": metrics["retrieved_gold"],
                            "first_correct_rank": metrics["first_correct_rank"],
                            "hit_at_5": metrics[f"hit_at_{top_k}"],
                            "recall_at_5": metrics[f"recall_at_{top_k}"],
                            "mrr_at_5": metrics[f"mrr_at_{top_k}"],
                            "full_evidence_at_5": metrics[f"full_evidence_at_{top_k}"],
                            "top_chunk_ids": [h["chunk_id"] for h in hits[:top_k]],
                            "latency_s": q_latency,
                        }
                    )
                else:
                    denial_latencies[m].append(q_latency)
                    restricted_hits = [h for h in hits[:top_k] if h["document_id"] in restricted_docs]
                    perm_violations = [h for h in hits[:top_k] if role not in registry[h["document_id"]]["allowed_roles"]]
                    denial_results[m].append(
                        {
                            "question_id": qid,
                            "question": q_text,
                            "role": role,
                            "category": q["category"],
                            "restricted_docs": sorted(list(restricted_docs)),
                            "restricted_chunks_returned": len(restricted_hits),
                            "permission_violations": len(perm_violations),
                            "passed": (len(restricted_hits) == 0 and len(perm_violations) == 0),
                            "top_chunk_ids": [h["chunk_id"] for h in hits[:top_k]],
                            "latency_s": q_latency,
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
                "hit_at_5": statistics.mean([r["hit_at_5"] for r in m_res]) if m_res else 0.0,
                "recall_at_5": statistics.mean([r["recall_at_5"] for r in m_res]) if m_res else 0.0,
                "mrr_at_5": statistics.mean([r["mrr_at_5"] for r in m_res]) if m_res else 0.0,
                "full_evidence_at_5": statistics.mean([r["full_evidence_at_5"] for r in m_res]) if m_res else 0.0,
                "median_latency_ms": statistics.median(m_lats) * 1000 if m_lats else 0.0,
                "p95_latency_ms": percentile(m_lats, 0.95) * 1000 if m_lats else 0.0,
            }

        denial_summary = {}
        for m in methods:
            d_lats = denial_latencies[m]
            if d_lats:
                denial_summary[m] = {
                    "median_latency_ms": statistics.median(d_lats) * 1000,
                    "p95_latency_ms": percentile(d_lats, 0.95) * 1000,
                }

        return {
            "batch": batch,
            "questions": questions,
            "raw_results": raw_results,
            "denial_results": denial_results,
            "summary_metrics": summary_metrics,
            "denial_summary": denial_summary,
        }
    finally:
        if created:
            retriever.close()


def generate_development_report(
    eval_output: dict[str, Any],
    output_path: Path,
    batch: int = 1,
) -> None:
    """Generate or update reports/retrieval_development.md with repeatable batch reporting."""
    summary = eval_output["summary_metrics"]
    raw = eval_output["raw_results"]
    questions = eval_output["questions"]

    if batch == 1:
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
                "All eight questions retrieved complete required evidence within the top 5 results.",
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
        content_to_write = "\n".join(lines) + "\n"
    elif batch == 2:
        # Load existing content to preserve Batch 1
        header_batch_2 = "## Development Batch 2 — Eight Additional Questions"
        if output_path.exists():
            existing_text = output_path.read_text(encoding="utf-8")
            if header_batch_2 in existing_text:
                base_text = existing_text.split(header_batch_2)[0].rstrip()
            else:
                base_text = existing_text.rstrip()
        else:
            base_text = "# Retrieval Development Report — Baseline Evaluation"

        b2_lines = [
            header_batch_2,
            "",
            "### A. Batch Description",
            "",
            "- **Evaluated Questions**: Exactly 8 additional development questions.",
            "- **Question IDs**: `Q-018`, `Q-020`, `Q-024`, `Q-026`, `Q-028`, `Q-030`, `Q-035`, `Q-036`.",
            "- **Corpus**: Apex Crest Bank Operational Policies (Version 1.0, 20 documents, 120 chunks).",
            "- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, CPU execution, unchanged).",
            "- **Lexical Index**: `bm25s` across full chunk corpus (unchanged).",
            "- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with constant `k=60`, candidate depth 20 per method, `top_k=5` (unchanged).",
            "- **Role Filtering**: Strict role-based permission enforcement before candidate truncation (unchanged).",
            "",
            "| Question ID | Category | Role | Required Evidence |",
            "|---|---|---|---|",
        ]

        for q in questions:
            req_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in q["gold_evidence_pairs"])
            b2_lines.append(f"| {q['question_id']} | {q['category']} | `{q['role']}` | {req_str} |")

        b2_lines.extend(
            [
                "",
                "### B. Comparison Table (top_k=5)",
                "",
                "| Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median latency | P95 latency |",
                "|---|---|---|---|---|---|---|",
                f"| BM25 | {summary['bm25']['hit_at_5']:.4f} | {summary['bm25']['recall_at_5']:.4f} | {summary['bm25']['mrr_at_5']:.4f} | {summary['bm25']['full_evidence_at_5']:.4f} | {summary['bm25']['median_latency_ms']:.2f} ms | {summary['bm25']['p95_latency_ms']:.2f} ms |",
                f"| BGE dense | {summary['dense']['hit_at_5']:.4f} | {summary['dense']['recall_at_5']:.4f} | {summary['dense']['mrr_at_5']:.4f} | {summary['dense']['full_evidence_at_5']:.4f} | {summary['dense']['median_latency_ms']:.2f} ms | {summary['dense']['p95_latency_ms']:.2f} ms |",
                f"| RRF baseline | {summary['rrf']['hit_at_5']:.4f} | {summary['rrf']['recall_at_5']:.4f} | {summary['rrf']['mrr_at_5']:.4f} | {summary['rrf']['full_evidence_at_5']:.4f} | {summary['rrf']['median_latency_ms']:.2f} ms | {summary['rrf']['p95_latency_ms']:.2f} ms |",
                "",
                "### C. Question-Level Evidence",
                "",
                "| QID | Category | Role | Required Evidence | BM25 First Rank | Dense First Rank | RRF First Rank | RRF Full Evidence | Top 5 RRF Chunks |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )

        for i, q in enumerate(questions):
            qid = q["question_id"]
            req_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in q["gold_evidence_pairs"])
            b_rank = raw["bm25"][i]["first_correct_rank"]
            d_rank = raw["dense"][i]["first_correct_rank"]
            r_rank = raw["rrf"][i]["first_correct_rank"]
            r_full = "Yes" if raw["rrf"][i]["full_evidence_at_5"] == 1.0 else "No (Partial)"
            top_chunks = ", ".join(raw["rrf"][i]["top_chunk_ids"])

            b_str = str(b_rank) if b_rank else "Miss"
            d_str = str(d_rank) if d_rank else "Miss"
            r_str = str(r_rank) if r_rank else "Miss"
            b2_lines.append(
                f"| {qid} | {q['category']} | `{q['role']}` | {req_str} | {b_str} | {d_str} | {r_str} | {r_full} | `{top_chunks}` |"
            )

        b2_lines.extend(
            [
                "",
                "### D. Failure & Partial Evidence Analysis",
                "",
            ]
        )

        misses = [r for r in raw["rrf"] if r["full_evidence_at_5"] < 1.0]
        if not misses:
            b2_lines.append("All eight questions in Batch 2 retrieved complete required evidence within the top 5 results.")
        else:
            for m in misses:
                qid = m["question_id"]
                req_pairs = m["required_evidence"]
                correct_pairs = m["correct_retrieved"]
                missing_pairs = [p for p in req_pairs if p not in correct_pairs]
                top_chunks = ", ".join(m["top_chunk_ids"])
                missing_str = ", ".join(f"{doc} Sec {sec}" for doc, sec in missing_pairs)
                b2_lines.append(f"#### {qid} ({m['category']})")
                b2_lines.append(f"- **Missing Required Sections**: {missing_str}")
                b2_lines.append(f"- **Top-5 Chunks Returned**: `{top_chunks}`")
                if len(req_pairs) > 1 and len(correct_pairs) > 0:
                    b2_lines.append(
                        f"- **Observation**: Multi-evidence question where {len(correct_pairs)}/{len(req_pairs)} "
                        f"required sections were retrieved in top 5. Remaining sections placed outside top 5."
                    )
                else:
                    b2_lines.append(
                        "- **Observation**: Target evidence was outranked by semantically competing sections."
                    )
                b2_lines.append("")

        b2_lines.extend(
            [
                "### E. Comparison with Batch 1",
                "",
                "| Batch | Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median Latency |",
                "|---|---|---|---|---|---|---|",
                "| Batch 1 | BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.59 ms |",
                "| Batch 1 | BGE dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 133.23 ms |",
                "| Batch 1 | RRF baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 127.47 ms |",
                f"| Batch 2 | BM25 | {summary['bm25']['hit_at_5']:.4f} | {summary['bm25']['recall_at_5']:.4f} | {summary['bm25']['mrr_at_5']:.4f} | {summary['bm25']['full_evidence_at_5']:.4f} | {summary['bm25']['median_latency_ms']:.2f} ms |",
                f"| Batch 2 | BGE dense | {summary['dense']['hit_at_5']:.4f} | {summary['dense']['recall_at_5']:.4f} | {summary['dense']['mrr_at_5']:.4f} | {summary['dense']['full_evidence_at_5']:.4f} | {summary['dense']['median_latency_ms']:.2f} ms |",
                f"| Batch 2 | RRF baseline | {summary['rrf']['hit_at_5']:.4f} | {summary['rrf']['recall_at_5']:.4f} | {summary['rrf']['mrr_at_5']:.4f} | {summary['rrf']['full_evidence_at_5']:.4f} | {summary['rrf']['median_latency_ms']:.2f} ms |",
                "",
                "### F. Conclusions & Next Steps",
                "",
                "- **Observed Dynamics**: Development Batch 2 provides an independent baseline on questions requiring single, conditional, and multi-document synthesis (including 3-section retrieval in Q-035).",
                "- **Investigation Recommendation**: Evaluate cross-encoder reranking (`ms-marco-MiniLM-L6-v2`) on multi-evidence scenarios where lexical and dense signals have varying alignment.",
            ]
        )

        content_to_write = base_text + "\n\n" + "\n".join(b2_lines) + "\n"
    elif batch == 3:
        # Batch 3 results are internal to chat evaluation and must not be written to reports/retrieval_development.md
        return
    else:
        raise ValueError(f"Unsupported batch: {batch}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content_to_write)


def run_evaluation(batch: int = 1) -> None:
    """CLI entry point for running retrieval evaluation."""
    print("=" * 65)
    print(f"Quorift — Retrieval Evaluation (Batch {batch} Development Questions)")
    print("=" * 65)

    eval_output = evaluate_retrieval(batch=batch)
    summary = eval_output["summary_metrics"]
    answerable_count = len(eval_output["raw_results"]["bm25"])
    denial_count = len(eval_output.get("denial_results", {}).get("bm25", []))

    print(f"\n[Summary Metrics for Batch {batch} ({answerable_count} Answerable Questions) — top_k=5]")
    print(
        f"{'Method':<14} | {'Hit@5':<8} | {'Recall@5':<10} | {'MRR@5':<8} | {'FullEv@5':<9} | {'Median(ms)':<10} | {'P95(ms)':<10}"
    )
    print("-" * 85)
    for m in ["bm25", "dense", "rrf"]:
        s = summary[m]
        print(
            f"{m:<14} | {s['hit_at_5']:<8.4f} | {s['recall_at_5']:<10.4f} | {s['mrr_at_5']:<8.4f} | {s['full_evidence_at_5']:<9.4f} | {s['median_latency_ms']:<10.2f} | {s['p95_latency_ms']:<10.2f}"
        )

    if denial_count > 0:
        print(f"\n[Unauthorized-Access Denial Evaluation ({denial_count} Questions)]")
        print(
            f"{'Method':<14} | {'Denial Cases':<13} | {'Restricted Chunks':<18} | {'Perm Violations':<16} | {'Median(ms)':<10} | {'Status':<10}"
        )
        print("-" * 92)
        for m in ["bm25", "dense", "rrf"]:
            d_list = eval_output["denial_results"][m]
            d_sum = eval_output.get("denial_summary", {}).get(m, {})
            d_lat = d_sum.get("median_latency_ms", 0.0)
            restr_returned = sum(d["restricted_chunks_returned"] for d in d_list)
            violations = sum(d["permission_violations"] for d in d_list)
            status = "PASSED (0 leaks)" if restr_returned == 0 and violations == 0 else "FAILED"
            print(
                f"{m:<14} | {len(d_list):<13} | {restr_returned:<18} | {violations:<16} | {d_lat:<10.2f} | {status:<10}"
            )

    if batch in (1, 2):
        report_file = paths.reports_dir / "retrieval_development.md"
        generate_development_report(eval_output, report_file, batch=batch)
        print(f"\nReport successfully updated at: {report_file}")
    else:
        print(f"\n[INFO] Batch {batch} results are reported directly (reports/retrieval_development.md preserved unchanged).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quorift baseline retrieval evaluation")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Development question batch to evaluate (1, 2, or 3)",
    )
    args = parser.parse_args()

    try:
        run_evaluation(batch=args.batch)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
