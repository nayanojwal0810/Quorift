"""Tests for baseline retrieval evaluation logic, metrics calculation, and gold evidence matching."""

import pytest

from quorift.retrieval_eval import (
    BATCH_1_QIDS,
    BATCH_2_QIDS,
    BATCH_3_QIDS,
    compute_retrieval_metrics,
    evaluate_retrieval,
    generate_development_report,
    load_selected_questions,
)


def test_selected_questions_validation_batch_1():
    """Verify that load_selected_questions loads exactly Batch 1 questions."""
    questions = load_selected_questions(batch=1)
    assert len(questions) == 8
    qids = [q["question_id"] for q in questions]
    assert qids == BATCH_1_QIDS

    for q in questions:
        assert q["split"] == "development"
        assert q["corpus_state"] == "v1_active"
        assert q["expected_behavior"] == "answer"
        assert len(q["gold_evidence_pairs"]) >= 1
        for doc_id, sec_num in q["gold_evidence_pairs"]:
            assert isinstance(doc_id, str) and len(doc_id) in (5, 6)
            assert isinstance(sec_num, str) and "." in sec_num


def test_selected_questions_validation_batch_2():
    """Verify that load_selected_questions loads exactly Batch 2 questions."""
    questions = load_selected_questions(batch=2)
    assert len(questions) == 8
    qids = [q["question_id"] for q in questions]
    assert qids == BATCH_2_QIDS

    for q in questions:
        assert q["split"] == "development"
        assert q["corpus_state"] == "v1_active"
        assert q["expected_behavior"] == "answer"
        assert len(q["gold_evidence_pairs"]) >= 1
        for doc_id, sec_num in q["gold_evidence_pairs"]:
            assert isinstance(doc_id, str) and len(doc_id) in (5, 6)
            assert isinstance(sec_num, str) and "." in sec_num


def test_selected_questions_validation_batch_3():
    """Verify that load_selected_questions loads exactly Batch 3 questions (6 answerable, 2 denial)."""
    questions = load_selected_questions(batch=3)
    assert len(questions) == 8
    qids = [q["question_id"] for q in questions]
    assert qids == BATCH_3_QIDS

    answerable = [q for q in questions if q["expected_behavior"] == "answer"]
    denials = [q for q in questions if q["expected_behavior"] == "deny_unauthorized"]

    assert len(answerable) == 6
    assert len(denials) == 2
    assert [q["question_id"] for q in denials] == ["Q-041", "Q-043"]

    for q in questions:
        assert q["split"] == "development"
        assert q["corpus_state"] == "v1_active"
        assert len(q["gold_evidence_pairs"]) >= 1

    for q in answerable:
        assert q["is_denial"] is False

    for q in denials:
        assert q["is_denial"] is True
        assert len(q["restricted_docs"]) >= 1


def test_selected_questions_invalid_batch():
    """Verify that an invalid batch parameter raises a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid batch"):
        load_selected_questions(batch=4)


# 1. Single-evidence hit
def test_metric_single_evidence_hit():
    gold = [("LND-01", "2.0")]
    retrieved = [("LND-01", "2.0"), ("HR-01", "2.0"), ("FIN-01", "2.0")]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr_at_5"] == 1.0
    assert m["full_evidence_at_5"] == 1.0
    assert m["first_correct_rank"] == 1
    assert m["retrieved_gold"] == [("LND-01", "2.0")]


# 2. Single-evidence miss
def test_metric_single_evidence_miss():
    gold = [("LND-01", "2.0")]
    retrieved = [("HR-01", "2.0"), ("FIN-01", "2.0"), ("CMP-01", "2.0")]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 0.0
    assert m["recall_at_5"] == 0.0
    assert m["mrr_at_5"] == 0.0
    assert m["full_evidence_at_5"] == 0.0
    assert m["first_correct_rank"] is None
    assert m["retrieved_gold"] == []


# 3. Multi-evidence partial hit
def test_metric_multi_evidence_partial():
    gold = [("LND-01", "2.0"), ("LND-01", "3.0")]
    retrieved = [
        ("GEN-01", "1.0"),
        ("LND-01", "2.0"),  # Rank 2
        ("HR-01", "2.0"),
        ("FIN-01", "1.0"),
        ("CMP-01", "1.0"),
    ]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 0.5
    assert m["mrr_at_5"] == 0.5
    assert m["full_evidence_at_5"] == 0.0
    assert m["first_correct_rank"] == 2
    assert m["retrieved_gold"] == [("LND-01", "2.0")]


# 4. Multi-evidence complete hit
def test_metric_multi_evidence_complete():
    gold = [("CMP-01", "2.0"), ("LND-04", "2.0")]
    retrieved = [
        ("CMP-01", "2.0"),  # Rank 1
        ("LND-04", "2.0"),  # Rank 2
        ("GEN-01", "1.0"),
        ("HR-01", "1.0"),
        ("FIN-01", "1.0"),
    ]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr_at_5"] == 1.0
    assert m["full_evidence_at_5"] == 1.0
    assert m["first_correct_rank"] == 1
    assert set(m["retrieved_gold"]) == set(gold)


# 5. Correct evidence appearing at rank 5
def test_metric_correct_evidence_at_rank_5():
    gold = [("LND-01", "4.0")]
    retrieved = [
        ("HR-01", "1.0"),
        ("HR-02", "1.0"),
        ("FIN-01", "1.0"),
        ("CMP-01", "1.0"),
        ("LND-01", "4.0"),  # Rank 5
        ("GEN-01", "1.0"),
    ]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr_at_5"] == 1.0 / 5.0
    assert m["full_evidence_at_5"] == 1.0
    assert m["first_correct_rank"] == 5


# 6. Gold evidence appearing only after rank 5
def test_metric_gold_evidence_after_rank_5():
    gold = [("LND-01", "4.0")]
    retrieved = [
        ("HR-01", "1.0"),
        ("HR-02", "1.0"),
        ("FIN-01", "1.0"),
        ("CMP-01", "1.0"),
        ("GEN-01", "1.0"),  # Rank 5
        ("LND-01", "4.0"),  # Rank 6 (beyond cutoff)
    ]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 0.0
    assert m["recall_at_5"] == 0.0
    assert m["mrr_at_5"] == 0.0
    assert m["full_evidence_at_5"] == 0.0
    assert m["first_correct_rank"] is None
    assert m["retrieved_gold"] == []


# 7. Empty retrieved results
def test_metric_empty_retrieved_results():
    gold = [("LND-01", "2.0")]
    retrieved = []
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 0.0
    assert m["recall_at_5"] == 0.0
    assert m["mrr_at_5"] == 0.0
    assert m["full_evidence_at_5"] == 0.0
    assert m["first_correct_rank"] is None
    assert m["retrieved_gold"] == []


# 8. Duplicate retrieved evidence not increasing recall
def test_metric_duplicate_evidence_does_not_increase_recall():
    gold = [("LND-01", "2.0"), ("LND-01", "3.0")]
    # Result contains duplicate LND-01 Sec 2.0
    retrieved = [
        ("LND-01", "2.0"),  # Rank 1
        ("LND-01", "2.0"),  # Rank 2 (duplicate)
        ("HR-01", "1.0"),
        ("FIN-01", "1.0"),
        ("CMP-01", "1.0"),
    ]
    m = compute_retrieval_metrics(gold, retrieved, top_k=5)

    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 0.5  # Exactly 1 of 2 unique required, duplicate did not inflate
    assert m["full_evidence_at_5"] == 0.0
    assert len(m["retrieved_gold"]) == 1


# 9. Batch 3 evaluation metrics and denial integrity
def test_batch_3_evaluation_metrics_and_denial_results():
    """Verify that Batch 3 evaluates 6 answerable questions and 2 denial cases with 0 leaks."""
    eval_output = evaluate_retrieval(batch=3)

    assert eval_output["batch"] == 3
    assert len(eval_output["questions"]) == 8
    assert len(eval_output["raw_results"]["bm25"]) == 6
    assert len(eval_output["raw_results"]["dense"]) == 6
    assert len(eval_output["raw_results"]["rrf"]) == 6

    assert len(eval_output["denial_results"]["bm25"]) == 2
    assert len(eval_output["denial_results"]["dense"]) == 2
    assert len(eval_output["denial_results"]["rrf"]) == 2

    for m in ["bm25", "dense", "rrf"]:
        summary = eval_output["summary_metrics"][m]
        assert "hit_at_5" in summary
        assert "recall_at_5" in summary
        assert "mrr_at_5" in summary
        assert "full_evidence_at_5" in summary

        for denial in eval_output["denial_results"][m]:
            assert denial["restricted_chunks_returned"] == 0
            assert denial["permission_violations"] == 0
            assert denial["passed"] is True


# 10. Batch 3 must not modify reports/retrieval_development.md
def test_batch_3_does_not_modify_development_report(tmp_path):
    """Calling generate_development_report with batch=3 must leave reports untouched."""
    dummy_report = tmp_path / "retrieval_development.md"
    dummy_report.write_text("# Existing Report\n", encoding="utf-8")

    dummy_eval_output = {
        "batch": 3,
        "questions": [],
        "raw_results": {"bm25": [], "dense": [], "rrf": []},
        "denial_results": {"bm25": [], "dense": [], "rrf": []},
        "summary_metrics": {},
    }

    generate_development_report(dummy_eval_output, dummy_report, batch=3)
    assert dummy_report.read_text(encoding="utf-8") == "# Existing Report\n"
