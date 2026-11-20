"""Tests for baseline retrieval evaluation logic, metrics calculation, and gold evidence matching."""

import pytest

from quorift.retrieval_eval import SELECTED_QIDS, load_selected_questions


def test_selected_questions_validation():
    """Verify that load_selected_questions loads exactly the 8 specified questions."""
    questions = load_selected_questions()
    assert len(questions) == 8
    qids = [q["question_id"] for q in questions]
    assert qids == SELECTED_QIDS

    # Verify each question has valid fields and gold evidence pairs
    for q in questions:
        assert q["split"] == "development"
        assert q["corpus_state"] == "v1_active"
        assert q["expected_behavior"] == "answer"
        assert len(q["gold_evidence_pairs"]) >= 1
        for doc_id, sec_num in q["gold_evidence_pairs"]:
            assert isinstance(doc_id, str) and len(doc_id) in (5, 6)
            assert isinstance(sec_num, str) and "." in sec_num


def test_metric_calculation_single_evidence_hit():
    """Verify metrics for single-evidence hit at rank 1."""
    gold_pairs = [("LND-01", "2.0")]
    retrieved_pairs = [("LND-01", "2.0"), ("HR-01", "2.0"), ("FIN-01", "2.0")]

    hit_at_5 = 1.0 if any(p in retrieved_pairs for p in gold_pairs) else 0.0
    retrieved_gold = [p for p in gold_pairs if p in retrieved_pairs]
    recall_at_5 = len(retrieved_gold) / len(gold_pairs)
    full_evidence_at_5 = 1.0 if len(retrieved_gold) == len(gold_pairs) else 0.0

    first_rank = None
    for idx, p in enumerate(retrieved_pairs, start=1):
        if p in gold_pairs:
            first_rank = idx
            break
    mrr_at_5 = 1.0 / first_rank if first_rank else 0.0

    assert hit_at_5 == 1.0
    assert recall_at_5 == 1.0
    assert full_evidence_at_5 == 1.0
    assert mrr_at_5 == 1.0


def test_metric_calculation_single_evidence_miss():
    """Verify metrics when no gold evidence is retrieved."""
    gold_pairs = [("LND-01", "2.0")]
    retrieved_pairs = [("HR-01", "2.0"), ("FIN-01", "2.0"), ("CMP-01", "2.0")]

    hit_at_5 = 1.0 if any(p in retrieved_pairs for p in gold_pairs) else 0.0
    retrieved_gold = [p for p in gold_pairs if p in retrieved_pairs]
    recall_at_5 = len(retrieved_gold) / len(gold_pairs)
    full_evidence_at_5 = 1.0 if len(retrieved_gold) == len(gold_pairs) else 0.0

    first_rank = None
    for idx, p in enumerate(retrieved_pairs, start=1):
        if p in gold_pairs:
            first_rank = idx
            break
    mrr_at_5 = 1.0 / first_rank if first_rank else 0.0

    assert hit_at_5 == 0.0
    assert recall_at_5 == 0.0
    assert full_evidence_at_5 == 0.0
    assert mrr_at_5 == 0.0


def test_metric_calculation_multi_evidence_partial():
    """Verify metrics when 1 of 2 required evidence pairs is retrieved at rank 2."""
    gold_pairs = [("LND-01", "2.0"), ("LND-01", "3.0")]
    retrieved_pairs = [
        ("GEN-01", "1.0"),
        ("LND-01", "2.0"),  # Rank 2
        ("HR-01", "2.0"),
        ("FIN-01", "1.0"),
        ("CMP-01", "1.0"),
    ]

    hit_at_5 = 1.0 if any(p in retrieved_pairs for p in gold_pairs) else 0.0
    retrieved_gold = [p for p in gold_pairs if p in retrieved_pairs]
    recall_at_5 = len(retrieved_gold) / len(gold_pairs)
    full_evidence_at_5 = 1.0 if len(retrieved_gold) == len(gold_pairs) else 0.0

    first_rank = None
    for idx, p in enumerate(retrieved_pairs, start=1):
        if p in gold_pairs:
            first_rank = idx
            break
    mrr_at_5 = 1.0 / first_rank if first_rank else 0.0

    assert hit_at_5 == 1.0
    assert recall_at_5 == 0.5
    assert full_evidence_at_5 == 0.0  # Partial evidence: only 1 of 2
    assert mrr_at_5 == 0.5  # First correct rank is 2 -> 1/2


def test_metric_calculation_multi_evidence_complete():
    """Verify metrics when all multi-section evidence pairs are retrieved."""
    gold_pairs = [("CMP-01", "2.0"), ("LND-04", "2.0")]
    retrieved_pairs = [
        ("CMP-01", "2.0"),  # Rank 1
        ("LND-04", "2.0"),  # Rank 2
        ("GEN-01", "1.0"),
        ("HR-01", "1.0"),
        ("FIN-01", "1.0"),
    ]

    hit_at_5 = 1.0 if any(p in retrieved_pairs for p in gold_pairs) else 0.0
    retrieved_gold = [p for p in gold_pairs if p in retrieved_pairs]
    recall_at_5 = len(retrieved_gold) / len(gold_pairs)
    full_evidence_at_5 = 1.0 if len(retrieved_gold) == len(gold_pairs) else 0.0

    first_rank = None
    for idx, p in enumerate(retrieved_pairs, start=1):
        if p in gold_pairs:
            first_rank = idx
            break
    mrr_at_5 = 1.0 / first_rank if first_rank else 0.0

    assert hit_at_5 == 1.0
    assert recall_at_5 == 1.0
    assert full_evidence_at_5 == 1.0
    assert mrr_at_5 == 1.0
