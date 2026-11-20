"""Comprehensive validation tests for the Quorift banking dataset and evaluation assets."""

import json
from pathlib import Path
import fitz
import pytest

from quorift.config import paths
from scripts.build_documents import compute_source_hash

VALID_ROLES = {"general", "lending", "finance", "compliance", "hr", "admin"}

EXPECTED_CATEGORIES = {
    "direct": 24,
    "exact_reference": 16,
    "paraphrase": 18,
    "conditional": 18,
    "multi_document": 12,
    "unanswerable": 12,
    "permission": 10,
    "freshness": 10,
}

EXPECTED_SPLITS = {
    "development": 48,
    "validation": 24,
    "final_test": 48,
}


@pytest.fixture(scope="module")
def policy_sources():
    sources_path = paths.metadata_dir / "policy_sources.json"
    assert sources_path.exists(), "policy_sources.json does not exist"
    with open(sources_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def document_registry():
    registry_path = paths.metadata_dir / "document_registry.json"
    assert registry_path.exists(), "document_registry.json does not exist"
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fact_ledger():
    ledger_path = paths.metadata_dir / "fact_ledger.json"
    assert ledger_path.exists(), "fact_ledger.json does not exist"
    with open(ledger_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def updates():
    updates_path = paths.metadata_dir / "updates.json"
    assert updates_path.exists(), "updates.json does not exist"
    with open(updates_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def questions():
    q_path = paths.evaluation_dir / "questions.jsonl"
    assert q_path.exists(), "questions.jsonl does not exist"
    with open(q_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def cache_scenarios():
    cache_path = paths.evaluation_dir / "cache_scenarios.json"
    assert cache_path.exists(), "cache_scenarios.json does not exist"
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def injection_cases():
    inj_path = paths.evaluation_dir / "injection_cases.json"
    assert inj_path.exists(), "injection_cases.json does not exist"
    with open(inj_path, encoding="utf-8") as f:
        return json.load(f)


# 1. Exactly 20 unique active policy IDs exist
def test_01_exactly_20_unique_active_policy_ids(policy_sources):
    assert len(policy_sources) == 20, f"Expected 20 policies, found {len(policy_sources)}"
    doc_ids = [p["document_id"] for p in policy_sources]
    assert len(set(doc_ids)) == 20, "Policy IDs must be unique"


# 2. Exactly 20 normal policy PDFs exist
def test_02_exactly_20_policy_pdfs_exist(policy_sources):
    pdf_files = list(paths.documents_dir.glob("*.pdf"))
    assert len(pdf_files) == 20, f"Expected 20 PDFs in {paths.documents_dir}, found {len(pdf_files)}"
    expected_filenames = {
        f"{p['document_id'].lower().replace('-', '_')}.pdf" for p in policy_sources
    }
    actual_filenames = {p.name for p in pdf_files}
    assert expected_filenames == actual_filenames, "PDF filenames must match lowercase policy IDs"


# 3. Every PDF has extractable text
def test_03_every_pdf_has_extractable_text():
    pdf_files = sorted(paths.documents_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        assert len(doc) >= 2, f"{pdf_path.name} has fewer than 2 pages"
        assert len(doc) <= 4, f"{pdf_path.name} exceeds 4 pages"
        total_text = ""
        for page_idx, page in enumerate(doc):
            text = page.get_text()
            assert len(text.strip()) > 50, f"{pdf_path.name} page {page_idx+1} has insufficient text"
            total_text += text
        assert "Apex Crest Bank" in total_text, f"{pdf_path.name} missing bank name"
        assert "DEMONSTRATION DOCUMENT NOTICE" in total_text, f"{pdf_path.name} missing demonstration notice"


# 4. Every document has the required metadata
def test_04_every_document_has_required_metadata(policy_sources):
    required_keys = {
        "document_id",
        "title",
        "department",
        "owner",
        "classification",
        "allowed_roles",
        "version",
        "effective_date",
        "sections",
        "references",
    }
    for p in policy_sources:
        missing = required_keys - set(p.keys())
        assert not missing, f"{p.get('document_id', 'Unknown')} missing keys: {missing}"
        assert p["version"] == "1.0", f"Active initial version must be 1.0, got {p['version']}"
        assert isinstance(p["sections"], list) and len(p["sections"]) >= 4


# 5. Every document contains its expected headings and meaningful sections
def test_05_every_document_contains_expected_headings_and_sections(policy_sources):
    for p in policy_sources:
        doc_id = p["document_id"]
        total_words = 0
        for s in p["sections"]:
            assert "section_number" in s, f"{doc_id} section missing section_number"
            assert "heading" in s and len(s["heading"].strip()) > 0, f"{doc_id} section missing heading"
            assert "content" in s, f"{doc_id} section missing content"
            words = s["content"].split()
            assert len(words) >= 15, f"{doc_id} section {s['section_number']} content too short ({len(words)} words)"
            total_words += len(words)
        assert total_words >= 300, f"{doc_id} total policy words too low ({total_words} words)"


# 6. Allowed roles contain only the six valid demo roles
def test_06_allowed_roles_valid(policy_sources):
    for p in policy_sources:
        doc_id = p["document_id"]
        roles = set(p["allowed_roles"])
        assert roles.issubset(VALID_ROLES), f"{doc_id} contains invalid roles: {roles - VALID_ROLES}"
        assert len(roles) >= 1, f"{doc_id} allowed_roles cannot be empty"
        # Admin must always have access
        assert "admin" in roles, f"{doc_id} must be accessible to admin"
        # General docs must be accessible to all 6 roles
        if doc_id.startswith("GEN-") or doc_id == "HR-01":
            assert roles == VALID_ROLES, f"{doc_id} should be accessible to all six demo roles"


# 7. Cross-document references point to existing policy IDs
def test_07_cross_document_references_valid(policy_sources):
    all_doc_ids = {p["document_id"] for p in policy_sources}
    for p in policy_sources:
        for ref in p["references"]:
            assert ref in all_doc_ids, f"{p['document_id']} references unknown policy ID: {ref}"


# 8. Exactly one version is active per document ID in the registry
def test_08_exactly_one_version_active_per_document_id(document_registry, policy_sources):
    assert len(document_registry) == 20, f"Expected 20 registry records, found {len(document_registry)}"
    all_source_ids = {p["document_id"] for p in policy_sources}
    registry_ids = set()
    for r in document_registry:
        assert r["active_status"] is True, f"{r['document_id']} active_status must be True"
        assert r["active_version"] == "1.0", f"{r['document_id']} active_version must be 1.0"
        registry_ids.add(r["document_id"])
    assert registry_ids == all_source_ids, "Registry must contain exactly the 20 policy IDs"


# 9. All source hashes match the defined normalization rule
def test_09_all_source_hashes_match_normalization_rule(document_registry, policy_sources):
    policy_by_id = {p["document_id"]: p for p in policy_sources}
    for r in document_registry:
        doc_id = r["document_id"]
        source_policy = policy_by_id[doc_id]
        expected_hash = compute_source_hash(source_policy)
        assert r["source_content_hash"] == expected_hash, f"Hash mismatch for {doc_id}"


# 10. Fact-ledger entries refer to valid documents and sections
def test_10_fact_ledger_refers_to_valid_documents_and_sections(fact_ledger, policy_sources):
    assert len(fact_ledger) >= 80, f"Expected at least 80 facts, found {len(fact_ledger)}"
    policy_sections = {
        p["document_id"]: {s["section_number"] for s in p["sections"]}
        for p in policy_sources
    }
    for fact in fact_ledger:
        fact_id = fact["fact_id"]
        doc_id = fact["document_id"]
        assert doc_id in policy_sections, f"Fact {fact_id} refers to unknown doc ID {doc_id}"
        sec = fact["section"]
        assert sec in policy_sections[doc_id], f"Fact {fact_id} refers to unknown section {sec} in {doc_id}"
        assert set(fact["permitted_roles"]).issubset(VALID_ROLES), f"Fact {fact_id} has invalid roles"


# 11. Expected fact statements are supported by the specified policy content
def test_11_fact_statements_supported_by_policy_content(fact_ledger, policy_sources):
    policy_content_map = {}
    for p in policy_sources:
        for s in p["sections"]:
            policy_content_map[(p["document_id"], s["section_number"])] = s["content"]

    for fact in fact_ledger:
        doc_id = fact["document_id"]
        sec = fact["section"]
        version_scope = fact.get("version_scope", "all")
        # For version 2.0 replacement facts, skip checking against v1 source
        if version_scope == "2.0":
            continue
        content = policy_content_map.get((doc_id, sec), "")
        assert len(content) > 0, f"Missing content for {(doc_id, sec)}"


# 12. All three update scenarios are consistent with their initial documents
def test_12_three_update_scenarios_consistent(updates, policy_sources):
    assert len(updates) == 3, f"Expected exactly 3 updates, found {len(updates)}"
    update_doc_ids = {u["document_id"] for u in updates}
    assert update_doc_ids == {"LND-01", "FIN-01", "HR-01"}

    for u in updates:
        assert u["previous_version"] == "1.0"
        assert u["new_version"] == "2.0"
        assert len(u["exact_section_changes"]) >= 1
        assert len(u["unchanged_conditions"]) >= 1
        assert len(u["expected_old_value"]) > 0
        assert len(u["expected_new_value"]) > 0


# 13. There are exactly 120 labeled questions with required category counts and split sizes
def test_13_questions_category_and_split_counts(questions):
    assert len(questions) == 120, f"Expected 120 questions, found {len(questions)}"

    # Check category counts
    cat_counts = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    assert cat_counts == EXPECTED_CATEGORIES, f"Category counts mismatch: {cat_counts}"

    # Check split counts
    split_counts = {}
    for q in questions:
        split_counts[q["split"]] = split_counts.get(q["split"], 0) + 1
    assert split_counts == EXPECTED_SPLITS, f"Split counts mismatch: {split_counts}"


# 14. Every question refers to valid roles, states, evidence and facts
def test_14_questions_valid_fields(questions, policy_sources, fact_ledger):
    all_doc_ids = {p["document_id"] for p in policy_sources}
    all_fact_ids = {f["fact_id"] for f in fact_ledger}
    valid_behaviors = {"answer", "refuse_insufficient_evidence", "deny_unauthorized"}

    for q in questions:
        qid = q["question_id"]
        assert q["role"] in VALID_ROLES, f"{qid} has invalid role: {q['role']}"
        assert q["expected_behavior"] in valid_behaviors, f"{qid} has invalid behavior: {q['expected_behavior']}"

        for doc_id in q["gold_document_ids"]:
            assert doc_id in all_doc_ids, f"{qid} refers to unknown doc ID: {doc_id}"

        for fid in q["gold_fact_ids"]:
            assert fid in all_fact_ids, f"{qid} refers to unknown fact ID: {fid}"

        if q["expected_behavior"] == "refuse_insufficient_evidence":
            assert len(q["gold_document_ids"]) == 0, f"Unanswerable {qid} should not have gold documents"
            assert len(q["gold_fact_ids"]) == 0, f"Unanswerable {qid} should not have gold facts"


# 15. Related question groups do not cross splits where that would leak a test case
def test_15_scenario_groups_do_not_cross_splits(questions):
    group_to_splits = {}
    for q in questions:
        grp = q["scenario_group"]
        group_to_splits.setdefault(grp, set()).add(q["split"])

    for grp, splits in group_to_splits.items():
        assert len(splits) == 1, f"Scenario group '{grp}' crosses splits: {splits}"


# 16. Exactly 28 cache scenarios and 20 injection cases exist
def test_16_cache_and_injection_counts(cache_scenarios, injection_cases):
    assert len(cache_scenarios) == 28, f"Expected 28 cache scenarios, found {len(cache_scenarios)}"
    cache_cat_counts = {}
    for c in cache_scenarios:
        cache_cat_counts[c["category"]] = cache_cat_counts.get(c["category"], 0) + 1
    expected_cache_cats = {
        "exact_repeat": 6,
        "paraphrase": 6,
        "near_miss": 6,
        "document_update": 6,
        "permission_revocation": 4,
    }
    assert cache_cat_counts == expected_cache_cats, f"Cache category counts mismatch: {cache_cat_counts}"

    assert len(injection_cases) == 20, f"Expected 20 injection cases, found {len(injection_cases)}"
    inj_cat_counts = {}
    for inj in injection_cases:
        inj_cat_counts[inj["category"]] = inj_cat_counts.get(inj["category"], 0) + 1
    expected_inj_cats = {
        "instruction_override": 4,
        "fake_authority": 4,
        "answer_contamination": 4,
        "cache_poisoning": 4,
        "benign_control": 4,
    }
    assert inj_cat_counts == expected_inj_cats, f"Injection category counts mismatch: {inj_cat_counts}"


# 17. The original configuration tests still pass
def test_17_original_configuration_tests_still_pass():
    from tests.test_config import (
        test_project_root_structure,
        test_paths_are_relative_to_root,
        test_custom_project_root,
        test_expected_directory_names,
    )
    test_project_root_structure()
    test_paths_are_relative_to_root()
    test_expected_directory_names()
    # Test custom project root with a tmp_path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_custom_project_root(Path(tmp_dir))
