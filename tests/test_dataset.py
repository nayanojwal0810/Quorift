"""Comprehensive validation tests for the Quorift banking dataset and evaluation assets."""

import json
from pathlib import Path
import pymupdf
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
        doc = pymupdf.open(pdf_path)
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
        assert "admin" in roles, f"{doc_id} must be accessible to admin"
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


# 10. Fact-ledger entries have unique IDs and refer to valid documents and sections
def test_10_fact_ledger_unique_ids_and_valid_references(fact_ledger, policy_sources):
    assert len(fact_ledger) >= 80, f"Expected at least 80 facts, found {len(fact_ledger)}"
    fact_ids = [f["fact_id"] for f in fact_ledger]
    assert len(fact_ids) == len(set(fact_ids)), "Fact IDs must be strictly unique"

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


# 11. Fact statements and policy values are supported by source and update definitions
def test_11_fact_and_answer_support(fact_ledger, policy_sources, updates):
    policy_content_map = {}
    for p in policy_sources:
        for s in p["sections"]:
            policy_content_map[(p["document_id"], s["section_number"])] = s["content"]

    allowed_scopes = {"all", "1.0", "2.0"}
    for fact in fact_ledger:
        doc_id = fact["document_id"]
        sec = fact["section"]
        version_scope = fact.get("version_scope", "all")
        assert version_scope in allowed_scopes, f"Fact {fact['fact_id']} has unknown scope {version_scope}"

        # Source section content must exist
        content = policy_content_map.get((doc_id, sec), "")
        assert len(content) > 0, f"Missing source content for {(doc_id, sec)}"

    # Check version-specific replacement facts against update records
    facts_by_id = {f["fact_id"]: f for f in fact_ledger}
    assert "FACT-LND-01-01-V1" in facts_by_id and "FACT-LND-01-01-V2" in facts_by_id
    assert "INR 10 lakh" in facts_by_id["FACT-LND-01-01-V1"]["statement"]
    assert "INR 12 lakh" in facts_by_id["FACT-LND-01-01-V2"]["statement"]

    assert "FACT-FIN-01-01-V1" in facts_by_id and "FACT-FIN-01-01-V2" in facts_by_id
    assert "INR 5,000" in facts_by_id["FACT-FIN-01-01-V1"]["statement"]
    assert "INR 7,000" in facts_by_id["FACT-FIN-01-01-V2"]["statement"]

    assert "FACT-HR-01-01-V1" in facts_by_id and "FACT-HR-01-01-V2" in facts_by_id
    assert "18 working days" in facts_by_id["FACT-HR-01-01-V1"]["statement"]
    assert "20 working days" in facts_by_id["FACT-HR-01-01-V2"]["statement"]

    # Verify key approval limits and leave values directly in policy sources
    assert "INR 10 lakh" in policy_content_map[("LND-01", "2.0")]
    assert "INR 5,000" in policy_content_map[("FIN-01", "2.0")]
    assert "18 working days" in policy_content_map[("HR-01", "2.0")]


# 12. All three update scenarios are consistent with their initial documents and future dates
def test_12_three_update_scenarios_consistent(updates, policy_sources):
    assert len(updates) == 3, f"Expected exactly 3 updates, found {len(updates)}"
    update_doc_ids = {u["document_id"] for u in updates}
    assert update_doc_ids == {"LND-01", "FIN-01", "HR-01"}

    for u in updates:
        assert u["previous_version"] == "1.0"
        assert u["new_version"] == "2.0"
        assert u["new_effective_date"] == "2027-01-15", f"{u['document_id']} update date must be 2027-01-15"
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


# 14. Question permissions, fact consistency and valid fields
def test_14_question_permissions_and_fact_consistency(questions, policy_sources, fact_ledger):
    doc_roles = {p["document_id"]: set(p["allowed_roles"]) for p in policy_sources}
    all_doc_ids = set(doc_roles.keys())
    fact_map = {f["fact_id"]: f for f in fact_ledger}
    valid_behaviors = {"answer", "refuse_insufficient_evidence", "deny_unauthorized"}

    v2_doc_map = {
        "v2_lnd_active": "LND-01",
        "v2_fin_active": "FIN-01",
        "v2_hr_active": "HR-01",
    }

    for q in questions:
        qid = q["question_id"]
        role = q["role"]
        behavior = q["expected_behavior"]

        assert role in VALID_ROLES, f"{qid} has invalid role: {role}"
        assert behavior in valid_behaviors, f"{qid} has invalid behavior: {behavior}"

        for doc_id in q["gold_document_ids"]:
            assert doc_id in all_doc_ids, f"{qid} refers to unknown doc ID: {doc_id}"

        for fid in q["gold_fact_ids"]:
            assert fid in fact_map, f"{qid} refers to unknown fact ID: {fid}"

        if behavior == "answer":
            # Role must have access to every required supporting document
            for doc_id in q["gold_document_ids"]:
                assert role in doc_roles[doc_id], f"{qid}: Role {role} cannot access {doc_id}"

            # Supporting facts must align with referenced docs, sections, and version scopes
            for fid in q["gold_fact_ids"]:
                fact = fact_map[fid]
                assert fact["document_id"] in q["gold_document_ids"], (
                    f"{qid}: Fact {fid} doc {fact['document_id']} not in gold docs {q['gold_document_ids']}"
                )
                assert fact["section"] in q["gold_sections"], (
                    f"{qid}: Fact {fid} section {fact['section']} not in gold sections {q['gold_sections']}"
                )

                scope = fact.get("version_scope", "all")
                corpus_state = q["corpus_state"]
                if corpus_state == "v1_active":
                    assert scope in ("all", "1.0"), f"{qid}: Fact {fid} scope {scope} incompatible with v1_active"
                elif corpus_state in v2_doc_map:
                    updated_doc = v2_doc_map[corpus_state]
                    if fact["document_id"] == updated_doc:
                        assert scope in ("all", "2.0"), (
                            f"{qid}: Fact {fid} scope {scope} incompatible with updated {updated_doc}"
                        )
                    else:
                        assert scope in ("all", "1.0"), (
                            f"{qid}: Fact {fid} scope {scope} incompatible with non-updated doc in {corpus_state}"
                        )

        elif behavior == "deny_unauthorized":
            assert len(q["gold_document_ids"]) > 0, f"Unauthorized question {qid} must reference restricted docs"
            assert any(role not in doc_roles[d] for d in q["gold_document_ids"]), (
                f"Unauthorized question {qid} has role {role} which is permitted for all docs {q['gold_document_ids']}"
            )

        elif behavior == "refuse_insufficient_evidence":
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


# 16. Cache scenario correctness and injection eligibility
def test_16_cache_and_injection_validation(cache_scenarios, injection_cases, policy_sources):
    doc_roles = {p["document_id"]: set(p["allowed_roles"]) for p in policy_sources}
    all_doc_ids = set(doc_roles.keys())
    policy_sections = {
        p["document_id"]: {s["section_number"] for s in p["sections"]}
        for p in policy_sources
    }

    # Validate cache scenarios
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

    for s in cache_scenarios:
        reqs = s["requests"]
        assert len(reqs) == 2, f"Scenario {s['scenario_id']} must have exactly 2 requests"

        # Step 1 must always warm the cache without intervening events
        assert reqs[0]["step"] == 1
        assert reqs[0]["intervening_event"] is None
        assert reqs[0]["expected_decision"] == "MISS"
        assert reqs[0]["role"] in VALID_ROLES

        # Validate structured intervening events in Step 2
        ev = reqs[1]["intervening_event"]
        if s["category"] == "permission_revocation":
            # Role remains unchanged between requests
            assert reqs[0]["role"] == reqs[1]["role"], (
                f"{s['scenario_id']} revocation test must retain the same role across requests"
            )
            assert isinstance(ev, dict) and ev.get("type") == "permission_revocation"
            assert ev.get("document_id") in all_doc_ids
            assert ev.get("role") in VALID_ROLES
            assert ev.get("role") == reqs[1]["role"]
            assert reqs[1]["expected_decision"] == "REJECT_ROLE"

        elif s["category"] == "document_update":
            assert isinstance(ev, dict) and ev.get("type") == "document_update"
            assert ev.get("document_id") in {"LND-01", "FIN-01", "HR-01"}
            assert ev.get("new_version") == "2.0"
            assert reqs[1]["expected_decision"] == "REJECT_VERSION"

        else:
            assert ev is None, f"{s['scenario_id']} of category {s['category']} should not have intervening event"

    # Validate injection cases
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

    for inj in injection_cases:
        cid = inj["case_id"]
        doc_id = inj["target_document_id"]
        sec = inj["target_section"]
        role = inj["employee_role"]

        assert doc_id in all_doc_ids, f"{cid} targets unknown doc ID {doc_id}"
        assert sec in policy_sections[doc_id], f"{cid} targets unknown section {sec} in {doc_id}"
        # Every injection case must use an authorized role for its target document
        assert role in doc_roles[doc_id], (
            f"{cid}: Role {role} must be authorized for target document {doc_id} to test injection"
        )

        if inj["category"] == "benign_control":
            assert inj["is_malicious"] is False, f"{cid} benign control must have is_malicious=False"
        else:
            assert inj["is_malicious"] is True, f"{cid} attack case must have is_malicious=True"
