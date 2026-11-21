"""Validation tests for the Quorift banking policy documents, metadata, and registry."""

import json
from pathlib import Path
import pymupdf
import pytest

from quorift.config import paths
from scripts.build_documents import compute_source_hash

VALID_ROLES = {"general", "lending", "finance", "compliance", "hr", "admin"}


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
def updates():
    updates_path = paths.metadata_dir / "updates.json"
    assert updates_path.exists(), "updates.json does not exist"
    with open(updates_path, encoding="utf-8") as f:
        return json.load(f)


# 1. Exactly 20 unique active policy IDs exist
def test_01_exactly_20_unique_active_policy_ids(policy_sources):
    assert len(policy_sources) == 20, f"Expected 20 policies, found {len(policy_sources)}"
    doc_ids = [p["document_id"] for p in policy_sources]
    assert len(doc_ids) == len(set(doc_ids)), "Policy IDs must be unique"

    expected_prefixes = {"GEN-", "LND-", "FIN-", "CMP-", "HR-"}
    actual_prefixes = {pid.split("-")[0] + "-" for pid in doc_ids}
    assert actual_prefixes == expected_prefixes, f"Mismatch in expected prefixes: {actual_prefixes}"

    prefix_counts = {}
    for pid in doc_ids:
        prefix = pid.split("-")[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    assert prefix_counts == {"GEN": 3, "LND": 5, "FIN": 4, "CMP": 4, "HR": 4}


# 2. Exactly 20 policy PDFs exist in the documents directory
def test_02_exactly_20_policy_pdfs_exist(policy_sources):
    pdf_files = list(paths.documents_dir.glob("*.pdf"))
    assert len(pdf_files) == 20, f"Expected 20 PDFs in {paths.documents_dir}, found {len(pdf_files)}"

    expected_filenames = {
        f"{p['document_id'].lower().replace('-', '_')}.pdf"
        for p in policy_sources
    }
    actual_filenames = {f.name for f in pdf_files}
    assert actual_filenames == expected_filenames, f"PDF filenames mismatch: {actual_filenames ^ expected_filenames}"


# 3. Every PDF has extractable text and is readable
def test_03_every_pdf_has_extractable_text():
    pdf_files = list(paths.documents_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        doc = pymupdf.open(pdf_path)
        assert len(doc) >= 1, f"{pdf_path.name} has no pages"
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        assert len(full_text.strip()) > 100, f"{pdf_path.name} has insufficient extractable text ({len(full_text)} chars)"
        doc.close()


# 4. Every document has required metadata fields
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


# 5. Every document contains expected headings and sections
def test_05_every_document_contains_expected_headings_and_sections(policy_sources):
    for p in policy_sources:
        doc_id = p["document_id"]
        sections = p["sections"]
        assert len(sections) >= 3, f"{doc_id} must have at least 3 sections, found {len(sections)}"

        section_numbers = [s["section_number"] for s in sections]
        assert len(section_numbers) == len(set(section_numbers)), f"{doc_id} has duplicate section numbers"

        total_words = 0
        for s in sections:
            assert "heading" in s, f"{doc_id} section missing heading"
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


# 10. All three update scenarios are consistent with their initial documents and future dates
def test_10_three_update_scenarios_consistent(updates, policy_sources):
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

    policy_content_map = {}
    for p in policy_sources:
        for s in p["sections"]:
            policy_content_map[(p["document_id"], s["section_number"])] = s["content"]

    assert "INR 10 lakh" in policy_content_map[("LND-01", "2.0")]
    assert "INR 5,000" in policy_content_map[("FIN-01", "2.0")]
    assert "18 working days" in policy_content_map[("HR-01", "2.0")]


# 11. Role permission coverage and registry alignment
def test_11_role_permission_coverage(policy_sources, document_registry):
    doc_roles = {p["document_id"]: set(p["allowed_roles"]) for p in policy_sources}
    reg_roles = {r["document_id"]: set(r["allowed_roles"]) for r in document_registry}

    assert doc_roles == reg_roles, "Source policies and document registry allowed_roles must match"

    for role in VALID_ROLES:
        accessible_docs = [doc_id for doc_id, roles in doc_roles.items() if role in roles]
        assert len(accessible_docs) > 0, f"Role {role} has no accessible documents"
        if role == "admin":
            assert len(accessible_docs) == 20, "Admin must have access to all 20 policies"
        elif role == "general":
            assert all(doc.startswith("GEN-") or doc == "HR-01" for doc in accessible_docs)
