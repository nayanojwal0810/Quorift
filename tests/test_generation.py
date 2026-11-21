"""Focused tests for grounded answer prompt construction, validation, citation verification, and input preparation."""

import json
from pathlib import Path
import pytest

from quorift.config import paths
from quorift.generation import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    VALID_ROLES,
    build_grounded_prompt,
    generate_empty_evidence_response,
    parse_generation_response,
    resolve_citation_metadata,
    validate_generation_output,
    validate_generation_request,
)


@pytest.fixture
def sample_registry():
    """Trusted document registry fixture."""
    return {
        "LND-01": {
            "document_id": "LND-01",
            "active_version": "1.0",
            "source_content_hash": "hash_lnd_01",
            "active_status": True,
            "allowed_roles": ["lending", "admin"],
        },
        "GEN-02": {
            "document_id": "GEN-02",
            "active_version": "1.0",
            "source_content_hash": "hash_gen_02",
            "active_status": True,
            "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        },
        "FIN-01": {
            "document_id": "FIN-01",
            "active_version": "1.0",
            "source_content_hash": "hash_fin_01",
            "active_status": True,
            "allowed_roles": ["finance", "admin"],
        },
    }


@pytest.fixture
def sample_chunk_lnd01():
    """Valid authorized chunk fixture for LND-01."""
    return {
        "chunk_id": "LND-01|1.0|2.0",
        "document_id": "LND-01",
        "title": "Personal Loan Approval Authority",
        "section_number": "2.0",
        "section_heading": "Approval Limits",
        "version": "1.0",
        "source_hash": "hash_lnd_01",
        "page_start": 1,
        "page_end": 1,
        "source_pdf": "data/documents/lnd_01.pdf",
        "allowed_roles": ["lending", "admin"],
        "text": "Branch Manager may approve personal loans up to INR 10 lakh.",
    }


@pytest.fixture
def sample_chunk_gen02():
    """Valid authorized chunk fixture for GEN-02."""
    return {
        "chunk_id": "GEN-02|1.0|2.0",
        "document_id": "GEN-02",
        "title": "Travel and Expense Policy",
        "section_number": "2.0",
        "section_heading": "Lodging Ceilings",
        "version": "1.0",
        "source_hash": "hash_gen_02",
        "page_start": 2,
        "page_end": 3,
        "source_pdf": "data/documents/gen_02.pdf",
        "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        "text": "Overnight lodging in Tier 1 Metros is capped at INR 6,500 per night.",
    }


# 1. Correct prompt construction
def test_correct_prompt_construction(sample_chunk_gen02):
    """build_grounded_prompt constructs expected chat messages and includes system prompt."""
    question = "What is the lodging ceiling in Tier 1 cities?"
    messages = build_grounded_prompt(question, [sample_chunk_gen02])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "What is the lodging ceiling in Tier 1 cities?" in messages[1]["content"]
    assert "--- EXCERPT START ---" in messages[1]["content"]
    assert "--- EXCERPT END ---" in messages[1]["content"]


# 2. Evidence includes valid source IDs
def test_evidence_includes_valid_source_ids(sample_chunk_gen02):
    """Prompt user message formats source chunk ID, version, section, and page range."""
    messages = build_grounded_prompt("Test question", [sample_chunk_gen02])
    content = messages[1]["content"]

    assert "Chunk ID: GEN-02|1.0|2.0" in content
    assert "Document: Travel and Expense Policy (Version 1.0)" in content
    assert "Section: 2.0 - Lodging Ceilings" in content
    assert "Pages: Pages 2-3" in content
    assert "Overnight lodging in Tier 1 Metros is capped at INR 6,500 per night." in content


# 3. Gold answers are not included in the prompt
def test_gold_answers_not_included_in_prompt(sample_chunk_gen02):
    """Ensures prompts do not contain hidden gold answers, fact IDs, or evaluation labels."""
    messages = build_grounded_prompt("What is the lodging limit?", [sample_chunk_gen02])
    full_prompt = messages[0]["content"] + " " + messages[1]["content"]

    assert "expected_answer" not in full_prompt
    assert "gold_fact_ids" not in full_prompt
    assert "gold_evidence_pairs" not in full_prompt
    assert "expected_behavior" not in full_prompt


# 4. Missing registry permissions fail closed
def test_missing_registry_permissions_fail_closed(sample_registry, sample_chunk_lnd01):
    """validate_generation_request fails closed if registry allowed_roles is missing or empty."""
    del sample_registry["LND-01"]["allowed_roles"]
    req = {
        "question": "Can Branch Manager approve loans?",
        "role": "lending",
        "retrieved_chunks": [sample_chunk_lnd01],
    }
    with pytest.raises(ValueError, match="missing valid allowed_roles list"):
        validate_generation_request(req, sample_registry)


# 5. Unauthorized evidence is rejected
def test_unauthorized_evidence_is_rejected(sample_registry, sample_chunk_lnd01):
    """validate_generation_request rejects request when user role cannot access chunk document."""
    # General user attempting to use lending-restricted chunk LND-01
    req = {
        "question": "Can Branch Manager approve loans?",
        "role": "general",
        "retrieved_chunks": [sample_chunk_lnd01],
    }
    with pytest.raises(ValueError, match="Unauthorized evidence: role 'general' cannot access"):
        validate_generation_request(req, sample_registry)


# 6. Stale document versions are rejected
def test_stale_document_versions_rejected(sample_registry, sample_chunk_lnd01):
    """validate_generation_request rejects chunks with version mismatch against registry."""
    stale_chunk = dict(sample_chunk_lnd01)
    stale_chunk["version"] = "2.0"
    req = {
        "question": "What is the limit?",
        "role": "lending",
        "retrieved_chunks": [stale_chunk],
    }
    with pytest.raises(ValueError, match="version '2.0' != registry active version '1.0'"):
        validate_generation_request(req, sample_registry)


# 7. Fabricated citation IDs are rejected
def test_fabricated_citation_ids_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output rejects citations not present in the supplied evidence."""
    parsed = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": ["GEN-99|1.0|9.0"],  # Fabricated ID
    }
    with pytest.raises(ValueError, match="Fabricated citation ID 'GEN-99\\|1.0\\|9.0'"):
        validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")


# 8. Duplicate citations are rejected
def test_duplicate_citations_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output rejects duplicate citations in model response."""
    parsed = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": ["GEN-02|1.0|2.0", "GEN-02|1.0|2.0"],
    }
    with pytest.raises(ValueError, match="Duplicate citation IDs found"):
        validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")


# 9. Missing citations on an answered response are rejected
def test_missing_citations_on_answered_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output rejects status answered with empty citations list."""
    parsed = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": [],
    }
    with pytest.raises(ValueError, match="Status 'answered' requires at least one valid citation"):
        validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")


# 10. Valid refusal with empty citations is accepted
def test_valid_refusal_accepted(sample_registry, sample_chunk_gen02):
    """validate_generation_output accepts valid insufficient_evidence refusal with exact REFUSAL_MESSAGE."""
    parsed = {
        "status": "insufficient_evidence",
        "answer": f"  {REFUSAL_MESSAGE}  ",  # Surrounding whitespace stripped
        "citations": [],
    }
    validated = validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")
    assert validated["status"] == "insufficient_evidence"
    assert validated["citations"] == []
    assert validated["answer"] == REFUSAL_MESSAGE


# 10b. Refusal with arbitrary or fabricated text is rejected
def test_refusal_with_fabricated_or_unsupported_text_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output rejects any refusal that differs from REFUSAL_MESSAGE."""
    parsed = {
        "status": "insufficient_evidence",
        "answer": "I cannot verify this, but personal loans generally require INR 10 lakh.",
        "citations": [],
    }
    with pytest.raises(ValueError, match="Refusal answer must exactly match the approved refusal message"):
        validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")


# 11. Refusal with citations is rejected
def test_refusal_with_citations_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output rejects status insufficient_evidence when citations are provided."""
    parsed = {
        "status": "insufficient_evidence",
        "answer": REFUSAL_MESSAGE,
        "citations": ["GEN-02|1.0|2.0"],
    }
    with pytest.raises(ValueError, match="Refusal status 'insufficient_evidence' must have empty citations"):
        validate_generation_output(parsed, [sample_chunk_gen02], sample_registry, role="general")


# 11b. Malformed citation types produce clear ValueError
def test_malformed_citation_types_rejected(sample_registry, sample_chunk_gen02):
    """validate_generation_output raises clear ValueError when citations contain non-string types or not a list."""
    # Citations not a list
    parsed_not_list = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": "GEN-02|1.0|2.0",
    }
    with pytest.raises(ValueError, match="Citations must be a list"):
        validate_generation_output(parsed_not_list, [sample_chunk_gen02], sample_registry, role="general")

    # Citation item is integer
    parsed_int_item = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": [123],
    }
    with pytest.raises(ValueError, match="Citation ID must be a string"):
        validate_generation_output(parsed_int_item, [sample_chunk_gen02], sample_registry, role="general")

    # Citation item is dictionary
    parsed_dict_item = {
        "status": "answered",
        "answer": "The lodging cap is INR 6,500.",
        "citations": [{"id": "GEN-02|1.0|2.0"}],
    }
    with pytest.raises(ValueError, match="Citation ID must be a string"):
        validate_generation_output(parsed_dict_item, [sample_chunk_gen02], sample_registry, role="general")


# 12. Malformed JSON is rejected
def test_malformed_json_rejected():
    """parse_generation_response raises clear ValueError for malformed JSON strings."""
    with pytest.raises(ValueError, match="Malformed JSON in model output"):
        parse_generation_response("This is plain text without JSON.")

    with pytest.raises(ValueError, match="Malformed JSON in model output"):
        parse_generation_response("{status: 'answered'}")  # Invalid unquoted JSON


# 13. Empty evidence produces a refusal without model execution
def test_empty_evidence_produces_refusal():
    """generate_empty_evidence_response returns standard insufficient_evidence refusal."""
    resp = generate_empty_evidence_response()
    assert resp["status"] == "insufficient_evidence"
    assert resp["citations"] == []
    assert resp["answer"] == REFUSAL_MESSAGE


# 14. Correct source metadata is resolved from trusted records
def test_correct_source_metadata_resolved(sample_chunk_gen02):
    """resolve_citation_metadata resolves human-readable document, section, and page information."""
    meta = resolve_citation_metadata("GEN-02|1.0|2.0", [sample_chunk_gen02])
    assert meta["chunk_id"] == "GEN-02|1.0|2.0"
    assert meta["document_id"] == "GEN-02"
    assert meta["title"] == "Travel and Expense Policy"
    assert meta["version"] == "1.0"
    assert meta["section_number"] == "2.0"
    assert meta["page_start"] == 2
    assert meta["page_end"] == 3
    assert "GEN-02 — Travel and Expense Policy" in meta["display_citation"]
    assert "Pages 2-3" in meta["display_citation"]
