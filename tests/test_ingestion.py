"""Tests for PDF ingestion, section parsing, and chunk generation."""

import json
from pathlib import Path
import pytest
import pymupdf

from quorift.config import paths
from quorift.ingestion import (
    build_chunks,
    extract_sections_from_pdf,
    load_chunks_jsonl,
    save_chunks_jsonl,
)


@pytest.fixture(scope="module")
def document_registry():
    with open(paths.metadata_dir / "document_registry.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def policy_sources():
    with open(paths.metadata_dir / "policy_sources.json", encoding="utf-8") as f:
        return {p["document_id"]: p for p in json.load(f)}


@pytest.fixture(scope="module")
def chunks():
    return build_chunks()


# 1. All 20 PDFs can be processed
def test_all_20_pdfs_can_be_processed(document_registry):
    assert len(document_registry) == 20
    for reg in document_registry:
        pdf_path = paths.project_root / reg["relative_pdf_path"]
        assert pdf_path.exists(), f"Missing PDF: {pdf_path}"
        sections = extract_sections_from_pdf(pdf_path, doc_id=reg["document_id"])
        assert len(sections) >= 4, f"{reg['document_id']} has fewer than 4 sections"


# 2. Every expected policy section is extracted
def test_every_expected_section_extracted(chunks, policy_sources):
    assert len(chunks) == 120, f"Expected 120 chunks, got {len(chunks)}"
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c["document_id"], []).append(c)

    assert len(chunks_by_doc) == 20
    for doc_id, doc_chunks in chunks_by_doc.items():
        expected_sections = policy_sources[doc_id]["sections"]
        assert len(doc_chunks) == len(expected_sections), f"Mismatch in chunk count for {doc_id}"
        actual_sec_nums = [c["section_number"] for c in doc_chunks]
        expected_sec_nums = [s["section_number"] for s in expected_sections]
        assert actual_sec_nums == expected_sec_nums


# 3. No chunk has empty text
def test_no_chunk_has_empty_text(chunks):
    for c in chunks:
        assert c["text"].strip(), f"Chunk {c['chunk_id']} has empty text"
        # Must have header line and section line
        lines = [line.strip() for line in c["text"].split("\n") if line.strip()]
        assert len(lines) >= 3, f"Chunk {c['chunk_id']} text is improperly structured"


# 4. There are no duplicate chunk IDs
def test_no_duplicate_chunk_ids(chunks):
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"


# 5. Document and version metadata are correct
def test_document_and_version_metadata(chunks, document_registry):
    reg_map = {r["document_id"]: r for r in document_registry}
    for c in chunks:
        doc_id = c["document_id"]
        reg = reg_map[doc_id]
        assert c["version"] == reg["active_version"]
        assert c["title"] == reg["title"]
        assert c["source_hash"] == reg["source_content_hash"]
        expected_id = f"{doc_id}|{reg['active_version']}|{c['section_number']}"
        assert c["chunk_id"] == expected_id


# 6. Page references are valid
def test_page_references_valid(chunks):
    doc_cache: dict[str, int] = {}
    for c in chunks:
        pdf_rel = c["source_pdf"]
        if pdf_rel not in doc_cache:
            pdf_path = paths.project_root / pdf_rel
            doc = pymupdf.open(pdf_path)
            doc_cache[pdf_rel] = len(doc)

        total_pages = doc_cache[pdf_rel]
        assert 1 <= c["page_start"] <= c["page_end"] <= total_pages, (
            f"Invalid page reference in {c['chunk_id']}: {c['page_start']}..{c['page_end']} (total {total_pages})"
        )


# 7. Document permissions match the trusted registry
def test_document_permissions_match_registry(chunks, document_registry):
    reg_map = {r["document_id"]: r for r in document_registry}
    for c in chunks:
        reg = reg_map[c["document_id"]]
        assert set(c["allowed_roles"]) == set(reg["allowed_roles"])


# 8. Extracted text contains key policy facts and excludes disclaimer and footers
def test_extracted_text_contains_key_facts_and_excludes_chrome(chunks):
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Fact 1: LND-01 Section 2.0 contains INR 10 lakh Branch Manager limit
    lnd_chunk = chunk_map["LND-01|1.0|2.0"]
    assert "INR 10 lakh" in lnd_chunk["text"]

    # Fact 2: FIN-01 Section 2.0 contains INR 5,000 Team Lead limit
    fin_chunk = chunk_map["FIN-01|1.0|2.0"]
    assert "INR 5,000" in fin_chunk["text"]

    # Fact 3: HR-01 Section 2.0 contains 18 working days annual leave
    hr_chunk = chunk_map["HR-01|1.0|2.0"]
    assert "18 working days" in hr_chunk["text"]

    # Confirm disclaimers and running headers/footers are excluded from all chunk texts
    for c in chunks:
        assert "DEMONSTRATION DOCUMENT NOTICE" not in c["text"]
        assert "CONFIDENTIAL & PROPRIETARY — APEX CREST BANK" not in c["text"]
        assert "Page 1 of" not in c["text"]
        assert "Page 2 of" not in c["text"]


# 9. JSONL save and load round-trip
def test_jsonl_round_trip(chunks, tmp_path):
    out_file = tmp_path / "chunks.jsonl"
    save_chunks_jsonl(chunks, out_file)
    assert out_file.exists()

    reloaded = load_chunks_jsonl(out_file)
    assert len(reloaded) == len(chunks)
    assert [c["chunk_id"] for c in reloaded] == [c["chunk_id"] for c in chunks]
