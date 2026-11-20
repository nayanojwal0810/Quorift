"""Tests for HybridRetriever and baseline hybrid retrieval (Dense + BM25 + RRF)."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from quorift.indexing import (
    build_bm25_index,
    build_manifest,
    build_qdrant_index,
    save_manifest,
)
from quorift.ingestion import save_chunks_jsonl
from quorift.retrieval import HybridRetriever, RRF_CONSTANT, retrieve


@pytest.fixture
def mock_embedding_model():
    """Lightweight dummy model returning fixed normalized 768D embeddings."""
    mock = MagicMock()

    def dummy_encode(texts, **kwargs):
        arr = np.zeros((len(texts), 768), dtype=np.float32)
        arr[:, 0] = 1.0
        return arr

    mock.encode = MagicMock(side_effect=dummy_encode)
    return mock


@pytest.fixture
def retrieval_test_fixture(tmp_path, mock_embedding_model):
    """Sets up a complete lightweight indexes directory for testing retrieval."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    qdrant_dir = indexes_dir / "qdrant"
    bm25_dir = indexes_dir / "bm25"
    chunks_path = indexes_dir / "chunks.jsonl"
    manifest_path = indexes_dir / "index_manifest.json"

    # Corpus: 5 chunks with distinct permissions, version, and content
    chunks = [
        {
            "chunk_id": "LND-01|1.0|2.0",
            "document_id": "LND-01",
            "title": "Personal Loan Policy",
            "department": "Retail Lending",
            "section_number": "2.0",
            "section_heading": "Loan Approval Limits",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["lending", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/lnd_01.pdf",
            "source_hash": "hash_lnd_01",
            "text": "LND-01 | Personal Loan Policy\nSection 2.0: Loan Approval Limits\n\nBranch Manager may approve personal loans up to INR 10 lakh.",
        },
        {
            "chunk_id": "LND-01|1.0|3.0",
            "document_id": "LND-01",
            "title": "Personal Loan Policy",
            "department": "Retail Lending",
            "section_number": "3.0",
            "section_heading": "Loan Interest Rates",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 2,
            "page_end": 2,
            "allowed_roles": ["lending", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/lnd_01.pdf",
            "source_hash": "hash_lnd_01",
            "text": "LND-01 | Personal Loan Policy\nSection 3.0: Loan Interest Rates\n\nPersonal loan interest rates start from 10.5 percent per annum.",
        },
        {
            "chunk_id": "CMP-01|1.0|2.0",
            "document_id": "CMP-01",
            "title": "Customer KYC Policy",
            "department": "Compliance",
            "section_number": "2.0",
            "section_heading": "KYC Verification",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["compliance", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/cmp_01.pdf",
            "source_hash": "hash_cmp_01",
            "text": "CMP-01 | Customer KYC Policy\nSection 2.0: KYC Verification\n\nKYC verification requires passport, voter card, or driving license.",
        },
        {
            "chunk_id": "HR-01|1.0|2.0",
            "document_id": "HR-01",
            "title": "Leave Policy",
            "department": "Human Resources",
            "section_number": "2.0",
            "section_heading": "Annual Leave",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
            "classification": "General Internal",
            "source_pdf": "data/documents/hr_01.pdf",
            "source_hash": "hash_hr_01",
            "text": "HR-01 | Leave Policy\nSection 2.0: Annual Leave\n\nAll employees receive 18 days of annual leave.",
        },
        {
            "chunk_id": "LND-01|2.0|2.0",
            "document_id": "LND-01",
            "title": "Personal Loan Policy v2",
            "department": "Retail Lending",
            "section_number": "2.0",
            "section_heading": "Future Limits",
            "version": "2.0",  # Inactive version
            "effective_date": "2026-07-01",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["lending", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/lnd_01_v2.pdf",
            "source_hash": "hash_lnd_01_v2",
            "text": "LND-01 | Personal Loan Policy v2\nSection 2.0: Future Limits\n\nFuture personal loan limits will increase to INR 15 lakh.",
        },
    ]

    # Save chunks.jsonl
    save_chunks_jsonl(chunks, chunks_path)

    # Deterministic 768D embeddings
    raw_vectors = np.zeros((len(chunks), 768), dtype=np.float32)
    for i in range(len(chunks)):
        raw_vectors[i, i] = 1.0

    # Build Qdrant index
    q_client = build_qdrant_index(
        chunks=chunks,
        embeddings=raw_vectors,
        storage_path=qdrant_dir,
        collection_name="quorift_chunks",
    )
    q_client.close()

    # Build BM25 index
    build_bm25_index(chunks=chunks, storage_path=bm25_dir)

    # Build manifest
    manifest = build_manifest(
        chunks=chunks,
        model_name="mock-model",
        embedding_dim=768,
        collection_name="quorift_chunks",
        bm25_dir=str(bm25_dir),
    )
    save_manifest(manifest, manifest_path)

    retriever = HybridRetriever(
        indexes_dir=indexes_dir,
        model=mock_embedding_model,
        collection_name="quorift_chunks",
    )

    yield retriever
    retriever.close()


def test_invalid_role_rejection(retrieval_test_fixture):
    """Retriever must reject invalid roles."""
    with pytest.raises(ValueError, match="Invalid role"):
        retrieval_test_fixture.retrieve("What is the loan limit?", role="hacker")

    with pytest.raises(ValueError, match="Invalid role"):
        retrieval_test_fixture.dense_search("What is the loan limit?", role="guest")

    with pytest.raises(ValueError, match="Invalid role"):
        retrieval_test_fixture.bm25_search("What is the loan limit?", role="")


def test_empty_query_rejection(retrieval_test_fixture):
    """Retriever must reject empty queries."""
    with pytest.raises(ValueError, match="Question must not be empty"):
        retrieval_test_fixture.retrieve("", role="general")

    with pytest.raises(ValueError, match="Question must not be empty"):
        retrieval_test_fixture.retrieve("   ", role="general")


def test_role_filtering_excludes_disallowed_chunks(retrieval_test_fixture):
    """Lending role should see lending and general internal docs, but never compliance."""
    results = retrieval_test_fixture.retrieve("KYC passport voter", role="lending", top_k=5)
    result_doc_ids = [r["document_id"] for r in results]

    assert "CMP-01" not in result_doc_ids
    for r in results:
        assert "lending" in r["allowed_roles"]


def test_active_version_filtering(retrieval_test_fixture):
    """Inactive version 2.0 chunk must never appear in retrieved results."""
    results = retrieval_test_fixture.retrieve("Future personal loan limits", role="lending", top_k=5)
    for r in results:
        assert r["version"] == "1.0"
        assert r["chunk_id"] != "LND-01|2.0|2.0"


def test_duplicate_chunk_merging_and_rrf_calculation(retrieval_test_fixture):
    """When a chunk is returned by both dense and BM25, it must be merged with combined RRF score."""
    # Query matching LND-01|1.0|2.0 strongly in both dense (vector direction) and BM25 (exact keywords)
    results = retrieval_test_fixture.retrieve("Branch Manager loan approval limits", role="lending", top_k=5)
    assert len(results) > 0

    top_chunk = results[0]
    assert top_chunk["chunk_id"] == "LND-01|1.0|2.0"

    # Verify both dense and bm25 ranks are populated
    assert top_chunk["dense_rank"] is not None
    assert top_chunk["bm25_rank"] is not None

    expected_score = (1.0 / (RRF_CONSTANT + top_chunk["dense_rank"])) + (
        1.0 / (RRF_CONSTANT + top_chunk["bm25_rank"])
    )
    assert pytest.approx(top_chunk["rrf_score"], rel=1e-5) == expected_score


def test_bm25_permission_filtering_before_truncation(retrieval_test_fixture):
    """BM25 search must filter permissions across entire corpus so allowed docs are not starved."""
    # Query with compliance keywords under role 'compliance'
    hits = retrieval_test_fixture.bm25_search("passport voter driving license", role="compliance", candidate_k=5)
    doc_ids = [h["document_id"] for h in hits]
    assert "CMP-01" in doc_ids


def test_metadata_preservation(retrieval_test_fixture):
    """Retrieved results must preserve exact metadata fields."""
    results = retrieval_test_fixture.retrieve("loan approval limits", role="lending", top_k=1)
    assert len(results) == 1
    hit = results[0]

    assert hit["document_id"] == "LND-01"
    assert hit["title"] == "Personal Loan Policy"
    assert hit["section_number"] in ("2.0", "3.0")
    assert hit["version"] == "1.0"
    assert hit["page_start"] in (1, 2)
    assert hit["page_end"] in (1, 2)
    assert hit["source_pdf"] == "data/documents/lnd_01.pdf"
    assert "lending" in hit["allowed_roles"]
    assert "text" in hit
    assert hit["rank"] == 1


def test_convenience_retrieve_function(tmp_path, retrieval_test_fixture):
    """The module-level retrieve() function works correctly with an existing retriever."""
    results = retrieve(
        question="annual leave entitlement",
        role="general",
        top_k=2,
        retriever=retrieval_test_fixture,
    )
    assert len(results) >= 1
    assert results[0]["document_id"] == "HR-01"
