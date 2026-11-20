"""Tests for HybridRetriever and baseline hybrid retrieval (Dense + BM25 + RRF)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from quorift.indexing import (
    DEFAULT_COLLECTION_NAME,
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
def sample_test_corpus():
    """Returns 4 chunks across 3 active documents."""
    return [
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
    ]


@pytest.fixture
def retrieval_test_fixture(tmp_path, mock_embedding_model, sample_test_corpus):
    """Sets up a complete lightweight indexes directory and registry for testing retrieval."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    qdrant_dir = indexes_dir / "qdrant"
    bm25_dir = indexes_dir / "bm25"
    chunks_path = indexes_dir / "chunks.jsonl"
    manifest_path = indexes_dir / "index_manifest.json"
    registry_path = tmp_path / "document_registry.json"

    chunks = sample_test_corpus

    # Create matching trusted registry
    dummy_registry = [
        {
            "document_id": "LND-01",
            "active_status": True,
            "active_version": "1.0",
            "source_content_hash": "hash_lnd_01",
            "allowed_roles": ["lending", "admin"],
        },
        {
            "document_id": "CMP-01",
            "active_status": True,
            "active_version": "1.0",
            "source_content_hash": "hash_cmp_01",
            "allowed_roles": ["compliance", "admin"],
        },
        {
            "document_id": "HR-01",
            "active_status": True,
            "active_version": "1.0",
            "source_content_hash": "hash_hr_01",
            "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
        },
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

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
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    q_client.close()

    # Build BM25 index
    build_bm25_index(chunks=chunks, storage_path=bm25_dir)

    # Build manifest
    manifest = build_manifest(
        chunks=chunks,
        model_name="mock-model",
        embedding_dim=768,
        collection_name=DEFAULT_COLLECTION_NAME,
        bm25_dir=str(bm25_dir),
    )
    save_manifest(manifest, manifest_path)

    retriever = HybridRetriever(
        indexes_dir=indexes_dir,
        registry_path=registry_path,
        model=mock_embedding_model,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    yield retriever
    retriever.close()


def test_retriever_successful_startup(retrieval_test_fixture):
    """Retriever starts cleanly and initializes active registry."""
    assert len(retrieval_test_fixture.active_registry) == 3
    assert "LND-01" in retrieval_test_fixture.active_registry
    assert retrieval_test_fixture.active_registry["LND-01"]["active_version"] == "1.0"


def test_retriever_stale_manifest_rejected(tmp_path, mock_embedding_model, sample_test_corpus):
    """A manifest whose version disagrees with the trusted registry must be rejected."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    dummy_registry = [
        {"document_id": "LND-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_lnd_01", "allowed_roles": ["lending", "admin"]},
        {"document_id": "CMP-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_cmp_01", "allowed_roles": ["compliance", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_test_corpus, indexes_dir / "chunks.jsonl")
    build_bm25_index(sample_test_corpus, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(sample_test_corpus), 768), dtype=np.float32)
    q_client = build_qdrant_index(sample_test_corpus, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()

    # Manifest contains stale active_version 2.0 for LND-01
    manifest = build_manifest(sample_test_corpus, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    for d in manifest["active_documents"]:
        if d["document_id"] == "LND-01":
            d["active_version"] = "2.0"
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    with pytest.raises(ValueError, match="Active version mismatch"):
        HybridRetriever(
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            model=mock_embedding_model,
        )


def test_retriever_changed_source_hash_rejected(tmp_path, mock_embedding_model, sample_test_corpus):
    """A manifest whose source hash disagrees with the registry must be rejected."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    # Registry has an updated source hash
    dummy_registry = [
        {"document_id": "LND-01", "active_status": True, "active_version": "1.0", "source_content_hash": "new_hash_lnd_01", "allowed_roles": ["lending", "admin"]},
        {"document_id": "CMP-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_cmp_01", "allowed_roles": ["compliance", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_test_corpus, indexes_dir / "chunks.jsonl")
    build_bm25_index(sample_test_corpus, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(sample_test_corpus), 768), dtype=np.float32)
    q_client = build_qdrant_index(sample_test_corpus, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()

    manifest = build_manifest(sample_test_corpus, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    with pytest.raises(ValueError, match="Source hash mismatch"):
        HybridRetriever(
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            model=mock_embedding_model,
        )


def test_retriever_incorrect_active_version_rejected(tmp_path, mock_embedding_model, sample_test_corpus):
    """An index containing a chunk version mismatch against the registry must be rejected."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    dummy_registry = [
        {"document_id": "LND-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_lnd_01", "allowed_roles": ["lending", "admin"]},
        {"document_id": "CMP-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_cmp_01", "allowed_roles": ["compliance", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    # Corrupt one chunk's version to 2.0
    corrupt_chunks = list(sample_test_corpus)
    corrupt_chunks[0] = dict(corrupt_chunks[0])
    corrupt_chunks[0]["version"] = "2.0"

    save_chunks_jsonl(corrupt_chunks, indexes_dir / "chunks.jsonl")
    build_bm25_index(corrupt_chunks, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(corrupt_chunks), 768), dtype=np.float32)
    q_client = build_qdrant_index(corrupt_chunks, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()

    manifest = build_manifest(sample_test_corpus, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    with pytest.raises(ValueError, match="version 2.0 != registry active version 1.0"):
        HybridRetriever(
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            model=mock_embedding_model,
        )


def test_retriever_inactive_document_rejected(tmp_path, mock_embedding_model, sample_test_corpus):
    """An index containing chunks for an unregistered or inactive document must be rejected."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    # Registry marks CMP-01 as inactive
    dummy_registry = [
        {"document_id": "LND-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_lnd_01", "allowed_roles": ["lending", "admin"]},
        {"document_id": "CMP-01", "active_status": False, "active_version": "1.0", "source_content_hash": "hash_cmp_01", "allowed_roles": ["compliance", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_test_corpus, indexes_dir / "chunks.jsonl")
    build_bm25_index(sample_test_corpus, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(sample_test_corpus), 768), dtype=np.float32)
    q_client = build_qdrant_index(sample_test_corpus, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()

    manifest = build_manifest(sample_test_corpus, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    with pytest.raises(ValueError, match="Active document set mismatch"):
        HybridRetriever(
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            model=mock_embedding_model,
        )


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


def test_duplicate_chunk_merging_and_rrf_calculation(retrieval_test_fixture):
    """When a chunk is returned by both dense and BM25, it must be merged with combined RRF score."""
    results = retrieval_test_fixture.retrieve("Branch Manager loan approval limits", role="lending", top_k=5)
    assert len(results) > 0

    top_chunk = results[0]
    assert top_chunk["chunk_id"] == "LND-01|1.0|2.0"

    assert top_chunk["dense_rank"] is not None
    assert top_chunk["bm25_rank"] is not None

    expected_score = (1.0 / (RRF_CONSTANT + top_chunk["dense_rank"])) + (
        1.0 / (RRF_CONSTANT + top_chunk["bm25_rank"])
    )
    assert pytest.approx(top_chunk["rrf_score"], rel=1e-5) == expected_score


def test_bm25_permission_filtering_before_truncation(retrieval_test_fixture):
    """BM25 search must filter permissions across entire corpus so allowed docs are not starved."""
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


def test_authorized_access_lending_lnd01(retrieval_test_fixture):
    """Authorized lending user successfully retrieves permitted LND-01 policy chunks."""
    results = retrieval_test_fixture.retrieve("personal loan approval authority limits", role="lending", top_k=5)
    assert len(results) > 0
    retrieved_docs = [r["document_id"] for r in results]
    assert "LND-01" in retrieved_docs
    for r in results:
        assert "lending" in r["allowed_roles"]


def test_unauthorized_access_general_hr04(tmp_path, mock_embedding_model):
    """General user querying restricted HR-04 content receives 0 HR-04 chunks across dense, BM25, and RRF."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    chunks = [
        {
            "chunk_id": "HR-04|1.0|2.0",
            "document_id": "HR-04",
            "title": "Executive Compensation Policy",
            "department": "Human Resources",
            "section_number": "2.0",
            "section_heading": "Band 4 Salary Ranges",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["hr", "admin"],
            "classification": "Strictly Confidential",
            "source_pdf": "data/documents/hr_04.pdf",
            "source_hash": "hash_hr_04",
            "text": "HR-04 | Executive Compensation\nSection 2.0: Band 4 Salary Ranges\n\nExecutive salaries are confidential.",
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
            "text": "HR-01 | Leave Policy\nSection 2.0: Annual Leave\n\nAll employees receive 18 days annual leave.",
        },
    ]

    dummy_registry = [
        {"document_id": "HR-04", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_04", "allowed_roles": ["hr", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(chunks, indexes_dir / "chunks.jsonl")
    build_bm25_index(chunks, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(chunks), 768), dtype=np.float32)
    q_client = build_qdrant_index(chunks, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()
    manifest = build_manifest(chunks, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    retriever = HybridRetriever(
        indexes_dir=indexes_dir,
        registry_path=registry_path,
        model=mock_embedding_model,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    try:
        query = "What are the executive compensation brackets for Band 4?"
        bm25_hits = retriever.bm25_search(query, role="general", candidate_k=5)
        assert all(h["document_id"] != "HR-04" for h in bm25_hits)

        dense_hits = retriever.dense_search(query, role="general", candidate_k=5)
        assert all(h["document_id"] != "HR-04" for h in dense_hits)

        rrf_hits = retriever.retrieve(query, role="general", top_k=5)
        assert all(h["document_id"] != "HR-04" for h in rrf_hits)
    finally:
        retriever.close()


def test_unauthorized_access_lending_cmp01(retrieval_test_fixture):
    """Lending user querying KYC content receives 0 CMP-01 chunks across BM25, dense, and RRF."""
    query = "What are the internal investigation response windows for AML alerts?"
    bm25_hits = retrieval_test_fixture.bm25_search(query, role="lending", candidate_k=5)
    assert all(h["document_id"] != "CMP-01" for h in bm25_hits)

    dense_hits = retrieval_test_fixture.dense_search(query, role="lending", candidate_k=5)
    assert all(h["document_id"] != "CMP-01" for h in dense_hits)

    rrf_hits = retrieval_test_fixture.retrieve(query, role="lending", top_k=5)
    assert all(h["document_id"] != "CMP-01" for h in rrf_hits)


def test_registry_permission_mismatch_fails_initialization(tmp_path, mock_embedding_model, sample_test_corpus):
    """Retriever initialization fails when index permissions diverge from registry."""
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir(parents=True)
    registry_path = tmp_path / "document_registry.json"

    dummy_registry = [
        {"document_id": "LND-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_lnd_01", "allowed_roles": ["admin"]},
        {"document_id": "CMP-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_cmp_01", "allowed_roles": ["compliance", "admin"]},
        {"document_id": "HR-01", "active_status": True, "active_version": "1.0", "source_content_hash": "hash_hr_01", "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"]},
    ]
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_test_corpus, indexes_dir / "chunks.jsonl")
    build_bm25_index(sample_test_corpus, indexes_dir / "bm25")
    raw_vectors = np.zeros((len(sample_test_corpus), 768), dtype=np.float32)
    q_client = build_qdrant_index(sample_test_corpus, raw_vectors, indexes_dir / "qdrant", DEFAULT_COLLECTION_NAME)
    q_client.close()

    manifest = build_manifest(sample_test_corpus, "mock-model", 768, DEFAULT_COLLECTION_NAME, str(indexes_dir / "bm25"))
    save_manifest(manifest, indexes_dir / "index_manifest.json")

    with pytest.raises(ValueError, match="Permission metadata mismatch for LND-01"):
        HybridRetriever(
            indexes_dir=indexes_dir,
            registry_path=registry_path,
            model=mock_embedding_model,
        )


def test_registry_authoritative_when_chunk_claims_unauthorized_role(retrieval_test_fixture):
    """Registry permissions take precedence even if chunk claims an unauthorized role."""
    tampered_chunk = {
        "chunk_id": "LND-01|1.0|2.0",
        "document_id": "LND-01",
        "version": "1.0",
        "source_hash": "hash_lnd_01",
        "allowed_roles": ["general", "lending", "admin"],
    }
    assert not retrieval_test_fixture.is_candidate_valid(tampered_chunk, role="general")
    assert retrieval_test_fixture.is_candidate_valid(tampered_chunk, role="lending")


def test_is_candidate_valid_fails_closed_missing_or_invalid_registry_roles(retrieval_test_fixture):
    """is_candidate_valid fails closed if registry permissions are missing, empty, or invalid."""
    valid_chunk = {
        "chunk_id": "LND-01|1.0|2.0",
        "document_id": "LND-01",
        "version": "1.0",
        "source_hash": "hash_lnd_01",
        "allowed_roles": ["lending", "admin"],
    }

    # Backup original LND-01 registry entry
    orig_entry = dict(retrieval_test_fixture.active_registry["LND-01"])

    try:
        # 1. Missing allowed_roles in registry
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        del retrieval_test_fixture.active_registry["LND-01"]["allowed_roles"]
        assert not retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")

        # 2. Empty allowed_roles in registry
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        retrieval_test_fixture.active_registry["LND-01"]["allowed_roles"] = []
        assert not retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")

        # 3. Non-list allowed_roles in registry
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        retrieval_test_fixture.active_registry["LND-01"]["allowed_roles"] = "lending,admin"
        assert not retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")

        # 4. Duplicate roles in registry
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        retrieval_test_fixture.active_registry["LND-01"]["allowed_roles"] = ["lending", "lending"]
        assert not retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")

        # 5. Invalid role name in registry
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        retrieval_test_fixture.active_registry["LND-01"]["allowed_roles"] = ["lending", "root"]
        assert not retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")

        # 6. Missing allowed_roles in chunk metadata
        retrieval_test_fixture.active_registry["LND-01"] = dict(orig_entry)
        chunk_missing_roles = dict(valid_chunk)
        del chunk_missing_roles["allowed_roles"]
        assert not retrieval_test_fixture.is_candidate_valid(chunk_missing_roles, role="lending")

        # 7. Non-list allowed_roles in chunk metadata
        chunk_bad_roles = dict(valid_chunk)
        chunk_bad_roles["allowed_roles"] = "lending"
        assert not retrieval_test_fixture.is_candidate_valid(chunk_bad_roles, role="lending")

        # 8. Normal authorized retrieval still passes
        assert retrieval_test_fixture.is_candidate_valid(valid_chunk, role="lending")
    finally:
        retrieval_test_fixture.active_registry["LND-01"] = orig_entry
