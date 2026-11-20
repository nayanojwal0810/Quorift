"""Tests for Qdrant vector indexing, BM25 lexical indexing, and manifest generation."""

from pathlib import Path
import numpy as np
import pytest

from quorift.indexing import (
    build_bm25_index,
    build_manifest,
    build_qdrant_index,
    chunk_id_to_uuid,
    load_bm25_index,
    load_manifest,
    load_qdrant_index,
    save_manifest,
    search_bm25,
    search_qdrant,
)


@pytest.fixture
def sample_chunks():
    """Lightweight test corpus of 4 chunks with different permissions and topics."""
    return [
        {
            "chunk_id": "LND-01|1.0|2.0",
            "document_id": "LND-01",
            "title": "Personal Loan Approval Authority",
            "department": "Retail Lending",
            "section_number": "2.0",
            "section_heading": "Approval Limits",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["lending", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/lnd_01.pdf",
            "source_hash": "hash_lnd_01",
            "text": "LND-01 | Personal Loan Approval Authority\nSection 2.0: Approval Limits\n\nBranch Manager may approve personal loans up to INR 10 lakh.",
        },
        {
            "chunk_id": "CMP-01|1.0|2.0",
            "document_id": "CMP-01",
            "title": "Customer KYC Verification",
            "department": "Compliance",
            "section_number": "2.0",
            "section_heading": "Officially Valid Documents",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["compliance", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/cmp_01.pdf",
            "source_hash": "hash_cmp_01",
            "text": "CMP-01 | Customer KYC Verification\nSection 2.0: Officially Valid Documents\n\nPassport, Voter ID, and Driving Licence are valid identity proofs.",
        },
        {
            "chunk_id": "HR-01|1.0|2.0",
            "document_id": "HR-01",
            "title": "Leave and Attendance Policy",
            "department": "Human Resources",
            "section_number": "2.0",
            "section_heading": "Annual Leave Entitlement",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["general", "lending", "finance", "compliance", "hr", "admin"],
            "classification": "General Internal",
            "source_pdf": "data/documents/hr_01.pdf",
            "source_hash": "hash_hr_01",
            "text": "HR-01 | Leave and Attendance Policy\nSection 2.0: Annual Leave Entitlement\n\nConfirmed employees receive 18 working days of annual leave.",
        },
        {
            "chunk_id": "FIN-01|1.0|2.0",
            "document_id": "FIN-01",
            "title": "Employee Reimbursement Authority",
            "department": "Finance",
            "section_number": "2.0",
            "section_heading": "Reimbursement Tiers",
            "version": "1.0",
            "effective_date": "2026-01-15",
            "page_start": 1,
            "page_end": 1,
            "allowed_roles": ["finance", "admin"],
            "classification": "Confidential",
            "source_pdf": "data/documents/fin_01.pdf",
            "source_hash": "hash_fin_01",
            "text": "FIN-01 | Employee Reimbursement Authority\nSection 2.0: Reimbursement Tiers\n\nTeam Lead may approve reimbursement claims up to INR 5,000 per claim.",
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Deterministic normalized 8-dimensional test vectors."""
    raw = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# 1. Qdrant stores and retrieves points
def test_qdrant_build_and_search(sample_chunks, sample_embeddings, tmp_path):
    qdrant_dir = tmp_path / "qdrant"
    client = build_qdrant_index(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        storage_path=qdrant_dir,
        collection_name="test_collection",
    )
    client.close()

    reloaded_client = load_qdrant_index(qdrant_dir)
    info = reloaded_client.get_collection("test_collection")
    assert info.points_count == len(sample_chunks)

    # Search for first chunk vector
    hits = search_qdrant(
        client=reloaded_client,
        query_vector=sample_embeddings[0],
        collection_name="test_collection",
        limit=2,
    )
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "LND-01|1.0|2.0"
    assert hits[0]["document_id"] == "LND-01"
    assert hits[0]["score"] > 0.99
    reloaded_client.close()


# 2. BM25 indexing saves, reloads, and retrieves correctly
def test_bm25_build_and_search(sample_chunks, tmp_path):
    bm25_dir = tmp_path / "bm25"
    retriever, chunk_ids = build_bm25_index(sample_chunks, bm25_dir)
    assert len(chunk_ids) == len(sample_chunks)

    reloaded_retriever, reloaded_ids = load_bm25_index(bm25_dir)
    assert reloaded_ids == chunk_ids

    # Query for loan
    hits = search_bm25(reloaded_retriever, reloaded_ids, query="personal loan approval", limit=2)
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "LND-01|1.0|2.0"
    assert hits[0]["document_id"] == "LND-01"


# 3. Chunk IDs remain consistent across both indexes
def test_chunk_ids_consistency(sample_chunks, sample_embeddings, tmp_path):
    qdrant_dir = tmp_path / "qdrant"
    bm25_dir = tmp_path / "bm25"

    q_client = build_qdrant_index(sample_chunks, sample_embeddings, qdrant_dir, "test_col")
    q_client.close()

    _, bm25_chunk_ids = build_bm25_index(sample_chunks, bm25_dir)

    reloaded_q = load_qdrant_index(qdrant_dir)
    q_points = reloaded_q.scroll(collection_name="test_col", limit=10)[0]
    q_chunk_ids = [p.payload["chunk_id"] for p in q_points]
    reloaded_q.close()

    assert set(q_chunk_ids) == set(bm25_chunk_ids)
    assert set(q_chunk_ids) == {c["chunk_id"] for c in sample_chunks}


# 4. Stored metadata survives reload
def test_metadata_survives_reload(sample_chunks, sample_embeddings, tmp_path):
    qdrant_dir = tmp_path / "qdrant"
    client = build_qdrant_index(sample_chunks, sample_embeddings, qdrant_dir, "meta_col")
    client.close()

    reloaded = load_qdrant_index(qdrant_dir)
    point = reloaded.retrieve("meta_col", [chunk_id_to_uuid("LND-01|1.0|2.0")])[0]
    payload = point.payload
    assert payload["document_id"] == "LND-01"
    assert payload["title"] == "Personal Loan Approval Authority"
    assert payload["allowed_roles"] == ["lending", "admin"]
    assert payload["source_hash"] == "hash_lnd_01"
    reloaded.close()


# 5. Role-filtered Qdrant search excludes disallowed documents
def test_role_filtering(sample_chunks, sample_embeddings, tmp_path):
    qdrant_dir = tmp_path / "qdrant"
    client = build_qdrant_index(sample_chunks, sample_embeddings, qdrant_dir, "role_col")
    client.close()

    reloaded = load_qdrant_index(qdrant_dir)
    # Lending role filter: should see LND-01 and HR-01 (general), but NOT CMP-01 or FIN-01
    hits = search_qdrant(
        client=reloaded,
        query_vector=sample_embeddings[0],
        collection_name="role_col",
        limit=10,
        role_filter="lending",
    )
    hit_docs = [h["document_id"] for h in hits]
    assert "LND-01" in hit_docs
    assert "HR-01" in hit_docs
    assert "CMP-01" not in hit_docs
    assert "FIN-01" not in hit_docs

    # Compliance role filter: should see CMP-01 and HR-01, but NOT LND-01 or FIN-01
    hits_cmp = search_qdrant(
        client=reloaded,
        query_vector=sample_embeddings[1],
        collection_name="role_col",
        limit=10,
        role_filter="compliance",
    )
    hit_cmp_docs = [h["document_id"] for h in hits_cmp]
    assert "CMP-01" in hit_cmp_docs
    assert "HR-01" in hit_cmp_docs
    assert "LND-01" not in hit_cmp_docs
    assert "FIN-01" not in hit_cmp_docs
    reloaded.close()


# 6. Rebuilding does not duplicate entries
def test_rebuild_does_not_duplicate(sample_chunks, sample_embeddings, tmp_path):
    qdrant_dir = tmp_path / "qdrant"
    c1 = build_qdrant_index(sample_chunks, sample_embeddings, qdrant_dir, "dedup_col")
    c1.close()

    # Rebuild in same location
    c2 = build_qdrant_index(sample_chunks, sample_embeddings, qdrant_dir, "dedup_col")
    info = c2.get_collection("dedup_col")
    assert info.points_count == len(sample_chunks)
    c2.close()


# 7. Manifest generation and reload verification
def test_manifest_round_trip(sample_chunks, tmp_path):
    manifest_path = tmp_path / "index_manifest.json"
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="test_col",
        bm25_dir="test_bm25",
    )
    save_manifest(manifest, manifest_path)
    assert manifest_path.exists()

    reloaded = load_manifest(manifest_path)
    assert reloaded["total_chunks"] == len(sample_chunks)
    assert reloaded["total_active_documents"] == 4
    assert reloaded["embedding_dimension"] == 8
