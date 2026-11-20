"""Tests for Qdrant vector indexing, BM25 lexical indexing, and manifest generation."""

from pathlib import Path
import numpy as np
import pytest

from quorift.indexing import (
    DEFAULT_COLLECTION_NAME,
    build_bm25_index,
    build_manifest,
    build_qdrant_index,
    chunk_id_to_uuid,
    load_bm25_index,
    load_manifest,
    load_qdrant_index,
    safe_promote_indexes,
    save_manifest,
    search_bm25,
    search_qdrant,
    verify_index_identity,
)
from quorift.ingestion import save_chunks_jsonl


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


# 8. Index identity verification catches mismatched chunk IDs
def test_verify_index_identity_mismatched_chunks(sample_chunks, sample_embeddings, tmp_path):
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    q_dir = idx_dir / "qdrant"
    b_dir = idx_dir / "bm25"
    chunks_path = idx_dir / "chunks.jsonl"
    manifest_path = idx_dir / "index_manifest.json"

    # Save chunks.jsonl with 4 chunks
    save_chunks_jsonl(sample_chunks, chunks_path)

    # Build Qdrant with all 4 chunks
    q_client = build_qdrant_index(sample_chunks, sample_embeddings, q_dir, "test_col")
    q_client.close()

    # Build BM25 with only first 3 chunks (deliberate mismatch)
    build_bm25_index(sample_chunks[:3], b_dir)

    # Save manifest
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="test_col",
        bm25_dir=str(b_dir),
    )
    save_manifest(manifest, manifest_path)

    with pytest.raises(ValueError, match="Chunk ID mismatch"):
        verify_index_identity(idx_dir, collection_name="test_col")


# 9. Successful staging promotion replaces active directory
def test_safe_promote_indexes_success(sample_chunks, sample_embeddings, tmp_path):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    backup_dir = tmp_path / "backup"
    staging_dir.mkdir()

    # Create dummy registry matching sample chunks
    dummy_registry = [
        {
            "document_id": c["document_id"],
            "active_status": True,
            "active_version": c["version"],
            "source_content_hash": c["source_hash"],
            "allowed_roles": c.get("allowed_roles", []),
        }
        for c in sample_chunks
    ]
    reg_file = tmp_path / "test_registry.json"
    import json
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    # Build valid indexes in staging
    save_chunks_jsonl(sample_chunks, staging_dir / "chunks.jsonl")
    q_client = build_qdrant_index(
        sample_chunks, sample_embeddings, staging_dir / "qdrant", "quorift_chunks"
    )
    q_client.close()
    build_bm25_index(sample_chunks, staging_dir / "bm25")
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="quorift_chunks",
        bm25_dir=str(staging_dir / "bm25"),
    )
    save_manifest(manifest, staging_dir / "index_manifest.json")

    safe_promote_indexes(
        staging_dir=staging_dir,
        active_dir=active_dir,
        backup_dir=backup_dir,
        registry_path=reg_file,
        collection_name="quorift_chunks",
    )

    assert active_dir.exists()
    assert (active_dir / "chunks.jsonl").exists()
    assert (active_dir / "index_manifest.json").exists()
    assert not staging_dir.exists()
    assert not backup_dir.exists()


# 10. Existing backup must not be silently deleted when active exists
def test_safe_promote_indexes_existing_backup_not_deleted(tmp_path):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    backup_dir = tmp_path / "backup"
    staging_dir.mkdir()
    active_dir.mkdir()
    backup_dir.mkdir()

    sentinel_backup = backup_dir / "preserved_backup.txt"
    sentinel_backup.write_text("critical_backup_data", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unresolved backup directory found"):
        safe_promote_indexes(
            staging_dir=staging_dir,
            active_dir=active_dir,
            backup_dir=backup_dir,
        )

    # Backup directory and content must remain untouched
    assert backup_dir.exists()
    assert sentinel_backup.exists()
    assert sentinel_backup.read_text(encoding="utf-8") == "critical_backup_data"


# 11. Failed staged verification leaves existing active index intact
def test_safe_promote_indexes_verification_failure_preserves_active(tmp_path):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    backup_dir = tmp_path / "backup"
    active_dir.mkdir()
    staging_dir.mkdir()

    sentinel = active_dir / "active_sentinel.txt"
    sentinel.write_text("existing_active_content", encoding="utf-8")

    # Put corrupt/empty staging index
    (staging_dir / "chunks.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        safe_promote_indexes(
            staging_dir=staging_dir,
            active_dir=active_dir,
            backup_dir=backup_dir,
        )

    # Active dir must still exist with sentinel intact
    assert active_dir.exists()
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "existing_active_content"
    assert not backup_dir.exists()


# 12. Failed promotion restores previous active index
def test_safe_promote_indexes_failed_promotion_restores_active(
    sample_chunks, sample_embeddings, tmp_path, monkeypatch
):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    backup_dir = tmp_path / "backup"
    staging_dir.mkdir()
    active_dir.mkdir()

    sentinel_active = active_dir / "active_original.txt"
    sentinel_active.write_text("active_v1", encoding="utf-8")

    dummy_registry = [
        {
            "document_id": c["document_id"],
            "active_status": True,
            "active_version": c["version"],
            "source_content_hash": c["source_hash"],
            "allowed_roles": c.get("allowed_roles", []),
        }
        for c in sample_chunks
    ]
    reg_file = tmp_path / "test_registry.json"
    import json
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    # Build valid indexes in staging
    save_chunks_jsonl(sample_chunks, staging_dir / "chunks.jsonl")
    q_client = build_qdrant_index(
        sample_chunks, sample_embeddings, staging_dir / "qdrant", "quorift_chunks"
    )
    q_client.close()
    build_bm25_index(sample_chunks, staging_dir / "bm25")
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="quorift_chunks",
        bm25_dir=str(staging_dir / "bm25"),
    )
    save_manifest(manifest, staging_dir / "index_manifest.json")

    # Deliberately fail the post-promotion verification
    orig_verify = verify_index_identity
    calls = []

    def failing_verify(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:  # Second call is post-promotion verification
            raise ValueError("Deliberate post-promotion failure")
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr("quorift.indexing.verify_index_identity", failing_verify)

    with pytest.raises(RuntimeError, match="rolled back to previous active index"):
        safe_promote_indexes(
            staging_dir=staging_dir,
            active_dir=active_dir,
            backup_dir=backup_dir,
            registry_path=reg_file,
            collection_name="quorift_chunks",
        )

    # Active index should be restored
    assert active_dir.exists()
    assert (active_dir / "active_original.txt").exists()
    assert (active_dir / "active_original.txt").read_text(encoding="utf-8") == "active_v1"


# 13. Failed rollback does not falsely report success and preserves backup
def test_safe_promote_indexes_failed_rollback_preserves_backup_and_reports_error(
    sample_chunks, sample_embeddings, tmp_path, monkeypatch
):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    backup_dir = tmp_path / "backup"
    staging_dir.mkdir()
    active_dir.mkdir()

    (active_dir / "sentinel.txt").write_text("orig_active", encoding="utf-8")

    dummy_registry = [
        {
            "document_id": c["document_id"],
            "active_status": True,
            "active_version": c["version"],
            "source_content_hash": c["source_hash"],
            "allowed_roles": c.get("allowed_roles", []),
        }
        for c in sample_chunks
    ]
    reg_file = tmp_path / "test_registry.json"
    import json
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_chunks, staging_dir / "chunks.jsonl")
    q_client = build_qdrant_index(
        sample_chunks, sample_embeddings, staging_dir / "qdrant", "quorift_chunks"
    )
    q_client.close()
    build_bm25_index(sample_chunks, staging_dir / "bm25")
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="quorift_chunks",
        bm25_dir=str(staging_dir / "bm25"),
    )
    save_manifest(manifest, staging_dir / "index_manifest.json")

    # Cause promotion to fail and also block rollback rename
    orig_verify = verify_index_identity
    calls = []

    def failing_verify(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("Deliberate promotion failure")
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr("quorift.indexing.verify_index_identity", failing_verify)

    orig_rename = Path.rename

    def failing_rename(self, target):
        if str(self) == str(backup_dir) and str(target) == str(active_dir):
            raise OSError("Injected disk failure during rollback")
        return orig_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)

    with pytest.raises(RuntimeError, match="ROLLBACK FAILED"):
        safe_promote_indexes(
            staging_dir=staging_dir,
            active_dir=active_dir,
            backup_dir=backup_dir,
            registry_path=reg_file,
            collection_name="quorift_chunks",
        )

    # Backup must be preserved
    assert backup_dir.exists()


# 14. Missing active index with recoverable backup is safely handled
def test_safe_promote_indexes_missing_active_recovers_backup(
    sample_chunks, sample_embeddings, tmp_path
):
    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"  # Does not exist initially
    backup_dir = tmp_path / "backup"
    staging_dir.mkdir()
    backup_dir.mkdir()

    # Put structurally valid index into backup
    save_chunks_jsonl(sample_chunks, backup_dir / "chunks.jsonl")
    q_client = build_qdrant_index(
        sample_chunks, sample_embeddings, backup_dir / "qdrant", "quorift_chunks"
    )
    q_client.close()
    build_bm25_index(sample_chunks, backup_dir / "bm25")
    manifest = build_manifest(
        chunks=sample_chunks,
        model_name="test-bge",
        embedding_dim=8,
        collection_name="quorift_chunks",
        bm25_dir=str(backup_dir / "bm25"),
    )
    save_manifest(manifest, backup_dir / "index_manifest.json")

    # Put valid index into staging as well
    dummy_registry = [
        {
            "document_id": c["document_id"],
            "active_status": True,
            "active_version": c["version"],
            "source_content_hash": c["source_hash"],
            "allowed_roles": c.get("allowed_roles", []),
        }
        for c in sample_chunks
    ]
    reg_file = tmp_path / "test_registry.json"
    import json
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(dummy_registry, f)

    save_chunks_jsonl(sample_chunks, staging_dir / "chunks.jsonl")
    q_client2 = build_qdrant_index(
        sample_chunks, sample_embeddings, staging_dir / "qdrant", "quorift_chunks"
    )
    q_client2.close()
    build_bm25_index(sample_chunks, staging_dir / "bm25")
    save_manifest(manifest, staging_dir / "index_manifest.json")

    safe_promote_indexes(
        staging_dir=staging_dir,
        active_dir=active_dir,
        backup_dir=backup_dir,
        registry_path=reg_file,
        collection_name="quorift_chunks",
    )

    assert active_dir.exists()
    assert (active_dir / "chunks.jsonl").exists()
    assert not backup_dir.exists()


# 15. Smoke checks raise error when target evidence is missing
def test_smoke_checks_fail_on_missing_evidence(sample_chunks, sample_embeddings, tmp_path):
    from unittest.mock import MagicMock
    from scripts.build_indexes import run_smoke_checks

    idx_dir = tmp_path / "idx_smoke"
    idx_dir.mkdir()

    # Build index with chunks that do NOT contain HR-01 or FIN-01
    missing_chunks = sample_chunks[:2]
    raw_vecs = np.zeros((len(missing_chunks), 768), dtype=np.float32)
    for i in range(len(missing_chunks)):
        raw_vecs[i, i] = 1.0

    save_chunks_jsonl(missing_chunks, idx_dir / "chunks.jsonl")
    q_client = build_qdrant_index(
        missing_chunks, raw_vecs, idx_dir / "qdrant", DEFAULT_COLLECTION_NAME
    )
    q_client.close()
    build_bm25_index(missing_chunks, idx_dir / "bm25")

    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.zeros((1, 768), dtype=np.float32))

    with pytest.raises(ValueError, match="Smoke check failed"):
        run_smoke_checks(idx_dir, mock_model)


# 16. Stale or mismatched permission metadata raises ValueError
def test_verify_index_identity_permission_mismatch(sample_chunks, sample_embeddings, tmp_path):
    """Indexes whose chunk permissions diverge from the trusted registry must fail verification."""
    idx_dir = tmp_path / "idx_perm_mismatch"
    idx_dir.mkdir()
    q_dir = idx_dir / "qdrant"
    b_dir = idx_dir / "bm25"
    chunks_path = idx_dir / "chunks.jsonl"
    manifest_path = idx_dir / "index_manifest.json"
    reg_file = tmp_path / "registry.json"

    # Registry only allows ['admin'] for LND-01
    reg_data = [
        {
            "document_id": c["document_id"],
            "active_status": True,
            "active_version": c["version"],
            "source_content_hash": c["source_hash"],
            "allowed_roles": ["admin"] if c["document_id"] == "LND-01" else c.get("allowed_roles", []),
        }
        for c in sample_chunks
    ]
    import json
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(reg_data, f)

    # Chunks still have ['lending', 'admin'] for LND-01
    save_chunks_jsonl(sample_chunks, chunks_path)
    manifest = build_manifest(sample_chunks, "test-model", 8, "quorift_chunks", str(b_dir))
    save_manifest(manifest, manifest_path)
    q_client = build_qdrant_index(sample_chunks, sample_embeddings, q_dir, "quorift_chunks")
    q_client.close()
    build_bm25_index(sample_chunks, b_dir)

    with pytest.raises(ValueError, match="Permission metadata mismatch for LND-01"):
        verify_index_identity(
            idx_dir,
            collection_name="quorift_chunks",
            validate_registry=True,
            registry_path=reg_file,
        )
