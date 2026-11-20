"""Dense embedding, Qdrant vector indexing, and BM25 lexical indexing for Quorift."""

import json
from pathlib import Path
from typing import Any
import uuid

import bm25s
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_COLLECTION_NAME = "banking_policies"
VECTOR_DIMENSION = 768


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID5 string from a chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load the sentence-transformers embedding model on CPU."""
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    batch_size: int = 4,
    is_query: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Compute normalized dense embeddings using BGE-base-en-v1.5.

    Prepend query instruction if is_query is True; do not prepend for passages.
    """
    if model is None:
        model = get_embedding_model(device=device)

    prepared_texts = [
        f"{BGE_QUERY_PREFIX}{t}" if is_query else t for t in texts
    ]

    embeddings = model.encode(
        prepared_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings_np = np.asarray(embeddings, dtype=np.float32)

    if embeddings_np.shape[1] != VECTOR_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: expected {VECTOR_DIMENSION}, got {embeddings_np.shape[1]}"
        )
    if not np.all(np.isfinite(embeddings_np)):
        raise ValueError("Generated embeddings contain non-finite numbers (NaN or Inf)")

    return embeddings_np


def build_qdrant_index(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    storage_path: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 32,
) -> QdrantClient:
    """Build or recreate a persistent local Qdrant collection for policy chunks."""
    storage_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(storage_path))

    # Recreate collection to ensure a clean build
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points: list[PointStruct] = []
    for idx, chunk in enumerate(chunks):
        point_id = chunk_id_to_uuid(chunk["chunk_id"])
        vec = embeddings[idx].tolist()
        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload=chunk,
            )
        )

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    return client


def load_qdrant_index(
    storage_path: Path,
) -> QdrantClient:
    """Connect to an existing persistent local Qdrant storage directory."""
    if not storage_path.exists():
        raise FileNotFoundError(f"Qdrant storage path does not exist: {storage_path}")
    return QdrantClient(path=str(storage_path))


def search_qdrant(
    client: QdrantClient,
    query_vector: list[float] | np.ndarray,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    limit: int = 5,
    role_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant using dense cosine similarity with optional role filtering."""
    vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
    query_filter: Filter | None = None

    if role_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="allowed_roles",
                    match=MatchValue(value=role_filter),
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=vec,
        query_filter=query_filter,
        limit=limit,
    ).points

    hits: list[dict[str, Any]] = []
    for rank, p in enumerate(results, start=1):
        payload = p.payload or {}
        hits.append(
            {
                "rank": rank,
                "chunk_id": payload.get("chunk_id", ""),
                "document_id": payload.get("document_id", ""),
                "section_number": payload.get("section_number", ""),
                "score": float(p.score),
                "payload": payload,
            }
        )
    return hits


def build_bm25_index(
    chunks: list[dict[str, Any]],
    storage_path: Path,
) -> tuple[bm25s.BM25, list[str]]:
    """Build and save a BM25 index and its chunk ID mapping to disk."""
    storage_path.mkdir(parents=True, exist_ok=True)
    corpus = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(str(storage_path))

    mapping_file = storage_path / "chunk_ids.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)

    return retriever, chunk_ids


def load_bm25_index(
    storage_path: Path,
) -> tuple[bm25s.BM25, list[str]]:
    """Load a persisted BM25 index and its chunk ID mapping from disk."""
    if not storage_path.exists():
        raise FileNotFoundError(f"BM25 index path does not exist: {storage_path}")

    retriever = bm25s.BM25.load(str(storage_path), load_corpus=True)
    mapping_file = storage_path / "chunk_ids.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"BM25 chunk ID mapping file missing: {mapping_file}")

    with open(mapping_file, encoding="utf-8") as f:
        chunk_ids = json.load(f)

    return retriever, chunk_ids


def search_bm25(
    retriever: bm25s.BM25,
    chunk_ids: list[str],
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Query the BM25 index and return ranked results with scores and chunk IDs."""
    query_tokens = bm25s.tokenize(query, show_progress=False)
    top_indices, top_scores = retriever.retrieve(
        query_tokens,
        k=min(limit, len(chunk_ids)),
        show_progress=False,
    )

    indices = top_indices[0]
    scores = top_scores[0]

    hits: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        cid = chunk_ids[idx]
        parts = cid.split("|")
        doc_id = parts[0] if len(parts) >= 1 else ""
        sec_num = parts[2] if len(parts) >= 3 else ""
        hits.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "document_id": doc_id,
                "section_number": sec_num,
                "score": float(score),
            }
        )
    return hits


def build_manifest(
    chunks: list[dict[str, Any]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int = VECTOR_DIMENSION,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    bm25_dir: str = "bm25",
) -> dict[str, Any]:
    """Construct deterministic metadata manifest for the generated indexes."""
    doc_map: dict[str, dict[str, str]] = {}
    for c in chunks:
        doc_id = c["document_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "document_id": doc_id,
                "version": c["version"],
                "active_version": c["version"],
                "source_hash": c["source_hash"],
            }

    active_docs = [doc_map[k] for k in sorted(doc_map.keys())]

    return {
        "manifest_version": "1.0",
        "embedding_model": model_name,
        "embedding_dimension": embedding_dim,
        "collection_name": collection_name,
        "bm25_location": bm25_dir,
        "total_active_documents": len(active_docs),
        "total_chunks": len(chunks),
        "active_documents": active_docs,
    }


def save_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Save index manifest JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(input_path: Path) -> dict[str, Any]:
    """Load index manifest JSON from disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Manifest not found: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def verify_index_identity(
    indexes_dir: Path,
    registry_path: Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    validate_registry: bool = True,
) -> dict[str, Any]:
    """Verify that chunks.jsonl, Qdrant vectors, and BM25 index contain the exact same chunk IDs.

    Also validates active document versions and source hashes against registry and manifest,
    and checks that Qdrant payloads faithfully correspond to the chunk records.
    """
    from quorift.config import paths
    from quorift.ingestion import load_chunks_jsonl

    chunks_file = indexes_dir / "chunks.jsonl"
    manifest_file = indexes_dir / "index_manifest.json"
    qdrant_dir = indexes_dir / "qdrant"
    bm25_dir = indexes_dir / "bm25"

    if not chunks_file.exists():
        raise FileNotFoundError(f"Missing chunks.jsonl in {indexes_dir}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing index_manifest.json in {indexes_dir}")
    if not qdrant_dir.exists():
        raise FileNotFoundError(f"Missing qdrant directory in {indexes_dir}")
    if not bm25_dir.exists():
        raise FileNotFoundError(f"Missing bm25 directory in {indexes_dir}")

    # 1. Chunks from JSONL
    chunks = load_chunks_jsonl(chunks_file)
    jsonl_chunk_ids = [c["chunk_id"] for c in chunks]
    if len(jsonl_chunk_ids) != len(set(jsonl_chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected in chunks.jsonl")
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # 2. Chunks from BM25 mapping
    _, bm25_chunk_ids = load_bm25_index(bm25_dir)
    if len(bm25_chunk_ids) != len(set(bm25_chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected in BM25 index mapping")

    # 3. Chunks from Qdrant collection
    q_client = load_qdrant_index(qdrant_dir)
    try:
        q_points = []
        next_page = None
        while True:
            batch, next_page = q_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            q_points.extend(batch)
            if next_page is None:
                break
    finally:
        q_client.close()

    qdrant_chunk_ids = [p.payload["chunk_id"] for p in q_points if p.payload]
    if len(qdrant_chunk_ids) != len(set(qdrant_chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected in Qdrant collection")

    # 4. Compare all three complete sets
    set_jsonl = set(jsonl_chunk_ids)
    set_bm25 = set(bm25_chunk_ids)
    set_qdrant = set(qdrant_chunk_ids)

    if set_jsonl != set_bm25:
        missing = set_jsonl - set_bm25
        extra = set_bm25 - set_jsonl
        raise ValueError(f"Chunk ID mismatch between chunks.jsonl and BM25 index: missing={missing}, extra={extra}")

    if set_jsonl != set_qdrant:
        missing = set_jsonl - set_qdrant
        extra = set_qdrant - set_jsonl
        raise ValueError(f"Chunk ID mismatch between chunks.jsonl and Qdrant index: missing={missing}, extra={extra}")

    # 5. Verify active document versions and source hashes against current registry and manifest
    manifest = load_manifest(manifest_file)
    reg_path = registry_path or (paths.metadata_dir / "document_registry.json")

    if validate_registry and reg_path.exists():
        with open(reg_path, encoding="utf-8") as f:
            registry = json.load(f)

        active_reg_docs = [r for r in registry if r.get("active_status", True)]
        reg_active_ids = [r["document_id"] for r in active_reg_docs]
        if len(reg_active_ids) != len(set(reg_active_ids)):
            raise ValueError("Duplicate active document identities in registry")
        reg_active_map = {r["document_id"]: r for r in active_reg_docs}

        manifest_active_docs = manifest.get("active_documents", [])
        manifest_active_ids = [d["document_id"] for d in manifest_active_docs]
        if len(manifest_active_ids) != len(set(manifest_active_ids)):
            raise ValueError("Duplicate active document identities in index manifest")
        manifest_active_map = {d["document_id"]: d for d in manifest_active_docs}

        # Compare exact active document sets
        if set(reg_active_map.keys()) != set(manifest_active_map.keys()):
            missing_in_manifest = set(reg_active_map.keys()) - set(manifest_active_map.keys())
            extra_in_manifest = set(manifest_active_map.keys()) - set(reg_active_map.keys())
            raise ValueError(
                f"Active document set mismatch between registry and manifest: "
                f"missing={missing_in_manifest}, extra={extra_in_manifest}"
            )

        for doc_id, reg_doc in reg_active_map.items():
            man_doc = manifest_active_map[doc_id]
            expected_version = reg_doc.get("active_version")
            if expected_version is None:
                raise ValueError(f"Document {doc_id} in registry has missing active_version")
            man_version = man_doc.get("active_version") or man_doc.get("version")
            if man_version != expected_version:
                raise ValueError(
                    f"Active version mismatch for {doc_id} in manifest: "
                    f"manifest={man_version} vs registry={expected_version}"
                )
            if man_doc.get("source_hash") != reg_doc.get("source_content_hash"):
                raise ValueError(
                    f"Source hash mismatch for {doc_id} in manifest: "
                    f"manifest={man_doc.get('source_hash')} vs registry={reg_doc.get('source_content_hash')}"
                )

        # Confirm all chunks correspond to active documents and active versions
        chunk_doc_ids = set()
        for c in chunks:
            doc_id = c["document_id"]
            chunk_doc_ids.add(doc_id)
            if doc_id not in reg_active_map:
                raise ValueError(
                    f"Unexpected inactive or unregistered document {doc_id} found in chunk {c['chunk_id']}"
                )
            reg_doc = reg_active_map[doc_id]
            if c["version"] != reg_doc["active_version"]:
                raise ValueError(
                    f"Chunk {c['chunk_id']} version {c['version']} != registry active version {reg_doc['active_version']}"
                )
            if c["source_hash"] != reg_doc["source_content_hash"]:
                raise ValueError(
                    f"Chunk {c['chunk_id']} source_hash mismatch with registry"
                )

        if chunk_doc_ids != set(reg_active_map.keys()):
            missing_chunks_docs = set(reg_active_map.keys()) - chunk_doc_ids
            raise ValueError(
                f"Active documents in registry have no chunks in index: {missing_chunks_docs}"
            )

    # 6. Confirm Qdrant payload metadata corresponds faithfully to chunk records
    for p in q_points:
        cid = p.payload["chunk_id"]
        expected_chunk = chunk_map[cid]
        for key in ("document_id", "section_number", "version", "title", "allowed_roles", "source_hash"):
            if key in expected_chunk and p.payload.get(key) != expected_chunk.get(key):
                raise ValueError(f"Qdrant payload field mismatch for {cid} on {key}")

    return {
        "status": "verified",
        "total_chunks": len(set_jsonl),
        "chunk_ids": sorted(list(set_jsonl)),
    }


def safe_promote_indexes(
    staging_dir: Path,
    active_dir: Path,
    backup_dir: Path,
    registry_path: Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Safely promote verified staged indexes to the active directory with atomic rollback."""
    import shutil

    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_dir}")

    # Situation D — A backup already exists before rebuilding
    if backup_dir.exists():
        if active_dir.exists():
            raise RuntimeError(
                f"Unresolved backup directory found at '{backup_dir}' while active index '{active_dir}' exists. "
                "An interrupted build must be resolved before proceeding. "
                "Do not automatically delete existing backups."
            )
        else:
            # Active is missing but backup exists: restore only if structurally complete
            try:
                verify_index_identity(
                    indexes_dir=backup_dir,
                    collection_name=collection_name,
                    validate_registry=False,
                )
                backup_dir.rename(active_dir)
            except Exception as e:
                raise RuntimeError(
                    f"Active index is missing and pre-existing backup at '{backup_dir}' "
                    f"cannot be verified as structurally complete: {e}. Preserving backup and stopping."
                ) from e

    # Situation B — Staging is invalid: verify staging before touching active index
    verify_index_identity(
        indexes_dir=staging_dir,
        registry_path=registry_path,
        collection_name=collection_name,
        validate_registry=True,
    )

    # Situation A — Normal rebuild: move active index to temporary backup
    had_active = active_dir.exists()
    if had_active:
        active_dir.rename(backup_dir)

    try:
        staging_dir.rename(active_dir)
        # Verify promoted active directory
        verify_index_identity(
            indexes_dir=active_dir,
            registry_path=registry_path,
            collection_name=collection_name,
            validate_registry=True,
        )
        # Promotion succeeded: delete temporary backup
        if had_active and backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception as promote_err:
        # Situation C — Promotion fails: rollback to previous active index
        rollback_failed = False
        rollback_err_parts = []

        if active_dir.exists():
            try:
                shutil.rmtree(active_dir)
            except Exception as e:
                rollback_failed = True
                rollback_err_parts.append(f"Failed to remove incomplete active dir: {e}")

        if had_active and backup_dir.exists():
            try:
                backup_dir.rename(active_dir)
            except Exception as e:
                rollback_failed = True
                rollback_err_parts.append(f"Failed to restore backup to active: {e}")

        if had_active and not active_dir.exists():
            rollback_failed = True
            rollback_err_parts.append("Active directory is missing after rollback attempt")

        if rollback_failed:
            err_details = "; ".join(rollback_err_parts)
            raise RuntimeError(
                f"Index promotion failed ({promote_err}) AND ROLLBACK FAILED ({err_details}). "
                f"Backup preserved at '{backup_dir}'."
            ) from promote_err

        raise RuntimeError(
            f"Index promotion failed and was rolled back to previous active index. Error: {promote_err}"
        ) from promote_err
