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
