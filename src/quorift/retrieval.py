"""Baseline hybrid retrieval combining dense BGE vectors and BM25 lexical search via RRF."""

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from quorift.config import paths
from quorift.indexing import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    embed_texts,
    get_embedding_model,
    load_bm25_index,
    load_manifest,
    load_qdrant_index,
    search_bm25,
    search_qdrant,
)
from quorift.ingestion import load_chunks_jsonl

VALID_ROLES = {"general", "lending", "finance", "compliance", "hr", "admin"}
RRF_CONSTANT = 60
DEFAULT_CANDIDATE_LIMIT = 20


class HybridRetriever:
    """Manages loaded dense vector and BM25 lexical indexes for hybrid retrieval."""

    def __init__(
        self,
        indexes_dir: Path | None = None,
        model: SentenceTransformer | None = None,
        registry_path: Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.indexes_dir = indexes_dir or paths.indexes_dir
        self.collection_name = collection_name
        self.registry_path = registry_path or (paths.metadata_dir / "document_registry.json")

        if not self.indexes_dir.exists():
            raise FileNotFoundError(f"Indexes directory not found: {self.indexes_dir}")

        chunks_file = self.indexes_dir / "chunks.jsonl"
        manifest_file = self.indexes_dir / "index_manifest.json"
        qdrant_dir = self.indexes_dir / "qdrant"
        bm25_dir = self.indexes_dir / "bm25"

        # 1. Load chunk records
        raw_chunks = load_chunks_jsonl(chunks_file)
        self.chunks_by_id: dict[str, dict[str, Any]] = {
            c["chunk_id"]: c for c in raw_chunks
        }

        # 2. Load manifest
        self.manifest = load_manifest(manifest_file)

        # 3. Connect to Qdrant persistent local store
        self.qdrant_client: QdrantClient = load_qdrant_index(qdrant_dir)

        # 4. Load BM25 index and chunk ID map
        self.bm25_retriever, self.bm25_chunk_ids = load_bm25_index(bm25_dir)

        # 5. Embedding model (loaded once or passed in)
        if model is not None:
            self.model = model
        else:
            model_name = self.manifest.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            self.model = get_embedding_model(model_name=model_name, device="cpu")

    def dense_search(
        self,
        question: str,
        role: str,
        candidate_k: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Retrieve top candidate chunks using dense BGE query vectors and Qdrant role filtering."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(VALID_ROLES)}")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")

        q_vec = embed_texts([question.strip()], model=self.model, batch_size=1, is_query=True)[0]
        hits = search_qdrant(
            client=self.qdrant_client,
            query_vector=q_vec,
            collection_name=self.collection_name,
            limit=candidate_k,
            role_filter=role,
        )

        # Ensure active version filter
        filtered_hits: list[dict[str, Any]] = []
        for h in hits:
            chunk_data = self.chunks_by_id.get(h["chunk_id"])
            if chunk_data and chunk_data.get("version") == "1.0":
                filtered_hits.append(h)

        # Re-assign 1-based ranks
        for rank, h in enumerate(filtered_hits, start=1):
            h["rank"] = rank

        return filtered_hits

    def bm25_search(
        self,
        question: str,
        role: str,
        candidate_k: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Retrieve top candidate chunks using BM25 across corpus with role and version filtering."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(VALID_ROLES)}")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")

        # Query complete BM25 corpus to avoid missing relevant permitted chunks
        all_hits = search_bm25(
            retriever=self.bm25_retriever,
            chunk_ids=self.bm25_chunk_ids,
            query=question.strip(),
            limit=len(self.bm25_chunk_ids),
        )

        filtered_hits: list[dict[str, Any]] = []
        for h in all_hits:
            chunk_data = self.chunks_by_id.get(h["chunk_id"])
            if not chunk_data:
                continue
            # Filter: permitted role and active version
            if role in chunk_data.get("allowed_roles", []) and chunk_data.get("version") == "1.0":
                filtered_hits.append(
                    {
                        "rank": len(filtered_hits) + 1,
                        "chunk_id": h["chunk_id"],
                        "document_id": chunk_data["document_id"],
                        "section_number": chunk_data["section_number"],
                        "score": h["score"],
                        "payload": chunk_data,
                    }
                )
            if len(filtered_hits) >= candidate_k:
                break

        return filtered_hits

    def retrieve(
        self,
        question: str,
        role: str,
        top_k: int = 5,
        candidate_k: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k hybrid results using Reciprocal Rank Fusion of dense and BM25 candidates."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(VALID_ROLES)}")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        dense_candidates = self.dense_search(question, role=role, candidate_k=candidate_k)
        bm25_candidates = self.bm25_search(question, role=role, candidate_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        bm25_ranks: dict[str, int] = {}
        bm25_scores: dict[str, float] = {}

        for h in dense_candidates:
            cid = h["chunk_id"]
            dense_ranks[cid] = h["rank"]
            dense_scores[cid] = h["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_CONSTANT + h["rank"]))

        for h in bm25_candidates:
            cid = h["chunk_id"]
            bm25_ranks[cid] = h["rank"]
            bm25_scores[cid] = h["score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_CONSTANT + h["rank"]))

        # Deterministic sorting: descending by RRF score, tie-break by chunk ID
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        results: list[dict[str, Any]] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            chunk_data = self.chunks_by_id[cid]
            # Security sanity check: never return unauthorized evidence
            if role not in chunk_data.get("allowed_roles", []):
                continue

            results.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "document_id": chunk_data["document_id"],
                    "title": chunk_data["title"],
                    "section_number": chunk_data["section_number"],
                    "section_heading": chunk_data["section_heading"],
                    "version": chunk_data["version"],
                    "page_start": chunk_data["page_start"],
                    "page_end": chunk_data["page_end"],
                    "source_pdf": chunk_data["source_pdf"],
                    "allowed_roles": chunk_data["allowed_roles"],
                    "text": chunk_data["text"],
                    "rrf_score": rrf_scores[cid],
                    "dense_rank": dense_ranks.get(cid),
                    "dense_score": dense_scores.get(cid),
                    "bm25_rank": bm25_ranks.get(cid),
                    "bm25_score": bm25_scores.get(cid),
                }
            )

        return results

    def close(self) -> None:
        """Close Qdrant client connection cleanly."""
        if hasattr(self, "qdrant_client") and self.qdrant_client:
            self.qdrant_client.close()


def retrieve(
    question: str,
    role: str,
    top_k: int = 5,
    retriever: HybridRetriever | None = None,
    indexes_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience functional interface for hybrid retrieval."""
    created_retriever = False
    if retriever is None:
        retriever = HybridRetriever(indexes_dir=indexes_dir)
        created_retriever = True

    try:
        return retriever.retrieve(question=question, role=role, top_k=top_k)
    finally:
        if created_retriever:
            retriever.close()
