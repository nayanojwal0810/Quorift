"""Build and verify local Qdrant and BM25 indexes for Quorift banking policies with safe staging and rollback."""

import shutil
import sys
from pathlib import Path

from quorift.config import paths
from quorift.ingestion import build_chunks, save_chunks_jsonl
from quorift.indexing import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    build_bm25_index,
    build_manifest,
    build_qdrant_index,
    embed_texts,
    get_embedding_model,
    load_bm25_index,
    load_qdrant_index,
    safe_promote_indexes,
    save_manifest,
    search_bm25,
    search_qdrant,
    verify_index_identity,
)

SMOKE_QUESTIONS = [
    {
        "id": "Smoke-Q1",
        "question": "What is the Branch Manager personal loan approval limit?",
        "expected_doc": "LND-01",
        "expected_sec": "2.0",
    },
    {
        "id": "Smoke-Q2",
        "question": "How many annual leave days do confirmed employees receive?",
        "expected_doc": "HR-01",
        "expected_sec": "2.0",
    },
    {
        "id": "Smoke-Q3",
        "question": "What is the reimbursement approval limit for a Team Lead?",
        "expected_doc": "FIN-01",
        "expected_sec": "2.0",
    },
]


def run_smoke_checks(indexes_dir: Path, embed_model) -> list[dict]:
    """Execute the three smoke check queries against the given indexes directory."""
    qdrant_dir = indexes_dir / "qdrant"
    bm25_dir = indexes_dir / "bm25"

    q_client = load_qdrant_index(qdrant_dir)
    bm25_retriever, chunk_ids = load_bm25_index(bm25_dir)

    smoke_results = []
    try:
        for test_case in SMOKE_QUESTIONS:
            qid = test_case["id"]
            q_text = test_case["question"]
            exp_doc = test_case["expected_doc"]
            exp_sec = test_case["expected_sec"]

            print(f"\n--- {qid}: \"{q_text}\" ---")
            print(f"    Target Evidence: {exp_doc} Section {exp_sec}")

            # Dense retrieval
            q_emb = embed_texts([q_text], model=embed_model, batch_size=1, is_query=True)[0]
            dense_hits = search_qdrant(q_client, query_vector=q_emb, limit=5)

            print("\n    [Dense Qdrant Results]")
            dense_found_rank = None
            for h in dense_hits:
                is_match = (h["document_id"] == exp_doc and h["section_number"] == exp_sec)
                if is_match and dense_found_rank is None:
                    dense_found_rank = h["rank"]
                mark = " <-- [TARGET EVIDENCE]" if is_match else ""
                print(f"      Rank {h['rank']}: {h['chunk_id']} (Score: {h['score']:.4f}){mark}")

            # BM25 retrieval
            bm25_hits = search_bm25(bm25_retriever, chunk_ids=chunk_ids, query=q_text, limit=5)

            print("\n    [BM25 Lexical Results]")
            bm25_found_rank = None
            for h in bm25_hits:
                is_match = (h["document_id"] == exp_doc and h["section_number"] == exp_sec)
                if is_match and bm25_found_rank is None:
                    bm25_found_rank = h["rank"]
                mark = " <-- [TARGET EVIDENCE]" if is_match else ""
                print(f"      Rank {h['rank']}: {h['chunk_id']} (Score: {h['score']:.4f}){mark}")

            # Verify top 5 smoke requirement
            if dense_found_rank is None or dense_found_rank > 5:
                raise ValueError(
                    f"Smoke check failed for {qid} ('{q_text}'): "
                    f"Target evidence {exp_doc} Section {exp_sec} was not found in top 5 dense results."
                )
            if bm25_found_rank is None or bm25_found_rank > 5:
                raise ValueError(
                    f"Smoke check failed for {qid} ('{q_text}'): "
                    f"Target evidence {exp_doc} Section {exp_sec} was not found in top 5 BM25 results."
                )

            smoke_results.append(
                {
                    "qid": qid,
                    "question": q_text,
                    "expected": f"{exp_doc} Sec {exp_sec}",
                    "dense_rank": dense_found_rank,
                    "bm25_rank": bm25_found_rank,
                    "dense_hits": dense_hits,
                    "bm25_hits": bm25_hits,
                }
            )
    finally:
        q_client.close()

    return smoke_results


def run_build_indexes(
    active_dir: Path | None = None,
    staging_dir: Path | None = None,
    backup_dir: Path | None = None,
) -> dict:
    """Build chunks and indexes into a staging directory, verify, and promote safely."""
    target_active = active_dir or paths.indexes_dir
    target_staging = staging_dir or (paths.data_dir / ".indexes_staging")
    target_backup = backup_dir or (paths.data_dir / ".indexes_backup")

    print("=" * 60)
    print("Quorift — Safe Ingestion and Indexing Pipeline")
    print("=" * 60)

    # Safe handling of unresolved backup before rebuilding
    if target_backup.exists():
        if target_active.exists():
            raise RuntimeError(
                f"Unresolved backup directory found at '{target_backup}' while active index '{target_active}' exists. "
                "An interrupted build must be resolved before proceeding. "
                "Do not automatically delete existing backups."
            )
        else:
            print(f"Active index missing; restoring from verified backup '{target_backup}'...")
            try:
                verify_index_identity(
                    indexes_dir=target_backup,
                    collection_name=DEFAULT_COLLECTION_NAME,
                    validate_registry=False,
                )
                target_backup.rename(target_active)
                print("Active index successfully restored from backup.")
            except Exception as e:
                raise RuntimeError(
                    f"Active index is missing and backup at '{target_backup}' failed structural verification: {e}. "
                    "Preserving backup and stopping."
                ) from e

    # Clean prior staging directory if interrupted
    if target_staging.exists():
        shutil.rmtree(target_staging)
    target_staging.mkdir(parents=True, exist_ok=True)

    chunks_staging_file = target_staging / "chunks.jsonl"
    qdrant_staging_dir = target_staging / "qdrant"
    bm25_staging_dir = target_staging / "bm25"
    manifest_staging_file = target_staging / "index_manifest.json"

    # 1. Ingestion & Section Chunking
    print("\n[1/6] Ingesting 20 policy PDFs and building section chunks...")
    chunks = build_chunks()
    print(f"      Extracted {len(chunks)} chunks across 20 active policy documents.")

    # 2. Token Length Validation
    print("\n[2/6] Loading BGE embedding model and validating chunk token lengths...")
    embed_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL, device="cpu")
    max_seq_len = embed_model.max_seq_length
    print(f"      Model: {DEFAULT_EMBEDDING_MODEL} (max sequence length: {max_seq_len})")

    for c in chunks:
        tokens = embed_model.tokenize([c["text"]])["input_ids"]
        token_len = tokens.shape[1] if hasattr(tokens, "shape") else len(tokens[0])
        if token_len > max_seq_len:
            raise ValueError(
                f"Chunk {c['chunk_id']} exceeds max sequence length: {token_len} > {max_seq_len}"
            )
    print("      All 120 chunks fit comfortably within the model's sequence length limit.")

    # 3. Dense Embeddings
    print("\n[3/6] Generating dense embeddings (batch_size=4, device=CPU)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model=embed_model, batch_size=4, is_query=False)
    print(f"      Generated embeddings shape: {embeddings.shape} (normalized float32 vectors)")

    # 4. Save Staged Artifacts
    print("\n[4/6] Saving staged chunk records and building Qdrant + BM25 in staging...")
    save_chunks_jsonl(chunks, chunks_staging_file)

    qdrant_client = build_qdrant_index(
        chunks=chunks,
        embeddings=embeddings,
        storage_path=qdrant_staging_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    qdrant_client.close()

    build_bm25_index(chunks=chunks, storage_path=bm25_staging_dir)

    manifest = build_manifest(
        chunks=chunks,
        model_name=DEFAULT_EMBEDDING_MODEL,
        embedding_dim=VECTOR_DIMENSION,
        collection_name=DEFAULT_COLLECTION_NAME,
        bm25_dir="bm25",
    )
    save_manifest(manifest, manifest_staging_file)
    print("      Staging complete: chunks.jsonl, Qdrant vectors, BM25, and manifest.")

    # 5. Staged Verification & Smoke Checks
    print("\n[5/6] Verifying staged index identity and running smoke checks...")
    staged_verification = verify_index_identity(
        indexes_dir=target_staging,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    assert staged_verification["total_chunks"] == 120

    print("      Running retrieval smoke checks on staged index...")
    smoke_results = run_smoke_checks(target_staging, embed_model)

    # 6. Safe Promotion
    print("\n[6/6] Promoting staged indexes to active storage with atomic rollback...")
    safe_promote_indexes(
        staging_dir=target_staging,
        active_dir=target_active,
        backup_dir=target_backup,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    # Final active verification
    final_verification = verify_index_identity(
        indexes_dir=target_active,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    print(f"      Active index verified: {final_verification['total_chunks']} chunks match across all indexes.")

    print("\n" + "=" * 60)
    print("Safe Build & Verification Completed Successfully!")
    print("=" * 60)

    return {
        "chunks_count": len(chunks),
        "total_verified_chunks": final_verification["total_chunks"],
        "smoke_results": smoke_results,
    }


def main() -> None:
    try:
        run_build_indexes()
    except Exception as e:
        print(f"\n[ERROR] Index build failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
