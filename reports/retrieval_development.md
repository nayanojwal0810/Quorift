# Retrieval Development Report — Baseline Evaluation

## 1. Dataset & Setup

- **Evaluated Questions**: Exactly 8 development questions.
- **Corpus**: Apex Crest Bank Operational Policies (Version 1.0, 20 documents, 120 chunks).
- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, CPU execution).
- **Lexical Index**: `bm25s` with tokenization across full chunk corpus.
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with constant `k=60`, candidate depth 20 per method, `top_k=5`.
- **Role Filtering**: Strict role-based permission enforcement before candidate truncation.

| Question ID | Category | Role | Required Evidence |
|---|---|---|---|
| Q-001 | direct | `general` | GEN-02 Sec 2.0 |
| Q-010 | direct | `hr` | HR-02 Sec 3.0 |
| Q-012 | exact_reference | `lending` | LND-01 Sec 4.0 |
| Q-021 | paraphrase | `finance` | FIN-01 Sec 5.0 |
| Q-022 | paraphrase | `compliance` | CMP-04 Sec 2.0 |
| Q-025 | conditional | `lending` | LND-01 Sec 2.0, LND-01 Sec 3.0 |
| Q-033 | multi_document | `finance` | FIN-01 Sec 2.0, GEN-02 Sec 4.0 |
| Q-034 | multi_document | `admin` | CMP-01 Sec 2.0, LND-04 Sec 2.0 |

## 2. Comparison Metrics (top_k=5)

| Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median latency | P95 latency |
|---|---|---|---|---|---|---|
| BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.59 ms | 4.07 ms |
| BGE dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 133.23 ms | 224.81 ms |
| RRF baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 127.47 ms | 200.71 ms |

## 3. Question-Level Results

| QID | Required Evidence | BM25 First Rank | Dense First Rank | RRF First Rank | RRF Full Evidence |
|---|---|---|---|---|---|
| Q-001 | GEN-02 Sec 2.0 | 1 | 1 | 1 | Yes |
| Q-010 | HR-02 Sec 3.0 | 1 | 1 | 1 | Yes |
| Q-012 | LND-01 Sec 4.0 | 1 | 1 | 1 | Yes |
| Q-021 | FIN-01 Sec 5.0 | 1 | 1 | 1 | Yes |
| Q-022 | CMP-04 Sec 2.0 | 1 | 1 | 1 | Yes |
| Q-025 | LND-01 Sec 2.0, LND-01 Sec 3.0 | 1 | 1 | 1 | Yes |
| Q-033 | FIN-01 Sec 2.0, GEN-02 Sec 4.0 | 1 | 1 | 1 | Yes |
| Q-034 | CMP-01 Sec 2.0, LND-04 Sec 2.0 | 1 | 1 | 1 | Yes |

## 4. Failure & Partial Evidence Analysis

All eight questions retrieved complete required evidence within the top 5 results.
## 5. Baseline Conclusions & Next Steps

- **Observed Strengths**: Baseline hybrid RRF effectively combines lexical precision with semantic matching.
- **Permission Boundary Integrity**: Role filtering completely prevents unauthorized cross-department leakage.
- **Potential Improvements to Investigate in Next Tasks**:
  1. Cross-encoder reranking (`ms-marco-MiniLM-L6-v2`) to re-score and promote multi-section evidence into the top 3–5 slots.
  2. Exploration of candidate depth and fusion weights.

> [!NOTE]
> This report represents initial baseline measurements on 8 development questions without parameter tuning.

## Development Batch 2 — Eight Additional Questions

### A. Batch Description

- **Evaluated Questions**: Exactly 8 additional development questions.
- **Question IDs**: `Q-018`, `Q-020`, `Q-024`, `Q-026`, `Q-028`, `Q-030`, `Q-035`, `Q-036`.
- **Corpus**: Apex Crest Bank Operational Policies (Version 1.0, 20 documents, 120 chunks).
- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, CPU execution, unchanged).
- **Lexical Index**: `bm25s` across full chunk corpus (unchanged).
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with constant `k=60`, candidate depth 20 per method, `top_k=5` (unchanged).
- **Role Filtering**: Strict role-based permission enforcement before candidate truncation (unchanged).

| Question ID | Category | Role | Required Evidence |
|---|---|---|---|
| Q-018 | paraphrase | `lending` | LND-02 Sec 3.0 |
| Q-020 | paraphrase | `general` | GEN-02 Sec 4.0 |
| Q-024 | paraphrase | `hr` | HR-02 Sec 4.0 |
| Q-026 | conditional | `lending` | LND-02 Sec 4.0 |
| Q-028 | conditional | `finance` | FIN-01 Sec 2.0 |
| Q-030 | conditional | `compliance` | CMP-01 Sec 5.0 |
| Q-035 | multi_document | `lending` | LND-01 Sec 4.0, LND-03 Sec 2.0, LND-03 Sec 3.0 |
| Q-036 | multi_document | `hr` | HR-01 Sec 3.0, HR-02 Sec 5.0 |

### B. Comparison Table (top_k=5)

| Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median latency | P95 latency |
|---|---|---|---|---|---|---|
| BM25 | 1.0000 | 1.0000 | 0.7812 | 1.0000 | 0.70 ms | 4.77 ms |
| BGE dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 134.55 ms | 211.55 ms |
| RRF baseline | 1.0000 | 1.0000 | 0.8542 | 1.0000 | 126.84 ms | 239.16 ms |

### C. Question-Level Evidence

| QID | Category | Role | Required Evidence | BM25 First Rank | Dense First Rank | RRF First Rank | RRF Full Evidence | Top 5 RRF Chunks |
|---|---|---|---|---|---|---|---|---|
| Q-018 | paraphrase | `lending` | LND-02 Sec 3.0 | 4 | 1 | 3 | Yes | `LND-02|1.0|2.0, LND-02|1.0|5.0, LND-02|1.0|3.0, LND-02|1.0|4.0, LND-05|1.0|5.0` |
| Q-020 | paraphrase | `general` | GEN-02 Sec 4.0 | 2 | 1 | 1 | Yes | `GEN-02|1.0|4.0, GEN-02|1.0|5.0, GEN-02|1.0|1.0, GEN-02|1.0|2.0, GEN-02|1.0|6.0` |
| Q-024 | paraphrase | `hr` | HR-02 Sec 4.0 | 2 | 1 | 2 | Yes | `HR-02|1.0|3.0, HR-02|1.0|4.0, HR-01|1.0|3.0, HR-02|1.0|5.0, HR-02|1.0|6.0` |
| Q-026 | conditional | `lending` | LND-02 Sec 4.0 | 1 | 1 | 1 | Yes | `LND-02|1.0|4.0, LND-04|1.0|4.0, LND-05|1.0|5.0, LND-02|1.0|1.0, LND-02|1.0|3.0` |
| Q-028 | conditional | `finance` | FIN-01 Sec 2.0 | 1 | 1 | 1 | Yes | `FIN-01|1.0|2.0, FIN-01|1.0|5.0, FIN-01|1.0|6.0, FIN-01|1.0|1.0, GEN-02|1.0|4.0` |
| Q-030 | conditional | `compliance` | CMP-01 Sec 5.0 | 1 | 1 | 1 | Yes | `CMP-01|1.0|5.0, CMP-03|1.0|4.0, CMP-04|1.0|2.0, CMP-01|1.0|6.0, CMP-02|1.0|5.0` |
| Q-035 | multi_document | `lending` | LND-01 Sec 4.0, LND-03 Sec 2.0, LND-03 Sec 3.0 | 1 | 1 | 1 | Yes | `LND-03|1.0|2.0, LND-03|1.0|3.0, LND-01|1.0|4.0, LND-03|1.0|1.0, LND-01|1.0|3.0` |
| Q-036 | multi_document | `hr` | HR-01 Sec 3.0, HR-02 Sec 5.0 | 1 | 1 | 1 | Yes | `HR-01|1.0|3.0, HR-02|1.0|5.0, HR-01|1.0|1.0, HR-01|1.0|2.0, HR-02|1.0|4.0` |

### D. Failure & Partial Evidence Analysis

All eight questions in Batch 2 retrieved complete required evidence within the top 5 results.
### E. Comparison with Batch 1

| Batch | Method | Hit@5 | Evidence Recall@5 | MRR@5 | Full Evidence@5 | Median Latency |
|---|---|---|---|---|---|---|
| Batch 1 | BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.59 ms |
| Batch 1 | BGE dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 133.23 ms |
| Batch 1 | RRF baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 127.47 ms |
| Batch 2 | BM25 | 1.0000 | 1.0000 | 0.7812 | 1.0000 | 0.70 ms |
| Batch 2 | BGE dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 134.55 ms |
| Batch 2 | RRF baseline | 1.0000 | 1.0000 | 0.8542 | 1.0000 | 126.84 ms |

### F. Conclusions & Next Steps

- **Observed Dynamics**: Development Batch 2 provides an independent baseline on questions requiring single, conditional, and multi-document synthesis (including 3-section retrieval in Q-035).
- **Investigation Recommendation**: Evaluate cross-encoder reranking (`ms-marco-MiniLM-L6-v2`) on multi-evidence scenarios where lexical and dense signals have varying alignment.
