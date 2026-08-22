# Benchmark Report: 集合知プログラミング (2007) vs BigQuery VECTOR_SEARCH (2026)

- **Test Users Evaluated**: 15
- **Candidate Pool**: `users_10k` (data/users_10k.jsonl)

## 1. 精度 (Precision@K) & レイテンシ比較

| Search Method                               | Precision@5   | Precision@10   | Avg Latency   | P95 Latency   |
|---------------------------------------------|---------------|----------------|---------------|---------------|
| 1. PCI 2007 (Pearson / Centered Cosine)     | 41.3%         | 40.7%          | 38.4 ms       | 46.3 ms       |
| 2. PCI 2007 (Cosine - Shared Items)         | 36.0%         | 20.0%          | 21.7 ms       | 23.5 ms       |
| 3. PCI 2007 (Cosine - Zero-Filled Sparse)   | 100.0%        | 100.0%         | 23.5 ms       | 27.5 ms       |
| 4. BQ 2026 (VECTOR_SEARCH Brute Force)      | 100.0%        | 100.0%         | 1114.7 ms     | 2159.5 ms     |
| 5. BQ 2026 (VECTOR_SEARCH IVF/TreeAH Index) | 100.0%        | 100.0%         | 1008.4 ms     | 1178.1 ms     |

## 2. 手法間の一致度 (Top-10 Jaccard Overlap)

| Method      | 1. PCI 2007 (Pearson / Centered Cosine)   | 2. PCI 2007 (Cosine - Shared Items)   | 3. PCI 2007 (Cosine - Zero-Filled Sparse)   | 4. BQ 2026 (VECTOR_SEARCH Brute Force)   | 5. BQ 2026 (VECTOR_SEARCH IVF/TreeAH Index)   |
|-------------|-------------------------------------------|---------------------------------------|---------------------------------------------|------------------------------------------|-----------------------------------------------|
| 1. PCI 2007 | 100.0%                                    | 0.0%                                  | 0.0%                                        | 0.4%                                     | 0.4%                                          |
| 2. PCI 2007 | 0.0%                                      | 100.0%                                | 0.0%                                        | 0.0%                                     | 0.0%                                          |
| 3. PCI 2007 | 0.0%                                      | 0.0%                                  | 100.0%                                      | 0.4%                                     | 0.4%                                          |
| 4. BQ 2026  | 0.4%                                      | 0.0%                                  | 0.4%                                        | 100.0%                                   | 55.7%                                         |
| 5. BQ 2026  | 0.4%                                      | 0.0%                                  | 0.4%                                        | 55.7%                                    | 100.0%                                        |
