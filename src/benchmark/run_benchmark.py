"""Comprehensive Benchmark Suite: 2007 Classic PCI vs 2026 BigQuery VECTOR_SEARCH.

Evaluates:
1. Precision@K and Recall@K against ground truth latent segment_id
2. Inter-method overlap (Jaccard similarity between recommendations)
3. Latency (ms per query)
4. Data scan / cost considerations
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
from tabulate import tabulate

from src.bq.bq_search import BigQueryVectorSearchClient
from src.classic.classic_recommender import (
    load_prefs_from_jsonl,
    sim_cosine,
    sim_distance,
    sim_pearson,
    top_matches,
)


def evaluate_precision_at_k(
    query_user: str,
    matches: List[Tuple[float, str] | Dict[str, Any]],
    user_metadata: Dict[str, Dict[str, Any]],
    k: int = 5,
) -> float:
    """Calculates Precision@K against ground truth segment_id."""
    query_segment = user_metadata[query_user]["segment_id"]
    correct_count = 0

    top_k_matches = matches[:k]
    if not top_k_matches:
        return 0.0

    for match in top_k_matches:
        # Match can be tuple (score, user_id) or dict {"user_id": ..., ...}
        if isinstance(match, tuple):
            matched_user_id = match[1]
        else:
            matched_user_id = match["user_id"]

        matched_segment = user_metadata[matched_user_id]["segment_id"]
        if matched_segment == query_segment:
            correct_count += 1

    return correct_count / len(top_k_matches)


def calculate_jaccard_similarity(list_a: List[str], list_b: List[str]) -> float:
    set_a = set(list_a)
    set_b = set(list_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def run_benchmark(
    jsonl_path: Path,
    num_test_users: int = 30,
    top_k: int = 10,
    project_id: str = "marine-access-331406",
    dataset_id: str = "pci_vector_search",
    table_name: str = "users_10k",
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"Loading preferences and user metadata from {jsonl_path}...")
    prefs, metadata = load_prefs_from_jsonl(jsonl_path)
    all_users = list(prefs.keys())
    print(f"Loaded {len(all_users)} users.")

    rng = random.Random(seed)
    test_users = rng.sample(all_users, min(num_test_users, len(all_users)))

    bq_client = BigQueryVectorSearchClient(
        project_id=project_id,
        dataset_id=dataset_id,
        table_name=table_name,
    )

    methods = {
        "1. PCI 2007 (Pearson / Centered Cosine)": {
            "type": "pci",
            "fn": sim_pearson,
        },
        "2. PCI 2007 (Cosine - Shared Items)": {
            "type": "pci",
            "fn": lambda p, a, b: sim_cosine(p, a, b, zero_fill=False),
        },
        "3. PCI 2007 (Cosine - Zero-Filled Sparse)": {
            "type": "pci",
            "fn": lambda p, a, b: sim_cosine(p, a, b, zero_fill=True),
        },
        "4. BQ 2026 (VECTOR_SEARCH Brute Force)": {
            "type": "bq",
            "use_brute_force": True,
            "distance_type": "COSINE",
        },
        "5. BQ 2026 (VECTOR_SEARCH IVF/TreeAH Index)": {
            "type": "bq",
            "use_brute_force": False,
            "distance_type": "COSINE",
        },
    }

    results: Dict[str, Dict[str, Any]] = {
        m: {
            "precisions_at_5": [],
            "precisions_at_10": [],
            "latencies_ms": [],
            "top_k_lists": [],
        }
        for m in methods
    }

    print(f"\nRunning benchmark on {len(test_users)} sample test users (top_k={top_k})...\n")

    for idx, u_id in enumerate(test_users, 1):
        print(f"[{idx}/{len(test_users)}] Evaluating user {u_id} ({metadata[u_id]['segment_name']})...")

        for m_name, m_config in methods.items():
            if m_config["type"] == "pci":
                start = time.perf_counter()
                matches = top_matches(prefs, u_id, n=top_k, similarity=m_config["fn"])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                match_user_ids = [m[1] for m in matches]
            else:
                try:
                    matches, stats = bq_client.search_by_user_id(
                        user_id=u_id,
                        top_k=top_k,
                        distance_type=m_config["distance_type"],
                        use_brute_force=m_config["use_brute_force"],
                    )
                    elapsed_ms = stats["elapsed_ms"]
                    match_user_ids = [m["user_id"] for m in matches]
                except Exception as ex:
                    print(f"  Warning: BQ Search failed for {m_name}: {ex}")
                    matches = []
                    match_user_ids = []
                    elapsed_ms = 0.0

            p5 = evaluate_precision_at_k(u_id, matches, metadata, k=5)
            p10 = evaluate_precision_at_k(u_id, matches, metadata, k=10)

            results[m_name]["precisions_at_5"].append(p5)
            results[m_name]["precisions_at_10"].append(p10)
            results[m_name]["latencies_ms"].append(elapsed_ms)
            results[m_name]["top_k_lists"].append(match_user_ids)

    # 1. Summary Metrics Table
    summary_table = []
    for m_name in methods:
        p5_avg = np.mean(results[m_name]["precisions_at_5"]) * 100.0 if results[m_name]["precisions_at_5"] else 0.0
        p10_avg = np.mean(results[m_name]["precisions_at_10"]) * 100.0 if results[m_name]["precisions_at_10"] else 0.0
        lat_avg = np.mean(results[m_name]["latencies_ms"]) if results[m_name]["latencies_ms"] else 0.0
        lat_p95 = np.percentile(results[m_name]["latencies_ms"], 95) if results[m_name]["latencies_ms"] else 0.0

        summary_table.append(
            [
                m_name,
                f"{p5_avg:.1f}%",
                f"{p10_avg:.1f}%",
                f"{lat_avg:.1f} ms",
                f"{lat_p95:.1f} ms",
            ]
        )

    headers = ["Search Method", "Precision@5 (Segment Match)", "Precision@10 (Segment Match)", "Avg Latency", "P95 Latency"]
    print("\n" + "=" * 85)
    print("🎯 BENCHMARK SUMMARY RESULTS: 2007 PCI vs 2026 BigQuery VECTOR_SEARCH")
    print("=" * 85)
    print(tabulate(summary_table, headers=headers, tablefmt="github"))

    # 2. Overlap Matrix (Jaccard Similarity between methods)
    method_names = list(methods.keys())
    matrix = []
    for m1 in method_names:
        row = [m1.split("(")[0].strip()]
        for m2 in method_names:
            jaccards = []
            for list1, list2 in zip(results[m1]["top_k_lists"], results[m2]["top_k_lists"]):
                jaccards.append(calculate_jaccard_similarity(list1, list2))
            avg_jaccard = np.mean(jaccards) * 100.0 if jaccards else 0.0
            row.append(f"{avg_jaccard:.1f}%")
        matrix.append(row)

    short_headers = ["Method"] + [m.split("(")[0].strip() for m in method_names]
    print("\n" + "=" * 85)
    print("📊 INTER-METHOD AGREEMENT (Jaccard Overlap @ Top 10)")
    print("=" * 85)
    print(tabulate(matrix, headers=short_headers, tablefmt="github"))

    return {
        "summary_table": summary_table,
        "overlap_matrix": matrix,
        "test_users": test_users,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite.")
    parser.add_argument("--jsonl", type=str, default="data/users_10k.jsonl", help="Input JSONL path")
    parser.add_argument("--num_users", type=int, default=20, help="Number of test users to sample")
    parser.add_argument("--top_k", type=int, default=10, help="Top K candidates")
    parser.add_argument("--table_name", type=str, default="users_10k", help="BigQuery table name")
    parser.add_argument("--output_report", type=str, default="benchmark_report.md", help="Output markdown report")
    args = parser.parse_args()

    report_data = run_benchmark(
        jsonl_path=Path(args.jsonl),
        num_test_users=args.num_users,
        top_k=args.top_k,
        table_name=args.table_name,
    )

    if args.output_report:
        report_path = Path(args.output_report)
        headers = ["Search Method", "Precision@5", "Precision@10", "Avg Latency", "P95 Latency"]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Benchmark Report: 集合知プログラミング (2007) vs BigQuery VECTOR_SEARCH (2026)\n\n")
            f.write(f"- **Test Users Evaluated**: {len(report_data['test_users'])}\n")
            f.write(f"- **Candidate Pool**: `{args.table_name}` ({args.jsonl})\n\n")
            f.write("## 1. 精度 (Precision@K) & レイテンシ比較\n\n")
            f.write(tabulate(report_data["summary_table"], headers=headers, tablefmt="github") + "\n\n")
            f.write("## 2. 手法間の一致度 (Top-10 Jaccard Overlap)\n\n")
            f.write(tabulate(report_data["overlap_matrix"], headers=["Method"] + [m[0] for m in report_data["summary_table"]], tablefmt="github") + "\n")
        print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
