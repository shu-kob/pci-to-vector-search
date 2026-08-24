"""BigQuery VECTOR_SEARCH Client Module.

Executes vector searches on user embeddings using:
- Exact search (brute force) vs Approximate search (IVF/TreeAH index)
- Distance types: COSINE, EUCLIDEAN
- Queries by existing user_id or arbitrary query embedding vector
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import bigquery


class BigQueryVectorSearchClient:
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "pci_vector_search",
        table_name: str = "users_10k",
        location: str = "asia-northeast1",
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.location = location
        self.client = bigquery.Client(project=self.project_id, location=location)


    def search_by_user_id(
        self,
        user_id: str,
        top_k: int = 10,
        distance_type: str = "COSINE",
        use_brute_force: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search similar users for a given user_id."""
        bf_str = "true" if use_brute_force else "false"

        sql = f"""
        SELECT
          query.user_id AS query_user_id,
          base.user_id AS matched_user_id,
          base.segment_id AS matched_segment_id,
          base.segment_name AS matched_segment_name,
          base.age_band AS matched_age_band,
          base.gender AS matched_gender,
          base.area AS matched_area,
          base.profile_text AS matched_profile_text,
          base.interests AS matched_interests,
          base.ratings AS matched_ratings,
          distance
        FROM
          VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_id}.{self.table_name}`,
            'embedding',
            (
              SELECT user_id, embedding
              FROM `{self.project_id}.{self.dataset_id}.{self.table_name}`
              WHERE user_id = @user_id
            ),
            top_k => @top_k,
            distance_type => '{distance_type}',
            options => '{{"use_brute_force": {bf_str}}}'
          )
        WHERE base.user_id != query.user_id
        ORDER BY distance ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k + 1),  # +1 to exclude self
            ]
        )

        start_time = time.perf_counter()
        query_job = self.client.query(sql, job_config=job_config)
        rows = list(query_job.result())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "elapsed_ms": round(elapsed_ms, 2),
            "bytes_processed": query_job.total_bytes_processed or 0,
            "bytes_billed": query_job.total_bytes_billed or 0,
            "slot_millis": query_job.slot_millis or 0,
            "use_brute_force": use_brute_force,
            "distance_type": distance_type,
        }

        results = []
        for r in rows:
            # Cosine similarity conversion if COSINE: similarity = 1 - distance
            # distance for cosine in BQ VECTOR_SEARCH is cosine distance (1 - cos_sim)
            cos_sim = 1.0 - float(r["distance"]) if distance_type == "COSINE" else None
            results.append(
                {
                    "user_id": r["matched_user_id"],
                    "segment_id": r["matched_segment_id"],
                    "segment_name": r["matched_segment_name"],
                    "age_band": r["matched_age_band"],
                    "gender": r["matched_gender"],
                    "area": r["matched_area"],
                    "profile_text": r["matched_profile_text"],
                    "interests": list(r["matched_interests"]) if r["matched_interests"] else [],
                    "distance": float(r["distance"]),
                    "similarity": cos_sim,
                }
            )

        return results[:top_k], stats

    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 10,
        distance_type: str = "COSINE",
        use_brute_force: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Search similar users given an arbitrary query embedding vector."""
        bf_str = "true" if use_brute_force else "false"

        sql = f"""
        SELECT
          base.user_id AS matched_user_id,
          base.segment_id AS matched_segment_id,
          base.segment_name AS matched_segment_name,
          base.age_band AS matched_age_band,
          base.gender AS matched_gender,
          base.area AS matched_area,
          base.profile_text AS matched_profile_text,
          base.interests AS matched_interests,
          distance
        FROM
          VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_id}.{self.table_name}`,
            'embedding',
            (SELECT @embedding AS embedding),
            top_k => @top_k,
            distance_type => '{distance_type}',
            options => '{{"use_brute_force": {bf_str}}}'
          )
        ORDER BY distance ASC
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", embedding),
                bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            ]
        )

        start_time = time.perf_counter()
        query_job = self.client.query(sql, job_config=job_config)
        rows = list(query_job.result())
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "elapsed_ms": round(elapsed_ms, 2),
            "bytes_processed": query_job.total_bytes_processed or 0,
            "bytes_billed": query_job.total_bytes_billed or 0,
            "slot_millis": query_job.slot_millis or 0,
            "use_brute_force": use_brute_force,
            "distance_type": distance_type,
        }

        results = []
        for r in rows:
            cos_sim = 1.0 - float(r["distance"]) if distance_type == "COSINE" else None
            results.append(
                {
                    "user_id": r["matched_user_id"],
                    "segment_id": r["matched_segment_id"],
                    "segment_name": r["matched_segment_name"],
                    "age_band": r["matched_age_band"],
                    "gender": r["matched_gender"],
                    "area": r["matched_area"],
                    "profile_text": r["matched_profile_text"],
                    "interests": list(r["matched_interests"]) if r["matched_interests"] else [],
                    "distance": float(r["distance"]),
                    "similarity": cos_sim,
                }
            )

        return results, stats
