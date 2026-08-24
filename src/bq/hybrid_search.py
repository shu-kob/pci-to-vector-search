"""Hybrid What x Why Search Engine.

Combines:
- 1st Party Behavioral Logs (What: Sparse Ratings Similarity from 2007 PCI)
- Deep Consumer Intent Embeddings (Why: Dense Vector Search from 2026 BigQuery)

Provides balanced recommendation and candidate reranking based on both actions and motivations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.bq.bq_search import BigQueryVectorSearchClient
from src.classic.classic_recommender import sim_cosine, sim_pearson, top_matches


class HybridSearchEngine:
    def __init__(
        self,
        bq_client: BigQueryVectorSearchClient,
        prefs: Dict[str, Dict[str, float]],
        user_metadata: Dict[str, Dict[str, Any]],
    ):
        self.bq_client = bq_client
        self.prefs = prefs
        self.user_metadata = user_metadata

    def search_hybrid(
        self,
        user_id: str,
        top_k: int = 5,
        alpha: float = 0.5,  # Weight for What (0.0 = Pure Why, 1.0 = Pure What)
        candidate_pool_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid Search combining What (behavioral ratings) and Why (semantic embeddings)."""
        if user_id not in self.user_metadata:
            return []

        target_seg = self.user_metadata[user_id]["segment_id"]

        # 1. Get Top Why candidates from BigQuery (or local cache)
        why_results, _ = self.bq_client.search_by_user_id(
            user_id=user_id,
            top_k=candidate_pool_size,
            distance_type="COSINE",
        )
        why_candidates = {r["user_id"]: r["similarity"] for r in why_results if r["similarity"] is not None}

        # 2. Get Top What candidates from Classic PCI (Zero-filled cosine)
        what_raw = top_matches(
            self.prefs,
            user_id,
            n=candidate_pool_size,
            similarity=lambda p, a, b: sim_cosine(p, a, b, zero_fill=True),
        )
        what_candidates = {u: max(0.0, score) for score, u in what_raw}

        # 3. Union candidate set
        all_candidate_ids = set(why_candidates.keys()).union(set(what_candidates.keys()))

        scored_list = []
        for cand_id in all_candidate_ids:
            if cand_id == user_id:
                continue

            # What score (0 to 1)
            what_score = what_candidates.get(
                cand_id,
                max(0.0, sim_cosine(self.prefs, user_id, cand_id, zero_fill=True))
                if cand_id in self.prefs
                else 0.0,
            )

            # Why score (0 to 1)
            why_score = why_candidates.get(cand_id, 0.0)

            # Hybrid weighted combination
            hybrid_score = (alpha * what_score) + ((1.0 - alpha) * why_score)

            meta = self.user_metadata.get(cand_id, {})
            is_same_seg = meta.get("segment_id") == target_seg

            scored_list.append(
                {
                    "user_id": cand_id,
                    "hybrid_score": round(hybrid_score, 4),
                    "what_score": round(what_score, 4),
                    "why_score": round(why_score, 4),
                    "segment_id": meta.get("segment_id", ""),
                    "segment_name": meta.get("segment_name", ""),
                    "profile_text": meta.get("profile_text", ""),
                    "interests": meta.get("interests", []),
                    "is_same_segment": is_same_seg,
                }
            )

        # Sort by hybrid score descending
        scored_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored_list[:top_k]
