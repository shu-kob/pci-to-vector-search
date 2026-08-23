"""Web Application Backend for Side-by-Side Comparison: 2007 PCI vs 2026 BigQuery VECTOR_SEARCH.

FastAPI server providing endpoints for user inspection and real-time comparison.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.bq.bq_search import BigQueryVectorSearchClient
from src.classic.classic_recommender import (
    get_recommendations,
    load_prefs_from_jsonl,
    sim_cosine,
    sim_pearson,
    top_matches,
)

app = FastAPI(title="PCI to Vector Search: Two Eras Comparison")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global data store (in-memory for classic recommender)
DATA_FILE = Path("data/users_10k.jsonl")
if not DATA_FILE.exists():
    DATA_FILE = Path("data/users_1k.jsonl")

prefs: Dict[str, Dict[str, float]] = {}
user_metadata: Dict[str, Dict[str, Any]] = {}

if DATA_FILE.exists():
    print(f"Loading user data from {DATA_FILE}...")
    prefs, user_metadata = load_prefs_from_jsonl(DATA_FILE)
    print(f"Loaded {len(prefs)} users for Classic PCI engine.")

# BigQuery client
bq_client = BigQueryVectorSearchClient(
    project_id="YOUR_PROJECT_ID",
    dataset_id="pci_vector_search",
    table_name="users_10k" if Path("data/users_10k.jsonl").exists() else "users_1k",
)


class CompareRequest(BaseModel):
    user_id: str
    top_k: int = 5
    classic_method: str = "pearson"  # 'pearson', 'cosine_shared', 'cosine_zero'
    use_brute_force: bool = False


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Pass 20 sample users from different segments
    sample_users = []
    seen_segments = set()
    for uid, meta in user_metadata.items():
        seg = meta["segment_id"]
        if seg not in seen_segments or len(sample_users) < 15:
            sample_users.append(
                {
                    "user_id": uid,
                    "segment_name": meta["segment_name"],
                    "age_band": meta["age_band"],
                    "gender": meta["gender"],
                    "area": meta["area"],
                    "profile_snippet": (meta["profile_text"][:60] + "...") if len(meta["profile_text"]) > 60 else meta["profile_text"],
                }
            )
            seen_segments.add(seg)
        if len(sample_users) >= 20:
            break

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "sample_users": sample_users,
            "total_users": len(user_metadata),
            "table_name": bq_client.table_name,
        },
    )


@app.get("/api/user/{user_id}")
async def get_user_detail(user_id: str):
    if user_id not in user_metadata:
        raise HTTPException(status_code=404, detail="User not found")
    meta = user_metadata[user_id]
    return {
        "user_id": user_id,
        "segment_id": meta["segment_id"],
        "segment_name": meta["segment_name"],
        "age_band": meta["age_band"],
        "gender": meta["gender"],
        "area": meta["area"],
        "interests": meta["interests"],
        "profile_text": meta["profile_text"],
        "ratings": meta["ratings"],
    }


@app.post("/api/compare")
async def compare_recommendations(req: CompareRequest):
    user_id = req.user_id
    if user_id not in user_metadata:
        raise HTTPException(status_code=404, detail="User not found")

    target_user = user_metadata[user_id]
    target_segment = target_user["segment_id"]

    # 1. Run 2007 Classic PCI
    sim_fn = sim_pearson
    if req.classic_method == "cosine_shared":
        sim_fn = lambda p, a, b: sim_cosine(p, a, b, zero_fill=False)
    elif req.classic_method == "cosine_zero":
        sim_fn = lambda p, a, b: sim_cosine(p, a, b, zero_fill=True)

    t0 = time.perf_counter()
    classic_raw_matches = top_matches(prefs, user_id, n=req.top_k, similarity=sim_fn)
    classic_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Also compute item recommendations
    classic_item_recs = get_recommendations(prefs, user_id, similarity=sim_fn)[:3]

    classic_matches = []
    classic_correct = 0
    for score, m_uid in classic_raw_matches:
        m_meta = user_metadata.get(m_uid, {})
        is_same_seg = m_meta.get("segment_id") == target_segment
        if is_same_seg:
            classic_correct += 1

        # Find shared items
        target_items = prefs[user_id]
        m_items = prefs[m_uid]
        shared = [
            {"item_id": it, "user_score": target_items[it], "match_score": m_items[it]}
            for it in target_items
            if it in m_items
        ]

        classic_matches.append(
            {
                "user_id": m_uid,
                "score": round(score, 4),
                "segment_id": m_meta.get("segment_id", ""),
                "segment_name": m_meta.get("segment_name", ""),
                "profile_text": m_meta.get("profile_text", ""),
                "is_same_segment": is_same_seg,
                "shared_items_count": len(shared),
                "shared_items": shared,
            }
        )

    classic_precision = round((classic_correct / len(classic_matches) * 100.0) if classic_matches else 0.0, 1)

    # 2. Run 2026 BigQuery VECTOR_SEARCH
    bq_matches = []
    bq_correct = 0
    bq_stats = {}
    try:
        raw_bq_results, bq_stats = bq_client.search_by_user_id(
            user_id=user_id,
            top_k=req.top_k,
            distance_type="COSINE",
            use_brute_force=req.use_brute_force,
        )
        for item in raw_bq_results:
            is_same_seg = item["segment_id"] == target_segment
            if is_same_seg:
                bq_correct += 1
            bq_matches.append(
                {
                    "user_id": item["user_id"],
                    "similarity": round(item["similarity"], 4) if item["similarity"] is not None else 0.0,
                    "distance": round(item["distance"], 4),
                    "segment_id": item["segment_id"],
                    "segment_name": item["segment_name"],
                    "profile_text": item["profile_text"],
                    "interests": item["interests"],
                    "is_same_segment": is_same_seg,
                }
            )
    except Exception as ex:
        print(f"BQ Search error in web API: {ex}")
        bq_stats = {"elapsed_ms": 0.0, "bytes_processed": 0, "error": str(ex)}

    bq_precision = round((bq_correct / len(bq_matches) * 100.0) if bq_matches else 0.0, 1)

    # 3. Inter-method overlap (Jaccard)
    classic_uids = set(m["user_id"] for m in classic_matches)
    bq_uids = set(m["user_id"] for m in bq_matches)
    overlap_count = len(classic_uids.intersection(bq_uids))
    jaccard_similarity = round(
        (overlap_count / len(classic_uids.union(bq_uids)) * 100.0)
        if classic_uids.union(bq_uids)
        else 0.0,
        1,
    )

    return {
        "target_user": {
            "user_id": user_id,
            "segment_id": target_segment,
            "segment_name": target_user["segment_name"],
            "profile_text": target_user["profile_text"],
            "ratings_count": len(prefs[user_id]),
            "interests": target_user["interests"],
        },
        "classic": {
            "method_name": "集合知プログラミング (2007)",
            "method_variant": req.classic_method,
            "elapsed_ms": round(classic_elapsed_ms, 2),
            "precision": classic_precision,
            "matches": classic_matches,
            "item_recommendations": [
                {"score": round(s, 2), "item_id": it} for s, it in classic_item_recs
            ],
        },
        "modern": {
            "method_name": "BigQuery VECTOR_SEARCH (2026)",
            "elapsed_ms": bq_stats.get("elapsed_ms", 0.0),
            "bytes_processed": bq_stats.get("bytes_processed", 0),
            "precision": bq_precision,
            "use_brute_force": req.use_brute_force,
            "matches": bq_matches,
        },
        "comparison": {
            "overlap_count": overlap_count,
            "jaccard_similarity": jaccard_similarity,
        },
    }
