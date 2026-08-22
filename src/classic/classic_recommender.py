"""Classic 2007 Collaborative Filtering Engine based on Programming Collective Intelligence (PCI) by Toby Segaran.

Implements:
1. sim_distance (Euclidean distance metric)
2. sim_pearson (Pearson correlation coefficient / centered cosine)
3. sim_cosine (Cosine similarity over shared items or zero-filled item space)
4. top_matches (Find top N similar users)
5. get_recommendations (User-based collaborative filtering item recommendations)
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def sim_distance(prefs: Dict[str, Dict[str, float]], person1: str, person2: str) -> float:
    """Returns a distance-based similarity score for person1 and person2 (Euclidean)."""
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # If they have no ratings in common, return 0
    if len(si) == 0:
        return 0.0

    # Add up the squares of all the differences
    sum_of_squares = sum(
        pow(prefs[person1][item] - prefs[person2][item], 2)
        for item in prefs[person1]
        if item in prefs[person2]
    )

    return 1.0 / (1.0 + sqrt(sum_of_squares))


def sim_pearson(prefs: Dict[str, Dict[str, float]], p1: str, p2: str) -> float:
    """Returns the Pearson correlation coefficient for p1 and p2.

    Equivalent to Centered Cosine Similarity over shared items.
    """
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # Find the number of elements
    n = len(si)

    # If they are no ratings in common, return 0
    if n == 0:
        return 0.0

    # Add up all the preferences
    sum1 = sum(prefs[p1][it] for it in si)
    sum2 = sum(prefs[p2][it] for it in si)

    # Sum up the squares
    sum1_sq = sum(pow(prefs[p1][it], 2) for it in si)
    sum2_sq = sum(pow(prefs[p2][it], 2) for it in si)

    # Sum up the products
    p_sum = sum(prefs[p1][it] * prefs[p2][it] for it in si)

    # Calculate Pearson score
    num = p_sum - (sum1 * sum2 / n)
    den = sqrt((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n))
    if den == 0:
        return 0.0

    return float(num / den)


def sim_cosine(
    prefs: Dict[str, Dict[str, float]],
    p1: str,
    p2: str,
    zero_fill: bool = False,
) -> float:
    """Returns Cosine similarity between p1 and p2.

    If zero_fill is False (Standard PCI book interpretation):
      Computes dot product and norms ONLY over shared items.
    If zero_fill is True (Sparse vector cosine interpretation):
      Computes dot product over shared items, but norms over ALL rated items for each user.
    """
    shared = [it for it in prefs[p1] if it in prefs[p2]]
    if not shared:
        return 0.0

    dot = sum(prefs[p1][it] * prefs[p2][it] for it in shared)

    if not zero_fill:
        # Standard book-like shared items calculation
        n1 = sqrt(sum(pow(prefs[p1][it], 2) for it in shared))
        n2 = sqrt(sum(pow(prefs[p2][it], 2) for it in shared))
    else:
        # True sparse vector cosine (unrated items = 0)
        n1 = sqrt(sum(pow(prefs[p1][it], 2) for it in prefs[p1]))
        n2 = sqrt(sum(pow(prefs[p2][it], 2) for it in prefs[p2]))

    if n1 * n2 == 0:
        return 0.0

    return float(dot / (n1 * n2))


def top_matches(
    prefs: Dict[str, Dict[str, float]],
    person: str,
    n: int = 5,
    similarity: Callable[[Dict[str, Dict[str, float]], str, str], float] = sim_pearson,
) -> List[Tuple[float, str]]:
    """Returns the best matches for person from the prefs dictionary.

    Number of results and similarity function are optional params.
    """
    scores = [
        (similarity(prefs, person, other), other)
        for other in prefs
        if other != person
    ]

    # Sort the list so the highest scores appear at the top
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0:n]


def get_recommendations(
    prefs: Dict[str, Dict[str, float]],
    person: str,
    similarity: Callable[[Dict[str, Dict[str, float]], str, str], float] = sim_pearson,
) -> List[Tuple[float, str]]:
    """Gets recommendations for a person by using a weighted average of every other user's rankings."""
    totals: Dict[str, float] = {}
    sim_sums: Dict[str, float] = {}

    for other in prefs:
        # don't compare me to myself
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue

        for item in prefs[other]:
            # only score items I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0.0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                sim_sums.setdefault(item, 0.0)
                sim_sums[item] += sim

    # Create the normalized list
    rankings = [
        (total / sim_sums[item], item)
        for item, total in totals.items()
        if sim_sums[item] > 0
    ]

    # Return the sorted list
    rankings.sort(key=lambda x: x[0], reverse=True)
    return rankings


def load_prefs_from_jsonl(jsonl_path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    """Loads ratings matrix and user metadata from a generated JSONL file."""
    prefs: Dict[str, Dict[str, float]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            u_id = record["user_id"]
            user_ratings = {r["item_id"]: float(r["score"]) for r in record["ratings"]}
            prefs[u_id] = user_ratings
            metadata[u_id] = {
                "segment_id": record["segment_id"],
                "segment_name": record.get("segment_name", ""),
                "profile_text": record.get("profile_text", ""),
                "age_band": record.get("age_band", ""),
                "gender": record.get("gender", ""),
                "area": record.get("area", ""),
                "interests": record.get("interests", []),
                "ratings": record.get("ratings", []),
            }

    return prefs, metadata


if __name__ == "__main__":
    import sys

    data_file = Path("data/users_1k.jsonl")
    if not data_file.exists():
        print("Run generate_dataset.py first!")
        sys.exit(1)

    prefs, meta = load_prefs_from_jsonl(data_file)
    sample_user = "usr_000001"
    print(f"Sample User: {sample_user} ({meta[sample_user]['segment_name']})")
    print(f"Profile: {meta[sample_user]['profile_text']}")
    print(f"Rated items: {len(prefs[sample_user])}")

    print("\n--- Top Matches (Pearson) ---")
    matches_pearson = top_matches(prefs, sample_user, n=5, similarity=sim_pearson)
    for score, u in matches_pearson:
        print(f"User: {u} (Score: {score:.4f}, Segment: {meta[u]['segment_name']})")

    print("\n--- Top Matches (Cosine Shared) ---")
    matches_cos = top_matches(
        prefs, sample_user, n=5, similarity=lambda p, a, b: sim_cosine(p, a, b, zero_fill=False)
    )
    for score, u in matches_cos:
        print(f"User: {u} (Score: {score:.4f}, Segment: {meta[u]['segment_name']})")

    print("\n--- Top Matches (Cosine Zero-Filled) ---")
    matches_cos_zf = top_matches(
        prefs, sample_user, n=5, similarity=lambda p, a, b: sim_cosine(p, a, b, zero_fill=True)
    )
    for score, u in matches_cos_zf:
        print(f"User: {u} (Score: {score:.4f}, Segment: {meta[u]['segment_name']})")
