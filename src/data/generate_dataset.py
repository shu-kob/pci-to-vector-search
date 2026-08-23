"""Synthetic User Dataset Generator for PCI vs Vector Search Benchmark.

Generates realistic user profiles with:
1. Ground truth latent segment_id (used strictly for evaluation/benchmarking).
2. Sparse ratings matrix for 2007 Classic Collaborative Filtering.
3. Natural text profile for 2026 Modern Vector Embeddings.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_segments(yaml_path: Path) -> List[Dict[str, Any]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["segments"]


def build_generic_item_pool() -> List[Dict[str, Any]]:
    """Build a background pool of generic items that any user might occasionally rate."""
    categories = [
        ("movie", 20),
        ("book", 20),
        ("electronics", 20),
        ("food", 20),
        ("apparel", 20),
    ]
    pool = []
    for cat, count in categories:
        for i in range(1, count + 1):
            pool.append(
                {
                    "item_id": f"item_gen_{cat}_{i:03d}",
                    "category": cat,
                }
            )
    return pool


def generate_profile_text(segment: Dict[str, Any], rng: random.Random) -> str:
    template = rng.choice(segment["profile_templates"])
    slots = segment["slots"]
    filled = template
    for key, values in slots.items():
        placeholder = f"{{{key}}}"
        if placeholder in filled:
            val = rng.choice(values)
            filled = filled.replace(placeholder, val)
    return filled


def generate_user_record(
    user_idx: int,
    segments: List[Dict[str, Any]],
    generic_items: List[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    user_id = f"usr_{user_idx:06d}"

    # 1. Assign ground truth segment
    segment = rng.choice(segments)
    segment_id = segment["segment_id"]

    # 2. Demographics & Profile text
    age_band = rng.choice(segment["demographics"]["age_bands"])
    gender = rng.choice(segment["demographics"]["genders"])
    area = rng.choice(segment["demographics"]["areas"])

    num_interests = min(len(segment["interests"]), rng.randint(3, 5))
    interests = rng.sample(segment["interests"], num_interests)

    profile_text = generate_profile_text(segment, rng)

    # 3. Generate Sparse Item Ratings (for 2007 PCI algorithm)
    # Ratings are on a 1.0 - 5.0 scale (floats rounded to 1 decimal)
    ratings: List[Dict[str, Any]] = []
    rated_items_set = set()

    # Preferred items (high probability of rating, high score)
    for p_item in segment["preferred_items"]:
        if rng.random() < 0.85:  # 85% chance to rate preferred items
            score = round(
                min(5.0, max(1.0, rng.gauss(p_item["base_rating"], 0.3))), 1
            )
            ratings.append({"item_id": p_item["item_id"], "score": score})
            rated_items_set.add(p_item["item_id"])

    # Disliked items (50% chance to rate, low score)
    for d_item in segment["disliked_items"]:
        if rng.random() < 0.50:
            score = round(
                min(5.0, max(1.0, rng.gauss(d_item["base_rating"], 0.4))), 1
            )
            ratings.append({"item_id": d_item["item_id"], "score": score})
            rated_items_set.add(d_item["item_id"])

    # Generic background noise items (2 - 6 random items rated neutrally or randomly)
    num_noise = rng.randint(2, 6)
    noise_items = rng.sample(generic_items, num_noise)
    for n_item in noise_items:
        if n_item["item_id"] not in rated_items_set:
            score = round(rng.uniform(1.5, 4.5), 1)
            ratings.append({"item_id": n_item["item_id"], "score": score})
            rated_items_set.add(n_item["item_id"])

    # Shuffle ratings order
    rng.shuffle(ratings)

    return {
        "user_id": user_id,
        "segment_id": segment_id,
        "segment_name": segment["name"],
        "age_band": age_band,
        "gender": gender,
        "area": area,
        "interests": interests,
        "profile_text": profile_text,
        "ratings": ratings,
    }


def generate_dataset(
    n_samples: int,
    yaml_path: Path,
    output_path: Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    segments = load_segments(yaml_path)
    generic_items = build_generic_item_pool()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_samples} user records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(1, n_samples + 1):
            record = generate_user_record(idx, segments, generic_items, rng)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Successfully generated {n_samples} records.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic user dataset.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of records to generate (e.g. 1000, 10000, 100000)",
    )
    parser.add_argument(
        "--segments_path",
        type=str,
        default="src/data/segments.yaml",
        help="Path to segments YAML configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/users_10k.jsonl",
        help="Output JSONL filepath",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    yaml_path = Path(args.segments_path)
    output_path = Path(args.output)
    generate_dataset(args.n_samples, yaml_path, output_path, args.seed)


if __name__ == "__main__":
    main()
