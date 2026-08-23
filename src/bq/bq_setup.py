"""BigQuery and Vertex AI Embedding Setup Pipeline.

Generates embeddings for user profile texts and loads them into BigQuery table.
Creates Vector Index and monitors index build status.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import vertexai
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


def init_gcp(project_id: str, location: str) -> Tuple[bigquery.Client, TextEmbeddingModel]:
    vertexai.init(project=project_id, location=location)
    bq_client = bigquery.Client(project=project_id, location=location)
    embed_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
    return bq_client, embed_model



def create_dataset_if_not_exists(bq_client: bigquery.Client, dataset_id: str, location: str) -> None:
    dataset_ref = bigquery.DatasetReference(bq_client.project, dataset_id)
    try:
        bq_client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset.description = "Dataset for comparing PCI (2007) and BigQuery VECTOR_SEARCH (2026)"
        bq_client.create_dataset(dataset)
        print(f"Created dataset {dataset_id} in {location}.")


def generate_embeddings_batch(
    texts: List[str],
    embed_model: TextEmbeddingModel,
    batch_size: int = 100,
) -> List[List[float]]:
    """Generates embeddings in batches."""
    all_embeddings: List[List[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = [TextEmbeddingInput(text=t, task_type="SEMANTIC_SIMILARITY") for t in batch_texts]
        # Exponential backoff retry
        for attempt in range(5):
            try:
                res = embed_model.get_embeddings(inputs)
                embeddings = [e.values for e in res]
                all_embeddings.extend(embeddings)
                print(f"Embedded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(all_embeddings)}/{total})")
                break
            except Exception as ex:
                wait_sec = 2 ** (attempt + 1)
                print(f"Error embedding batch (attempt {attempt+1}): {ex}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
        time.sleep(0.1)  # small pause to respect rate limits

    return all_embeddings


def prepare_and_load_table(
    bq_client: bigquery.Client,
    embed_model: TextEmbeddingModel,
    dataset_id: str,
    table_name: str,
    jsonl_path: Path,
    cached_embedded_jsonl: Path,
    location: str,
    force_embed: bool = False,
) -> None:
    # 1. Read input dataset
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {jsonl_path}")

    # 2. Check if embedded cache exists
    if cached_embedded_jsonl.exists() and not force_embed:
        print(f"Using cached embedded data from {cached_embedded_jsonl}")
        embedded_records = []
        with open(cached_embedded_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    embedded_records.append(json.loads(line))
    else:
        print("Generating embeddings via Vertex AI text-embedding-005 with deduplication...")
        # Get unique texts to minimize API calls and avoid redundancy
        all_texts = [r["profile_text"] for r in records]
        unique_texts = list(dict.fromkeys(all_texts))
        print(f"Found {len(unique_texts)} unique profile texts across {len(all_texts)} records.")

        unique_embeddings = generate_embeddings_batch(unique_texts, embed_model, batch_size=100)
        text_to_embedding = {t: emb for t, emb in zip(unique_texts, unique_embeddings)}

        embedded_records = []
        for r in records:
            rec_copy = dict(r)
            rec_copy["embedding"] = text_to_embedding[r["profile_text"]]
            embedded_records.append(rec_copy)

        cached_embedded_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_embedded_jsonl, "w", encoding="utf-8") as f:
            for rec in embedded_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved embedded records to cache: {cached_embedded_jsonl}")


    # 3. Define BigQuery Schema
    schema = [
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("segment_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("segment_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("age_band", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("gender", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("area", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("interests", "STRING", mode="REPEATED"),
        bigquery.SchemaField("profile_text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField(
            "ratings",
            "RECORD",
            mode="REPEATED",
            fields=[
                bigquery.SchemaField("item_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("score", "FLOAT64", mode="REQUIRED"),
            ],
        ),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]

    table_ref = f"{bq_client.project}.{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    print(f"Loading {len(embedded_records)} rows into BigQuery table {table_ref}...")
    load_job = bq_client.load_table_from_json(
        embedded_records,
        table_ref,
        job_config=job_config,
    )
    load_job.result()  # Wait for the load job to complete
    print(f"Successfully loaded data into {table_ref} ({load_job.output_rows} rows).")


def create_vector_index(
    bq_client: bigquery.Client,
    dataset_id: str,
    table_name: str,
    index_name: str = "users_vec_idx",
    distance_type: str = "COSINE",
) -> None:
    """Creates a vector index on the embedding column."""
    sql = f"""
    CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
    ON `{bq_client.project}.{dataset_id}.{table_name}`(embedding)
    OPTIONS(
      index_type = 'IVF',
      distance_type = '{distance_type}'
    );
    """
    print(f"Executing Vector Index creation:\n{sql}")
    try:
        query_job = bq_client.query(sql)
        query_job.result()
        print("Vector Index creation query completed.")
    except Exception as e:
        print(f"Vector index creation note: {e}")



def check_index_status(bq_client: bigquery.Client, dataset_id: str) -> None:
    """Queries INFORMATION_SCHEMA to check vector index coverage."""
    sql = f"""
    SELECT
      table_name,
      index_name,
      index_status,
      coverage_percentage,
      last_refresh_time,
      distance_type,
      index_type
    FROM
      `{bq_client.project}.{dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    """
    try:
        df = bq_client.query(sql).to_dataframe()
        print("\n=== Vector Index Status ===")
        if df.empty:
            print("No vector indexes found yet.")
        else:
            print(df.to_string(index=False))
    except Exception as e:
        print("Could not query VECTOR_INDEXES:", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup BigQuery dataset and Vector Index.")
    parser.add_argument("--project_id", type=str, default="marine-access-331406", help="GCP Project ID")
    parser.add_argument("--location", type=str, default="asia-northeast1", help="BigQuery & Vertex AI location")
    parser.add_argument("--dataset_id", type=str, default="pci_vector_search", help="BigQuery dataset ID")
    parser.add_argument("--table_name", type=str, default="users_10k", help="Table name")
    parser.add_argument("--input_jsonl", type=str, default="data/users_10k.jsonl", help="Input user JSONL path")
    parser.add_argument("--cache_jsonl", type=str, default="data/users_10k_embedded.jsonl", help="Embedded cache path")
    parser.add_argument("--force_embed", action="store_true", help="Force re-generation of embeddings")
    parser.add_argument("--distance_type", type=str, default="COSINE", choices=["COSINE", "EUCLIDEAN"], help="Distance metric")
    args = parser.parse_args()

    bq_client, embed_model = init_gcp(args.project_id, args.location)

    create_dataset_if_not_exists(bq_client, args.dataset_id, args.location)

    prepare_and_load_table(
        bq_client=bq_client,
        embed_model=embed_model,
        dataset_id=args.dataset_id,
        table_name=args.table_name,
        jsonl_path=Path(args.input_jsonl),
        cached_embedded_jsonl=Path(args.cache_jsonl),
        location=args.location,
        force_embed=args.force_embed,
    )

    create_vector_index(
        bq_client=bq_client,
        dataset_id=args.dataset_id,
        table_name=args.table_name,
        index_name=f"{args.table_name}_{args.distance_type.lower()}_idx",
        distance_type=args.distance_type,
    )

    check_index_status(bq_client, args.dataset_id)


if __name__ == "__main__":
    from typing import Tuple
    main()
