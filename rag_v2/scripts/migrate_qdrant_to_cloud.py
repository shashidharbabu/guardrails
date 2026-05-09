#!/usr/bin/env python3
"""
Copy the local embedded NewRAG v2 Qdrant collection into Qdrant Cloud.

Dense vectors live in Qdrant Cloud after this migration. BM25 stays local in
rag_v2/bm25_combined.pkl and is combined with dense results by retrieval_v2.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, PointStruct


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local Qdrant points to Qdrant Cloud")
    parser.add_argument(
        "--local-path",
        default=os.environ.get(
            "RAG_V2_QDRANT_PATH",
            str(RAG_ROOT / "indexes_qdrant_data"),
        ),
        help="Local embedded Qdrant storage directory",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("RAG_V2_COLLECTION_NAME", "guardrails_rag_v2"),
        help="Collection name to copy",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the cloud collection first if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    timeout = int(os.environ.get("QDRANT_TIMEOUT", "120"))

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env before migrating")

    local_client = QdrantClient(path=args.local_path)
    cloud_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=timeout)

    local_info = local_client.get_collection(args.collection)
    vectors_config = local_info.config.params.vectors
    sparse_vectors_config = getattr(local_info.config.params, "sparse_vectors", None)

    cloud_exists = cloud_client.collection_exists(args.collection)
    if cloud_exists and args.recreate:
        print(f"Deleting existing cloud collection: {args.collection}")
        cloud_client.delete_collection(args.collection)
        cloud_exists = False

    if not cloud_exists:
        print(f"Creating cloud collection: {args.collection}")
        cloud_client.create_collection(
            collection_name=args.collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
    else:
        print(f"Using existing cloud collection: {args.collection}")

    total = 0
    offset = None
    while True:
        points, offset = local_client.scroll(
            collection_name=args.collection,
            limit=args.batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if points:
            cloud_client.upsert(
                collection_name=args.collection,
                points=[
                    PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                    for point in points
                ],
                wait=True,
            )
            total += len(points)
            print(f"Migrated {total}/{local_info.points_count} points")

        if offset is None:
            break

    for field_name, schema in (
        ("is_summary", PayloadSchemaType.BOOL),
        ("domain", PayloadSchemaType.KEYWORD),
        ("source_file", PayloadSchemaType.KEYWORD),
    ):
        try:
            cloud_client.create_payload_index(
                collection_name=args.collection,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception as exc:
            print(f"Payload index {field_name} skipped: {exc}")

    cloud_info = cloud_client.get_collection(args.collection)
    print(
        f"Done. Cloud collection {args.collection} has "
        f"{cloud_info.points_count} points."
    )

    local_client.close()
    cloud_client.close()


if __name__ == "__main__":
    main()
