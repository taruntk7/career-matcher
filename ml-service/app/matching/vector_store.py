import json
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
)

EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
METADATA_PATH = PROCESSED_DIR / "embedding_metadata.json"

CHROMA_DIR = PROCESSED_DIR / "chroma"


def build_vector_store():
    """Create and populate the O*NET ChromaDB collection."""

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata records: {len(metadata)}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="occupations",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    ids = [item["code"] for item in metadata]

    documents = [
        item["title"]
        for item in metadata
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadata,
    )

    print()
    print(f"Collection: {collection.name}")
    print(f"Documents stored: {collection.count()}")
    print(f"ChromaDB location: {CHROMA_DIR}")


if __name__ == "__main__":
    build_vector_store()