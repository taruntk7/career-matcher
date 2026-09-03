import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from embedding_text import load_occupations, build_embedding_text


MODEL_NAME = "all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
)


def load_model():
    """Load the sentence-transformer model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(model, occupations):
    """Generate embeddings for all occupations."""

    texts = [
        build_embedding_text(occupation)
        for occupation in occupations
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_embeddings(occupations, embeddings):
    """Save embeddings and corresponding occupation metadata."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = OUTPUT_DIR / "embeddings.npy"
    metadata_path = OUTPUT_DIR / "embedding_metadata.json"

    np.save(embeddings_path, embeddings)

    metadata = [
        {
            "code": occupation["code"],
            "title": occupation["title"],
        }
        for occupation in occupations
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME}")

    model = load_model()

    print("Loading occupations...")
    occupations = load_occupations()

    print(f"Loaded {len(occupations)} occupations")

    print("Generating embeddings...")

    embeddings = generate_embeddings(
        model,
        occupations,
    )

    print()
    print(f"Embedding shape: {embeddings.shape}")

    save_embeddings(
        occupations,
        embeddings,
    )