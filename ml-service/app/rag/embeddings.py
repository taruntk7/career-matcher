from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAG_DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "documents"
)

RAG_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "processed"
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# Load model
# =========================================================

def load_model() -> SentenceTransformer:
    """
    Load the Sentence Transformer model used for
    RAG document embeddings.
    """

    return SentenceTransformer(
        MODEL_NAME
    )


# =========================================================
# Load documents
# =========================================================

def load_documents() -> tuple[list[str], list[dict]]:
    """
    Load all RAG text documents.

    Returns:
        texts
        metadata
    """

    if not RAG_DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"RAG document directory not found:\n"
            f"{RAG_DOCUMENTS_DIR}"
        )

    texts = []
    metadata = []

    document_paths = sorted(
        RAG_DOCUMENTS_DIR.glob("*.txt")
    )

    if not document_paths:

        raise RuntimeError(
            "No RAG documents were found."
        )

    for path in document_paths:

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        # The filename is based on the O*NET code.
        code = path.stem.replace(
            "_",
            ".",
        )

        # Extract title from the first line:
        # "Occupation: Data Scientists"
        first_line = text.split(
            "\n",
            1
        )[0].strip()

        if first_line.startswith(
            "Occupation:"
        ):

            title = first_line[
                len("Occupation:"):
            ].strip()

        else:

            title = first_line

        texts.append(
            text
        )

        metadata.append(
            {
                "code": code,
                "title": title,
                "source": str(path),
            }
        )

    return texts, metadata


# =========================================================
# Generate embeddings
# =========================================================

def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
):
    """
    Generate normalized embeddings for the RAG documents.
    """

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


# =========================================================
# Save embeddings
# =========================================================

def save_embeddings(
    embeddings,
    metadata: list[dict],
) -> None:
    """
    Save RAG embeddings and metadata.
    """

    import numpy as np

    RAG_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        RAG_PROCESSED_DIR
        / "embeddings.npy"
    )

    metadata_path = (
        RAG_PROCESSED_DIR
        / "metadata.json"
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved embeddings to: "
        f"{embeddings_path}"
    )

    print(
        f"Saved metadata to: "
        f"{metadata_path}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 70)
    print("RAG EMBEDDING GENERATION")
    print("=" * 70)

    print()
    print(
        f"Loading documents from:\n"
        f"{RAG_DOCUMENTS_DIR}"
    )

    texts, metadata = (
        load_documents()
    )

    print(
        f"Loaded {len(texts)} documents."
    )

    print()
    print(
        f"Loading model: "
        f"{MODEL_NAME}"
    )

    model = load_model()

    print()
    print(
        "Generating embeddings..."
    )

    embeddings = generate_embeddings(
        model,
        texts,
    )

    print()
    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    save_embeddings(
        embeddings,
        metadata,
    )

    print()
    print(
        "RAG embedding generation complete."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()