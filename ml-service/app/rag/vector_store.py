from pathlib import Path
import json

import numpy as np
import chromadb


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAG_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "processed"
)

RAG_CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "chroma"
)

EMBEDDINGS_PATH = (
    RAG_PROCESSED_DIR
    / "embeddings.npy"
)

METADATA_PATH = (
    RAG_PROCESSED_DIR
    / "metadata.json"
)

COLLECTION_NAME = "career_guidance"


# =========================================================
# Build vector store
# =========================================================

def build_vector_store() -> None:
    """
    Store RAG document embeddings in a persistent
    local ChromaDB collection.
    """

    # -----------------------------------------------------
    # Validate input files
    # -----------------------------------------------------

    if not EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            f"Embeddings file not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

    # -----------------------------------------------------
    # Load embeddings
    # -----------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    # -----------------------------------------------------
    # Load metadata
    # -----------------------------------------------------

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Number of embeddings does not match "
            "number of metadata records."
        )

    if len(embeddings) == 0:

        raise ValueError(
            "No embeddings were found."
        )

    # -----------------------------------------------------
    # Create persistent ChromaDB client
    # -----------------------------------------------------

    RAG_CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(
            RAG_CHROMA_DIR
        )
    )

    # -----------------------------------------------------
    # Create / load collection
    # -----------------------------------------------------

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    # -----------------------------------------------------
    # Prepare records
    # -----------------------------------------------------

    ids = []
    documents = []
    metadatas = []

    for index, item in enumerate(
        metadata
    ):

        code = str(
            item.get(
                "code",
                index,
            )
        )

        title = str(
            item.get(
                "title",
                "",
            )
        )

        source = str(
            item.get(
                "source",
                "",
            )
        )

        ids.append(
            code
        )

        # Load the actual document text.
        document_path = Path(
            source
        )

        if not document_path.exists():

            raise FileNotFoundError(
                f"RAG document not found:\n"
                f"{document_path}"
            )

        document_text = (
            document_path.read_text(
                encoding="utf-8"
            )
            .strip()
        )

        documents.append(
            document_text
        )

        metadatas.append(
            {
                "code": code,
                "title": title,
                "source": source,
            }
        )

    # -----------------------------------------------------
    # Store / update documents
    # -----------------------------------------------------

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("RAG VECTOR STORE")
    print("=" * 70)

    print()
    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Documents stored: "
        f"{collection.count()}"
    )

    print(
        f"ChromaDB path: "
        f"{RAG_CHROMA_DIR}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    build_vector_store()