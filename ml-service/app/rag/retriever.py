import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
)

RAG_CHROMA_DIR = PROJECT_ROOT / "data" / "rag" / "chroma"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "career_guidance"


# --------------------------------------------------
# Load retriever
# --------------------------------------------------

def load_retriever():
    """
    Load the Sentence Transformer model and
    persistent RAG ChromaDB collection.
    """

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(RAG_CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return model, collection


# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def retrieve_documents(
    model,
    collection,
    query,
    top_k=3,
):
    """
    Retrieve semantically relevant documents.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    documents = []

    result_ids = results.get("ids", [[]])[0]
    result_docs = results.get("documents", [[]])[0]
    result_metadatas = results.get("metadatas", [[]])[0]
    result_distances = results.get("distances", [[]])[0]

    for i in range(len(result_ids)):

        distance = result_distances[i]

        documents.append(
            {
                "id": result_ids[i],
                "document": result_docs[i],
                "metadata": result_metadatas[i],
                "distance": distance,
                "similarity": 1.0 - distance,
            }
        )

    return documents


# --------------------------------------------------
# Exact occupation retrieval
# --------------------------------------------------

def retrieve_occupation(
    collection,
    occupation_code=None,
    occupation_title=None,
):
    """
    Retrieve the exact occupation document from ChromaDB.

    The preferred identifier is the O*NET occupation code.
    Title matching is provided as a fallback.
    """

    if occupation_code:

        results = collection.get(
            where={
                "code": occupation_code
            }
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])

        if documents:
            return {
                "id": ids[0],
                "document": documents[0],
                "metadata": metadatas[0],
                "similarity": 1.0,
            }

    if occupation_title:

        results = collection.get(
            where={
                "title": occupation_title
            }
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])

        if documents:
            return {
                "id": ids[0],
                "document": documents[0],
                "metadata": metadatas[0],
                "similarity": 1.0,
            }

    return None


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    model, collection = load_retriever()

    occupation = retrieve_occupation(
        collection=collection,
        occupation_title="Data Scientists",
    )

    print("\n" + "=" * 60)
    print("EXACT OCCUPATION RETRIEVAL")
    print("=" * 60)

    if occupation:

        print(f"Title: {occupation['metadata'].get('title')}")
        print(f"Code: {occupation['metadata'].get('code')}")
        print(f"Similarity: {occupation['similarity']:.3f}")

        print("\nDocument preview:")
        print(occupation["document"][:1500])

    else:
        print("Occupation not found.")