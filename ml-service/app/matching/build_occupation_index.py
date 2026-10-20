"""Build the ChromaDB occupation index used by matching.matcher.

Run from the repository root:
    python ml-service/app/matching/build_occupation_index.py

The script reads data/onet/processed/occupations.json and creates the
persistent ChromaDB collection at data/onet/processed/chroma.
"""

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("career-matcher.index-builder")

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "occupations"
BATCH_SIZE = 64

# This file lives at ml-service/app/matching/build_occupation_index.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_PATH = PROJECT_ROOT / "data" / "onet" / "processed" / "occupations.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "onet" / "processed" / "chroma"


def _item_names(items: Any, limit: int) -> list[str]:
    """Convert an O*NET list (strings or dictionaries) to readable labels."""
    if not isinstance(items, list):
        return []

    values: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("skill") or "").strip()
        else:
            name = str(item).strip()
        if name and name not in values:
            values.append(name)
        if len(values) >= limit:
            break
    return values


def _occupation_text(occupation: dict[str, Any]) -> str:
    """Build text used for embeddings; keep the stored Chroma document as title."""
    title = str(occupation.get("title") or "Unknown occupation").strip()
    description = str(occupation.get("description") or "").strip()
    tasks = _item_names(occupation.get("tasks"), 8)
    essential = _item_names(occupation.get("essential_skills"), 12)
    transferable = _item_names(occupation.get("transferable_skills"), 12)
    software = _item_names(occupation.get("software_skills"), 20)

    sections = [f"Occupation: {title}"]
    if description:
        sections.append(f"Description: {description}")
    if tasks:
        sections.append("Tasks: " + "; ".join(tasks))
    if essential:
        sections.append("Essential skills: " + "; ".join(essential))
    if transferable:
        sections.append("Transferable skills: " + "; ".join(transferable))
    if software:
        sections.append("Software and technologies: " + "; ".join(software))
    return "\n".join(sections)


def _collection_name(collection: Any) -> str:
    return collection if isinstance(collection, str) else collection.name


def main() -> None:
    logger.info("INDEX BUILD START | project_root=%s", PROJECT_ROOT)
    logger.info("O*NET JSON | path=%s | exists=%s", JSON_PATH, JSON_PATH.is_file())
    logger.info("Chroma directory | path=%s", CHROMA_DIR)

    if not JSON_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {JSON_PATH}. Run onet_loader.py first to generate occupations.json."
        )

    with JSON_PATH.open("r", encoding="utf-8") as file:
        occupations = json.load(file)

    if not isinstance(occupations, list) or not occupations:
        raise ValueError(f"Expected a non-empty JSON list of occupations in {JSON_PATH}")

    records: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in occupations:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        title = str(item.get("title") or "").strip()
        if not code or not title or code in seen_codes:
            continue
        seen_codes.add(code)
        records.append(item)

    if not records:
        raise ValueError("No valid occupations with both 'code' and 'title' were found.")

    logger.info("O*NET records loaded | count=%d", len(records))
    logger.info("Loading embedding model | name=%s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Rebuild only the occupation collection, not the separate RAG database.
    existing = {_collection_name(c) for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        logger.warning("Deleting existing collection before rebuilding | name=%s", COLLECTION_NAME)
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(records)
    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        embedding_texts = [_occupation_text(item) for item in batch]
        titles = [str(item["title"]).strip() for item in batch]
        ids = [str(item["code"]).strip() for item in batch]
        metadatas = []
        for item in batch:
            metadata = {
                "code": str(item["code"]).strip(),
                "title": str(item["title"]).strip(),
                "description": str(item.get("description") or ""),
                "job_zone": str(item.get("job_zone") if item.get("job_zone") is not None else "Unknown"),
            }
            metadatas.append(metadata)

        embeddings = model.encode(
            embedding_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Chroma's returned `documents` are occupation titles because matcher.py
        # currently uses results['documents'] as the displayed title. The vector
        # itself is computed from the richer occupation profile above.
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=titles,
            metadatas=metadatas,
        )

        logger.info(
            "INDEX BATCH SAVED | indexed=%d/%d | batch_size=%d",
            min(start + len(batch), total),
            total,
            len(batch),
        )

    logger.info(
        "INDEX BUILD COMPLETE | collection=%s | count=%d | path=%s",
        COLLECTION_NAME,
        collection.count(),
        CHROMA_DIR,
    )

    # Quick smoke test confirms the collection can be queried.
    probe = model.encode(
        ["software development programming computer systems"],
        normalize_embeddings=True,
    )[0]
    result = collection.query(
        query_embeddings=[probe.tolist()],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    logger.info("SMOKE TEST TOP TITLES | %s", result.get("documents", [[]])[0])


if __name__ == "__main__":
    main()
