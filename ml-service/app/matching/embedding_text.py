import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
    / "occupations.json"
)


# Limit the number of tasks included in the semantic representation.
# O*NET can contain many tasks for one occupation, and including all of
# them can make the embedding overly long and cause important information
# to be truncated by the sentence-transformer model.
MAX_TASKS = 10


def load_occupations() -> list[dict]:
    """Load the processed O*NET occupation dataset."""

    with open(PROCESSED_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def build_embedding_text(occupation: dict) -> str:
    """
    Convert an occupation into a semantic text representation.

    The semantic representation focuses on:
    - occupation title
    - occupation description
    - representative tasks
    - essential skills
    - transferable skills

    Software and technology are intentionally excluded here because
    technical matching is handled separately by technical_match.py.

    Numerical importance scores and Job Zone are also excluded because
    they are used separately during ranking and skill-gap analysis.
    """

    lines = [
        f"Occupation: {occupation['title']}",
        f"Description: {occupation['description']}",
    ]

    # ---------------------------------------------------------
    # Tasks
    # ---------------------------------------------------------

    tasks = occupation.get("tasks", [])

    if tasks:
        lines.append("\nTypical Tasks:")

        for task in tasks[:MAX_TASKS]:
            lines.append(f"- {task}")

    # ---------------------------------------------------------
    # Essential skills
    # ---------------------------------------------------------

    essential_skills = occupation.get(
        "essential_skills",
        [],
    )

    if essential_skills:
        lines.append("\nEssential Skills:")

        for skill in essential_skills:
            lines.append(f"- {skill['name']}")

    # ---------------------------------------------------------
    # Transferable skills
    # ---------------------------------------------------------

    transferable_skills = occupation.get(
        "transferable_skills",
        [],
    )

    if transferable_skills:
        lines.append("\nTransferable Skills:")

        for skill in transferable_skills:
            lines.append(f"- {skill['name']}")

    return "\n".join(lines)


if __name__ == "__main__":

    occupations = load_occupations()

    # Test the first occupation.
    text = build_embedding_text(
        occupations[0]
    )

    print(text)