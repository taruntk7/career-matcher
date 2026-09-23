from pathlib import Path
import json


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OCCUPATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
    / "occupations.json"
)

RAG_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag"
    / "documents"
)


# =========================================================
# Load occupations
# =========================================================

def load_occupations() -> list[dict]:
    """
    Load the processed O*NET occupation dataset.
    """

    if not OCCUPATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find occupations dataset:\n"
            f"{OCCUPATIONS_PATH}"
        )

    with open(
        OCCUPATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        occupations = json.load(f)

    if not isinstance(
        occupations,
        list,
    ):

        raise ValueError(
            "occupations.json must contain a list."
        )

    return occupations


# =========================================================
# Build one occupation document
# =========================================================

def build_occupation_document(
    occupation: dict,
) -> str:
    """
    Convert one structured O*NET occupation into
    a single textual RAG document.
    """

    title = str(
        occupation.get(
            "title",
            "",
        )
    ).strip()

    code = str(
        occupation.get(
            "code",
            "",
        )
    ).strip()

    description = str(
        occupation.get(
            "description",
            "",
        )
    ).strip()

    job_zone = occupation.get(
        "job_zone"
    )

    lines = []

    # -----------------------------------------------------
    # Occupation
    # -----------------------------------------------------

    lines.append(
        f"Occupation: {title}"
    )

    lines.append(
        f"O*NET-SOC Code: {code}"
    )

    # -----------------------------------------------------
    # Description
    # -----------------------------------------------------

    if description:

        lines.append("")
        lines.append("Description:")
        lines.append(description)

    # -----------------------------------------------------
    # Typical tasks
    # -----------------------------------------------------

    tasks = occupation.get(
        "tasks",
        [],
    )

    if tasks:

        lines.append("")
        lines.append("Typical Tasks:")

        for task in tasks:

            task_text = str(
                task
            ).strip()

            if task_text:
                lines.append(
                    f"- {task_text}"
                )

    # -----------------------------------------------------
    # Essential capabilities
    # -----------------------------------------------------

    essential_skills = occupation.get(
        "essential_skills",
        [],
    )

    if essential_skills:

        lines.append("")
        lines.append(
            "Essential Capabilities:"
        )

        for skill in essential_skills:

            if not isinstance(
                skill,
                dict,
            ):
                continue

            name = str(
                skill.get(
                    "name",
                    "",
                )
            ).strip()

            importance = skill.get(
                "importance"
            )

            if not name:
                continue

            if importance is not None:

                lines.append(
                    f"- {name} "
                    f"(importance: {float(importance):.2f})"
                )

            else:

                lines.append(
                    f"- {name}"
                )

    # -----------------------------------------------------
    # Transferable capabilities
    # -----------------------------------------------------

    transferable_skills = occupation.get(
        "transferable_skills",
        [],
    )

    if transferable_skills:

        lines.append("")
        lines.append(
            "Transferable Capabilities:"
        )

        for skill in transferable_skills:

            if not isinstance(
                skill,
                dict,
            ):
                continue

            name = str(
                skill.get(
                    "name",
                    "",
                )
            ).strip()

            importance = skill.get(
                "importance"
            )

            if not name:
                continue

            if importance is not None:

                lines.append(
                    f"- {name} "
                    f"(importance: {float(importance):.2f})"
                )

            else:

                lines.append(
                    f"- {name}"
                )

    # -----------------------------------------------------
    # Software / technology
    # -----------------------------------------------------

    software_skills = occupation.get(
        "software_skills",
        [],
    )

    if software_skills:

        lines.append("")
        lines.append(
            "Associated Software and Technology:"
        )

        seen_software = set()

        for software in software_skills:

            software_text = str(
                software
            ).strip()

            if not software_text:
                continue

            normalized = software_text.lower()

            if normalized in seen_software:
                continue

            seen_software.add(
                normalized
            )

            lines.append(
                f"- {software_text}"
            )

    # -----------------------------------------------------
    # Job Zone
    # -----------------------------------------------------

    if job_zone is not None:

        lines.append("")
        lines.append(
            f"Job Zone: {job_zone}"
        )

    return "\n".join(lines)


# =========================================================
# Build all RAG documents
# =========================================================

def build_all_documents() -> int:
    """
    Build one text document for every occupation.
    """

    occupations = load_occupations()

    RAG_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    created = 0

    for occupation in occupations:

        code = str(
            occupation.get(
                "code",
                "",
            )
        ).strip()

        title = str(
            occupation.get(
                "title",
                "occupation",
            )
        ).strip()

        if not code:
            continue

        document = build_occupation_document(
            occupation
        )

        output_path = (
            RAG_DATA_DIR
            / f"{code.replace('.', '_')}.txt"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(document)

        created += 1

    return created


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    count = build_all_documents()

    print()
    print("=" * 70)
    print("RAG DOCUMENT BUILDER")
    print("=" * 70)

    print()
    print(
        f"Created {count} occupation documents."
    )

    print(
        f"Saved to: {RAG_DATA_DIR}"
    )