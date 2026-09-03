from pathlib import Path
import json
import pandas as pd


# Project root:
# career-matcher/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ONET_RAW_DIR = PROJECT_ROOT / "data" / "onet" / "raw"


def load_occupations() -> pd.DataFrame:
    """Load basic occupation information."""
    path = ONET_RAW_DIR / "occupation_data.csv"

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Title": "title",
            "Description": "description",
        }
    )

    return df[["code", "title", "description"]]


def load_tasks() -> pd.DataFrame:
    """Load occupation task statements."""
    path = ONET_RAW_DIR / "task_statements.csv"

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Task": "task",
        }
    )

    return df[["code", "task"]]


def load_essential_skills() -> pd.DataFrame:
    """Load essential skills and their importance scores."""
    path = ONET_RAW_DIR / "essential_skills.csv"

    df = pd.read_csv(path)

    # We only want the Importance scale.
    df = df[df["Scale ID"] == "IM"].copy()

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Element Name": "skill",
            "Data Value": "importance",
        }
    )

    return df[["code", "skill", "importance"]]


def load_transferable_skills() -> pd.DataFrame:
    """Load transferable skills and their importance scores."""
    path = ONET_RAW_DIR / "transferable_skills.csv"

    df = pd.read_csv(path)

    # Only keep importance scores.
    df = df[df["Scale ID"] == "IM"].copy()

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Element Name": "skill",
            "Data Value": "importance",
        }
    )

    return df[["code", "skill", "importance"]]


def load_software_skills() -> pd.DataFrame:
    """Load software/technology examples associated with occupations."""
    path = ONET_RAW_DIR / "software_skills.csv"

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Workplace Example": "software",
        }
    )

    return df[["code", "software"]]


def load_job_zones() -> pd.DataFrame:
    """Load O*NET Job Zone information."""
    path = ONET_RAW_DIR / "job_zones.csv"

    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "O*NET-SOC Code": "code",
            "Job Zone": "job_zone",
        }
    )

    return df[["code", "job_zone"]]

def build_occupation_dataset() -> list[dict]:
    """Combine O*NET datasets into one occupation-level dataset."""

    occupations = load_occupations()
    tasks = load_tasks()
    essential_skills = load_essential_skills()
    transferable_skills = load_transferable_skills()
    software_skills = load_software_skills()
    job_zones = load_job_zones()

    # Group one-to-many data by occupation code.
    tasks_by_code = (
        tasks.groupby("code")["task"]
        .apply(list)
        .to_dict()
    )

    essential_by_code = (
        essential_skills.groupby("code")
        .apply(
            lambda group: [
                {
                    "name": row["skill"],
                    "importance": float(row["importance"]),
                }
                for _, row in group.iterrows()
            ],
            include_groups=False,
        )
        .to_dict()
    )

    transferable_by_code = (
        transferable_skills.groupby("code")
        .apply(
            lambda group: [
                {
                    "name": row["skill"],
                    "importance": float(row["importance"]),
                }
                for _, row in group.iterrows()
            ],
            include_groups=False,
        )
        .to_dict()
    )

    software_by_code = (
        software_skills.groupby("code")["software"]
        .apply(list)
        .to_dict()
    )

    job_zone_by_code = (
        job_zones.drop_duplicates("code")
        .set_index("code")["job_zone"]
        .to_dict()
    )

    result = []

    for _, occupation in occupations.iterrows():
        code = occupation["code"]

        result.append(
            {
                "code": code,
                "title": occupation["title"],
                "description": occupation["description"],
                "tasks": tasks_by_code.get(code, []),
                "essential_skills": essential_by_code.get(code, []),
                "transferable_skills": transferable_by_code.get(code, []),
                "software_skills": software_by_code.get(code, []),
                "job_zone": (
                    int(job_zone_by_code[code])
                    if code in job_zone_by_code
                    else None
                ),
            }
        )

    return result

if __name__ == "__main__":
    occupations = build_occupation_dataset()

    output_dir = PROJECT_ROOT / "data" / "onet" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "occupations.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(occupations, f, indent=2, ensure_ascii=False)

    print(f"Built dataset with {len(occupations)} occupations")
    print(f"Saved to: {output_path}")

    print()
    print("Data coverage:")
    print(f"Tasks: {sum(bool(x['tasks']) for x in occupations)}")
    print(
        "Essential skills:",
        sum(bool(x["essential_skills"]) for x in occupations)
    )
    print(
        "Transferable skills:",
        sum(bool(x["transferable_skills"]) for x in occupations)
    )
    print(
        "Software skills:",
        sum(bool(x["software_skills"]) for x in occupations)
    )
    print(
        "Job zones:",
        sum(x["job_zone"] is not None for x in occupations)
    )