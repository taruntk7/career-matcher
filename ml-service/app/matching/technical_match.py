from pathlib import Path
from typing import Dict, List
import math
import re
import sys


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

APP_DIR = (
    PROJECT_ROOT
    / "ml-service"
    / "app"
)

if str(APP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(APP_DIR),
    )


# =========================================================
# Utilities
# =========================================================

def normalize_skill(
    skill: str,
) -> str:
    """
    Normalize a skill name.
    """

    skill = str(
        skill
    ).lower().strip()

    skill = skill.replace(
        "&",
        "and",
    )

    skill = skill.replace(
        "-",
        " ",
    )

    skill = re.sub(
        r"[^a-z0-9\s]",
        "",
        skill,
    )

    skill = re.sub(
        r"\s+",
        " ",
        skill,
    )

    return skill


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a value into [0, 1].
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# =========================================================
# Aliases
# =========================================================

SKILL_ALIASES = {
    "sql":
        "structured query language sql",

    "structured query language":
        "structured query language sql",

    "scikit learn":
        "scikit-learn",

    "sklearn":
        "scikit-learn",

    "node js":
        "node.js",

    "nodejs":
        "node.js",

    "react js":
        "react",

    "reactjs":
        "react",

    "express js":
        "express",

    "expressjs":
        "express",

    "power bi":
        "microsoft power bi",

    "rest api":
        "restful api",

    "rest apis":
        "restful api",

    "web sockets":
        "websockets",
}


def canonicalize_skill_name(
    skill: str,
) -> str:
    """
    Normalize and apply aliases.
    """

    normalized = normalize_skill(
        skill
    )

    return SKILL_ALIASES.get(
        normalized,
        normalized,
    )


# =========================================================
# Non-technical skills
# =========================================================

NON_TECHNICAL_SKILLS = {
    "machine learning",
    "statistics",
    "data analysis",
    "data analytics",
    "predictive modeling",

    "leadership",
    "team leadership",
    "team management",
    "people management",
    "project management",
    "operations management",
    "planning",
    "strategic planning",

    "communication",
    "verbal communication",
    "written communication",
    "presentation",
    "presentations",

    "stakeholder management",
    "negotiation",
    "conflict resolution",

    "decision making",
    "decision-making",
    "problem solving",
    "problem-solving",

    "learning",
    "continuous learning",
}


def is_technical_skill(
    skill: str,
) -> bool:
    """
    Determine whether a user-provided skill is treated
    as an explicit technical/software skill.
    """

    normalized = normalize_skill(
        skill
    )

    return (
        normalized
        not in {
            normalize_skill(skill)
            for skill in NON_TECHNICAL_SKILLS
        }
    )


# =========================================================
# Extract technical user skills
# =========================================================

def extract_technical_user_skills(
    user_skills: List[str],
) -> List[str]:
    """
    Keep only explicit technical/software skills.
    """

    result = []

    seen = set()

    for skill in user_skills:

        if not is_technical_skill(
            skill
        ):
            continue

        canonical = canonicalize_skill_name(
            skill
        )

        if not canonical:
            continue

        if canonical in seen:
            continue

        result.append(
            skill
        )

        seen.add(
            canonical
        )

    return result


# =========================================================
# Occupation software skills
# =========================================================

def get_normalized_occupation_skills(
    occupation: Dict,
) -> Dict[str, str]:
    """
    Map normalized software skill names to their O*NET names.
    """

    result = {}

    for skill in occupation.get(
        "software_skills",
        [],
    ):

        canonical = canonicalize_skill_name(
            skill
        )

        if canonical:

            result[
                canonical
            ] = skill

    return result


# =========================================================
# Skill document frequency
# =========================================================

def build_software_document_frequency(
    occupations: List[Dict],
) -> Dict[str, int]:
    """
    Count how many occupations mention each software skill.

    This is used only to determine how distinctive a matched
    skill is across the occupation dataset.
    """

    document_frequency = {}

    for occupation in occupations:

        skills_in_occupation = set()

        for skill in occupation.get(
            "software_skills",
            [],
        ):

            canonical = canonicalize_skill_name(
                skill
            )

            if canonical:

                skills_in_occupation.add(
                    canonical
                )

        for skill in skills_in_occupation:

            document_frequency[
                skill
            ] = (
                document_frequency.get(
                    skill,
                    0,
                )
                + 1
            )

    return document_frequency


# =========================================================
# Skill specificity
# =========================================================

def calculate_skill_specificity(
    skill: str,
    document_frequency: Dict[str, int],
    occupation_count: int,
) -> float:
    """
    Give distinctive software skills more weight than
    extremely common software skills.

    Example:

        Git
            common -> lower weight

        Scikit-learn
            more occupation-specific -> higher weight
    """

    if occupation_count <= 0:
        return 1.0

    canonical = canonicalize_skill_name(
        skill
    )

    df = document_frequency.get(
        canonical,
        1,
    )

    df = max(
        1,
        min(
            df,
            occupation_count,
        ),
    )

    raw_idf = math.log(
        occupation_count / df
    )

    max_idf = math.log(
        occupation_count
    )

    if max_idf <= 0:
        return 1.0

    normalized_idf = (
        raw_idf
        / max_idf
    )

    # Common skills still matter.
    return clamp(
        0.30
        + 0.70 * normalized_idf
    )


# =========================================================
# Technical matching
# =========================================================

def calculate_technical_match(
    user_skills: List[str],
    occupation: Dict,
    all_occupations: List[Dict] | None = None,
) -> Dict:
    """
    Calculate occupation-specific technical fit.

    IMPORTANT:

    O*NET software_skills are treated as examples of software
    associated with an occupation.

    We therefore DO NOT calculate:

        matched software / all occupation software

    because that would incorrectly penalize a user for not
    knowing every software example listed by O*NET.

    Instead we calculate:

        weighted matched candidate skills
        --------------------------------
        weighted candidate technical skills

    Only the candidate's explicit technical skills are scored.

    A matched skill contributes more when it is distinctive
    to the occupation landscape and less when it is extremely
    common across occupations.
    """

    if all_occupations is None:

        all_occupations = [
            occupation
        ]

    occupation_count = max(
        1,
        len(all_occupations),
    )

    document_frequency = (
        build_software_document_frequency(
            all_occupations
        )
    )

    # -----------------------------------------------------
    # Candidate technical skills
    # -----------------------------------------------------

    technical_user_skills = (
        extract_technical_user_skills(
            user_skills
        )
    )

    user_by_normalized = {}

    for skill in technical_user_skills:

        canonical = canonicalize_skill_name(
            skill
        )

        if canonical:

            user_by_normalized[
                canonical
            ] = skill

    # -----------------------------------------------------
    # No candidate technical skills
    # -----------------------------------------------------

    if not user_by_normalized:

        return {
            "matched_skills": [],
            "missing_skills": [],
            "coverage_score": 0.0,
            "technical_applicable": False,
            "technical_user_skills": [],
            "user_technical_coverage": 0.0,
        }

    # -----------------------------------------------------
    # Occupation software vocabulary
    # -----------------------------------------------------

    occupation_skills = (
        get_normalized_occupation_skills(
            occupation
        )
    )

    # If O*NET does not provide software examples for this
    # occupation, technical matching is simply unavailable.
    if not occupation_skills:

        return {
            "matched_skills": [],
            "missing_skills": [],
            "coverage_score": 0.0,
            "technical_applicable": False,
            "technical_user_skills":
                technical_user_skills,
            "user_technical_coverage": 0.0,
        }

    # =====================================================
    # Match candidate skills
    # =====================================================

    matched_skills = []

    matched_normalized = set()

    for (
        normalized_user_skill,
        original_user_skill,
    ) in user_by_normalized.items():

        onet_name = occupation_skills.get(
            normalized_user_skill
        )

        if onet_name is None:
            continue

        specificity = (
            calculate_skill_specificity(
                onet_name,
                document_frequency,
                occupation_count,
            )
        )

        matched_skills.append(
            {
                "name":
                    original_user_skill,

                "onet_name":
                    onet_name,

                "specificity":
                    specificity,
            }
        )

        matched_normalized.add(
            normalized_user_skill
        )

    # =====================================================
    # Missing candidate skills
    # =====================================================

    missing_skills = []

    for (
        normalized_user_skill,
        original_user_skill,
    ) in user_by_normalized.items():

        if (
            normalized_user_skill
            not in matched_normalized
        ):

            missing_skills.append(
                {
                    "name":
                        original_user_skill,
                }
            )

    # =====================================================
    # Weighted candidate coverage
    # =====================================================

    total_weight = 0.0
    matched_weight = 0.0

    for (
        normalized_user_skill,
        original_user_skill,
    ) in user_by_normalized.items():

        specificity = (
            calculate_skill_specificity(
                original_user_skill,
                document_frequency,
                occupation_count,
            )
        )

        total_weight += (
            specificity
        )

        if (
            normalized_user_skill
            in occupation_skills
        ):

            matched_weight += (
                specificity
            )

    if total_weight > 0:

        technical_score = (
            matched_weight
            / total_weight
        )

    else:

        technical_score = 0.0

    technical_score = clamp(
        technical_score
    )

    return {
        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "coverage_score":
            technical_score,

        "technical_applicable":
            True,

        "technical_user_skills":
            technical_user_skills,

        "user_technical_coverage":
            technical_score,

        "technical_score":
            technical_score,
    }


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    from data.onet_loader import (
        build_occupation_dataset,
    )

    occupations = (
        build_occupation_dataset()
    )

    data_scientist = next(
        occupation
        for occupation in occupations
        if occupation[
            "title"
        ]
        == "Data Scientists"
    )

    software_developer = next(
        occupation
        for occupation in occupations
        if occupation[
            "title"
        ]
        == "Software Developers"
    )

    user_skills = [
        "Python",
        "Java",
        "C++",
        "JavaScript",
        "TypeScript",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Power BI",
        "Excel",
        "React",
        "Node.js",
        "Express",
        "FastAPI",
        "REST APIs",
        "WebSockets",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Git",
        "GitHub",
        "Docker",
        "Jupyter",
        "Linux",
    ]

    # -----------------------------------------------------
    # Data Scientists
    # -----------------------------------------------------

    result = calculate_technical_match(
        user_skills,
        data_scientist,
        occupations,
    )

    print()
    print("=" * 70)
    print("DATA SCIENTISTS")
    print("=" * 70)

    print()
    print(
        "Matched technical skills:"
    )

    for skill in result[
        "matched_skills"
    ]:

        print(
            f"- {skill['name']}"
            f" -> O*NET: "
            f"{skill['onet_name']}"
            f" | specificity: "
            f"{skill['specificity']:.3f}"
        )

    print()
    print(
        "Technical coverage:",
        f"{result['coverage_score']:.3f}",
    )

    # -----------------------------------------------------
    # Software Developers
    # -----------------------------------------------------

    result = calculate_technical_match(
        user_skills,
        software_developer,
        occupations,
    )

    print()
    print("=" * 70)
    print("SOFTWARE DEVELOPERS")
    print("=" * 70)

    print()
    print(
        "Matched technical skills:"
    )

    for skill in result[
        "matched_skills"
    ]:

        print(
            f"- {skill['name']}"
            f" -> O*NET: "
            f"{skill['onet_name']}"
            f" | specificity: "
            f"{skill['specificity']:.3f}"
        )

    print()
    print(
        "Technical coverage:",
        f"{result['coverage_score']:.3f}",
    )