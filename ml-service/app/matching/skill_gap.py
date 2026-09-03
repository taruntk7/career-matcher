import re
from typing import List, Dict


def normalize_skill(skill: str) -> str:
    """
    Normalize a skill name for comparison.
    """

    skill = str(skill).lower().strip()

    skill = skill.replace("&", "and")

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


# ---------------------------------------------------------
# Explicit aliases
# ---------------------------------------------------------

SKILL_ALIASES = {
    "problem solving": "complex problem solving",
    "problem solving skills": "complex problem solving",
    "decision making": "judgment and decision making",
    "decision making skills": "judgment and decision making",
}


def canonical_skill(skill: str) -> str:
    """
    Normalize and apply a small set of safe aliases.
    """

    normalized = normalize_skill(skill)

    return SKILL_ALIASES.get(
        normalized,
        normalized,
    )


def calculate_capability_match(
    user_skills: List[str],
    occupation: Dict,
) -> Dict:
    """
    Compare user-provided/inferred capabilities against
    both O*NET essential and transferable skills.

    IMPORTANT:

    This does not assume that an unmentioned skill is
    missing from the user's resume.

    It only measures positive evidence.

    Returns:

        matched_essential_skills
        matched_transferable_skills
        essential_coverage
        transferable_coverage
        capability_coverage
    """

    user_normalized = {
        canonical_skill(skill)
        for skill in user_skills
    }

    # -----------------------------------------------------
    # Essential skills
    # -----------------------------------------------------

    essential_skills = occupation.get(
        "essential_skills",
        [],
    )

    matched_essential = []

    total_essential_importance = 0.0
    matched_essential_importance = 0.0

    for skill in essential_skills:

        name = skill["name"]

        importance = float(
            skill["importance"]
        )

        normalized = canonical_skill(name)

        total_essential_importance += importance

        if normalized in user_normalized:

            matched_essential.append(
                {
                    "name": name,
                    "importance": importance,
                }
            )

            matched_essential_importance += importance

    if total_essential_importance > 0:

        essential_coverage = (
            matched_essential_importance
            / total_essential_importance
        )

    else:

        essential_coverage = 0.0

    # -----------------------------------------------------
    # Transferable skills
    # -----------------------------------------------------

    transferable_skills = occupation.get(
        "transferable_skills",
        [],
    )

    matched_transferable = []

    total_transferable_importance = 0.0
    matched_transferable_importance = 0.0

    for skill in transferable_skills:

        name = skill["name"]

        importance = float(
            skill["importance"]
        )

        normalized = canonical_skill(name)

        total_transferable_importance += importance

        if normalized in user_normalized:

            matched_transferable.append(
                {
                    "name": name,
                    "importance": importance,
                }
            )

            matched_transferable_importance += importance

    if total_transferable_importance > 0:

        transferable_coverage = (
            matched_transferable_importance
            / total_transferable_importance
        )

    else:

        transferable_coverage = 0.0

    # -----------------------------------------------------
    # Combined capability score
    #
    # Essential and transferable skills are kept separate
    # because they mean different things.
    #
    # For now, use an equal combination.
    # This is NOT the final career-ranking formula.
    # -----------------------------------------------------

    capability_coverage = (
        essential_coverage * 0.5
        + transferable_coverage * 0.5
    )

    return {
        "matched_essential_skills": matched_essential,
        "matched_transferable_skills": matched_transferable,
        "essential_coverage": essential_coverage,
        "transferable_coverage": transferable_coverage,
        "capability_coverage": capability_coverage,
    }


# ---------------------------------------------------------
# Backwards-compatible wrapper
# ---------------------------------------------------------

def calculate_skill_gap(
    user_skills: List[str],
    occupation: Dict,
) -> Dict:
    """
    Backwards-compatible wrapper.

    Existing code can still call calculate_skill_gap(),
    but internally it now uses the new capability model.
    """

    result = calculate_capability_match(
        user_skills,
        occupation,
    )

    return {
        "matched_skills": result[
            "matched_essential_skills"
        ],
        "missing_skills": [],
        "coverage_score": result[
            "essential_coverage"
        ],
        "matched_essential_skills": result[
            "matched_essential_skills"
        ],
        "matched_transferable_skills": result[
            "matched_transferable_skills"
        ],
        "essential_coverage": result[
            "essential_coverage"
        ],
        "transferable_coverage": result[
            "transferable_coverage"
        ],
        "capability_coverage": result[
            "capability_coverage"
        ],
    }


if __name__ == "__main__":

    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    sys.path.insert(
        0,
        str(PROJECT_ROOT / "ml-service" / "app"),
    )

    from data.onet_loader import (
        build_occupation_dataset
    )

    occupations = build_occupation_dataset()

    occupation = next(
        occupation
        for occupation in occupations
        if occupation["title"] == "Data Scientists"
    )

    user_skills = [
        "Mathematics",
        "Critical Thinking",
        "Programming",
        "Complex Problem Solving",
    ]

    result = calculate_capability_match(
        user_skills,
        occupation,
    )

    print()
    print(
        "Occupation:",
        occupation["title"],
    )

    print()
    print("User capabilities:")

    for skill in user_skills:
        print(f"- {skill}")

    print()
    print("Matched essential skills:")

    for skill in result[
        "matched_essential_skills"
    ]:
        print(
            f"- {skill['name']}"
            f" (importance: {skill['importance']})"
        )

    print()
    print("Matched transferable skills:")

    for skill in result[
        "matched_transferable_skills"
    ]:
        print(
            f"- {skill['name']}"
            f" (importance: {skill['importance']})"
        )

    print()
    print(
        "Essential coverage:",
        f"{result['essential_coverage']:.3f}",
    )

    print(
        "Transferable coverage:",
        f"{result['transferable_coverage']:.3f}",
    )

    print(
        "Capability coverage:",
        f"{result['capability_coverage']:.3f}",
    )