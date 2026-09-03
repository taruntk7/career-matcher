from pathlib import Path
import sys


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "ml-service"
        / "app"
    ),
)


# =========================================================
# Imports
# =========================================================

from matching.capability_mapping import (
    infer_capabilities,
)


# =========================================================
# Test helper
# =========================================================

def test_profile(
    name,
    skills,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("Explicit skills:")

    for skill in skills:
        print(f"- {skill}")

    capabilities = infer_capabilities(
        skills
    )

    print()
    print("Inferred capabilities:")

    if not capabilities:

        print("- None")
        return

    for capability in capabilities:

        print(
            f"- {capability['name']}"
            f" | evidence: "
            f"{', '.join(capability.get('evidence', []))}"
            f" | confidence: "
            f"{capability.get('confidence', 'unknown')}"
            f" | evidence strength: "
            f"{capability.get('evidence_strength', 'not-set')}"
        )


# =========================================================
# Tests
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Technical / analytical
    # -----------------------------------------------------

    data_science_skills = [
        "Python",
        "Machine Learning",
        "Data Analysis",
    ]

    test_profile(
        "DATA SCIENCE",
        data_science_skills,
    )

    # -----------------------------------------------------
    # Management / interpersonal
    # -----------------------------------------------------

    management_skills = [
        "Leadership",
        "Team Management",
        "Stakeholder Management",
        "Negotiation",
    ]

    test_profile(
        "MANAGEMENT",
        management_skills,
    )

    # -----------------------------------------------------
    # Law
    # -----------------------------------------------------

    law_skills = [
        "Legal Research",
        "Contract Drafting",
        "Case Analysis",
        "Negotiation",
        "Client Counseling",
    ]

    test_profile(
        "LAW",
        law_skills,
    )

    # -----------------------------------------------------
    # Single-skill tests
    #
    # These are important because we want to see whether
    # one explicit skill automatically creates too many
    # inferred capabilities.
    # -----------------------------------------------------

    single_skill_tests = {
        "ONLY PYTHON": [
            "Python",
        ],

        "ONLY MACHINE LEARNING": [
            "Machine Learning",
        ],

        "ONLY LEADERSHIP": [
            "Leadership",
        ],

        "ONLY NEGOTIATION": [
            "Negotiation",
        ],

        "ONLY LEGAL RESEARCH": [
            "Legal Research",
        ],
    }

    for name, skills in (
        single_skill_tests.items()
    ):

        test_profile(
            name,
            skills,
        )