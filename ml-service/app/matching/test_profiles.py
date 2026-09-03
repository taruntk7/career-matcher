from pathlib import Path
import sys


# =========================================================
# Project setup
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)

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

from matching.matcher import (
    load_matcher,
    match_occupations,
    enrich_with_skill_gaps,
    enrich_with_technical_skills,
)

from matching.capability_mapping import (
    infer_capabilities,
)

from matching.scoring import (
    rank_matches,
)

from matching.recommendation import (
    build_recommendation,
)


# =========================================================
# Run one test profile
# =========================================================

def run_profile(
    name,
    user_profile,
    user_skills,
):
    """
    Run one complete profile through the matcher.

    user_skills:
        Explicit technical / software / domain skills.

    inferred_capabilities:
        Derived separately from those explicit skills.

    IMPORTANT:

        Explicit skills -> technical matcher

        Inferred capabilities -> capability matcher

    They are NOT merged.
    """

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    # -----------------------------------------------------
    # User profile
    # -----------------------------------------------------

    print()
    print("User profile:")
    print(user_profile.strip())

    # -----------------------------------------------------
    # Explicit skills
    # -----------------------------------------------------

    print()
    print("Explicit skills:")

    for skill in user_skills:
        print(f"- {skill}")

    # -----------------------------------------------------
    # Infer capabilities
    # -----------------------------------------------------

    inferred_capabilities = infer_capabilities(
        user_skills
    )

    print()
    print("Inferred capabilities:")

    if inferred_capabilities:

        for capability in inferred_capabilities:

            print(
                f"- {capability['name']}"
                f" | evidence: "
                f"{', '.join(capability.get('evidence', []))}"
                f" | confidence: "
                f"{capability.get('confidence', 'inferred')}"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # Build semantic occupation profile
    # -----------------------------------------------------
    #
    # The occupation retrieval stage gets:
    #
    #   original user text
    #
    # plus:
    #
    #   inferred capability names
    #
    # This helps semantic retrieval understand broader
    # capabilities without turning them into technical skills.
    # -----------------------------------------------------

    capability_names = [
        capability["name"]
        for capability in inferred_capabilities
    ]

    matching_profile = user_profile

    if capability_names:

        matching_profile += (
            "\nRelevant capabilities: "
            + ", ".join(capability_names)
        )

    # -----------------------------------------------------
    # Find occupations
    # -----------------------------------------------------

    print()
    print("Finding occupations...")

    matches = match_occupations(
        model,
        collection,
        matching_profile,
        top_k=50,
    )

    # -----------------------------------------------------
    # Capability matching
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # Pass the ACTUAL inferred capability objects.
    #
    # Do NOT do:
    #
    #     user_skills + capability_names
    #
    # because that destroys:
    #
    #     evidence
    #     confidence
    #     evidence strength
    #
    # The capability matcher now understands objects
    # returned by infer_capabilities().
    # -----------------------------------------------------

    matches = enrich_with_skill_gaps(
        matches,
        user_skills,
        inferred_capabilities,
    )

    # -----------------------------------------------------
    # Technical matching
    # -----------------------------------------------------
    #
    # ONLY explicit skills go here.
    #
    # Capabilities such as:
    #
    #   Critical Thinking
    #   Coordination
    #   Negotiation
    #
    # must never become technical skills.
    # -----------------------------------------------------

    matches = enrich_with_technical_skills(
        matches,
        user_skills,
    )

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    matches = rank_matches(
        matches
    )

    # -----------------------------------------------------
    # Ranked results
    # -----------------------------------------------------

    print()
    print("Ranked matches:")
    print("-" * 70)

    for rank, match in enumerate(
        matches[:10],
        start=1,
    ):

        print(
            f"{rank}. {match['title']}"
            f" | final: "
            f"{match['final_score']:.3f}"
            f" | semantic: "
            f"{match['semantic_score']:.3f}"
            f" | essential: "
            f"{match.get('essential_score', 0.0):.3f}"
            f" | transferable: "
            f"{match.get('transferable_score', 0.0):.3f}"
            f" | technical: "
            f"{match.get('technical_score', 0.0):.3f}"
            f" | capability: "
            f"{match.get('capability_score', 0.0):.3f}"
        )

    # -----------------------------------------------------
    # No results guard
    # -----------------------------------------------------

    if not matches:
        print()
        print("No occupation matches found.")
        return

    # -----------------------------------------------------
    # Recommendation
    # -----------------------------------------------------

    recommendation = build_recommendation(
        matches[0]
    )

    print()
    print("Recommendation:")
    print("-" * 70)

    print(
        f"Occupation: "
        f"{recommendation['title']}"
    )

    print(
        f"Final score: "
        f"{recommendation['final_score']:.3f}"
    )

    # -----------------------------------------------------
    # Why this matches
    # -----------------------------------------------------

    print()
    print("Why this matches:")

    for reason in recommendation.get(
        "reasons",
        [],
    ):

        print(
            f"- {reason}"
        )

    # -----------------------------------------------------
    # Matched technical skills
    # -----------------------------------------------------

    print()
    print("Matched technical skills:")

    matched_technical = matches[0].get(
        "matched_technical_skills",
        [],
    )

    if matched_technical:

        for skill in matched_technical:

            print(
                f"- {skill['name']}"
                f" -> O*NET: "
                f"{skill['onet_name']}"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # Missing technical skills
    # -----------------------------------------------------

    print()
    print("Missing technical skills:")

    missing_technical = matches[0].get(
        "missing_technical_skills",
        [],
    )

    if missing_technical:

        for skill in missing_technical:

            print(
                f"- {skill['name']}"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # Matched essential capabilities
    # -----------------------------------------------------

    print()
    print("Matched essential capabilities:")

    matched_essential = matches[0].get(
        "matched_essential_skills",
        [],
    )

    if matched_essential:

        for capability in matched_essential:

            print(
                f"- {capability['name']}"
                f" (importance: "
                f"{capability['importance']}"
                f", evidence strength: "
                f"{capability.get('evidence_strength', 0.0):.2f}"
                f")"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # Matched transferable capabilities
    # -----------------------------------------------------

    print()
    print(
        "Matched transferable capabilities:"
    )

    matched_transferable = matches[0].get(
        "matched_transferable_skills",
        [],
    )

    if matched_transferable:

        for capability in matched_transferable:

            print(
                f"- {capability['name']}"
                f" (importance: "
                f"{capability['importance']}"
                f", evidence strength: "
                f"{capability.get('evidence_strength', 0.0):.2f}"
                f")"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # Missing essential capabilities
    # -----------------------------------------------------
    #
    # We deliberately do NOT call these "missing" if there
    # is no evidence. Your system should distinguish:
    #
    #   matched
    #   not evidenced
    #
    # rather than claiming that a resume proves absence.
    # -----------------------------------------------------

    print()
    print("Not-evidenced essential capabilities:")

    matched_names = {
        capability["name"].lower()
        for capability in matched_essential
    }

    occupation = matches[0].get(
        "_occupation",
        None,
    )

    # The current matcher does not expose the complete
    # occupation object here, so use recommendation data
    # when available.
    not_evidenced = recommendation.get(
        "missing_essential_skills",
        [],
    )

    not_evidenced = [
        capability
        for capability in not_evidenced
        if capability.get(
            "name",
            ""
        ).lower()
        not in matched_names
    ]

    if not_evidenced:

        for capability in not_evidenced[:5]:

            print(
                f"- {capability['name']}"
                f" (importance: "
                f"{capability['importance']})"
            )

    else:

        print("- None")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Load model + ChromaDB
    # -----------------------------------------------------

    model, collection = load_matcher()

    # =====================================================
    # PROFILE 1 — Technical / Data Science
    # =====================================================

    technical_profile = """
    Python programming, machine learning, statistics,
    data analysis, SQL, pandas, NumPy, scikit-learn,
    building predictive models and working with data.
    """

    technical_skills = [
        "Python",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
    ]

    run_profile(
        "PROFILE 1 — Technical / Data Science",
        technical_profile,
        technical_skills,
    )

    # =====================================================
    # PROFILE 2 — Management / Non-technical
    # =====================================================

    management_profile = """
    Team leadership, project management, communication,
    coordinating teams, planning business operations,
    stakeholder management, negotiation, decision making,
    managing people and improving organizational processes.
    """

    management_skills = [
        "Leadership",
        "Project Management",
        "Communication",
        "Team Management",
        "Planning",
        "Stakeholder Management",
        "Negotiation",
        "Decision Making",
        "Operations Management",
    ]

    run_profile(
        "PROFILE 2 — Management / Non-technical",
        management_profile,
        management_skills,
    )

    # =====================================================
    # PROFILE 3 — Law
    # =====================================================

    law_profile = """
    Legal research, contract drafting, case analysis,
    negotiation, client counseling, reviewing legal
    documents, analyzing cases and advising clients.
    """

    law_skills = [
        "Legal Research",
        "Contract Drafting",
        "Case Analysis",
        "Negotiation",
        "Client Counseling",
    ]

    run_profile(
        "PROFILE 3 — Law",
        law_profile,
        law_skills,
    )