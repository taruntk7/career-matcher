from pathlib import Path
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
# Imports
# =========================================================

from parsing.resume_analyzer import (
    analyze_resume_file,
)

from data.onet_loader import (
    build_occupation_dataset,
)

from matching.matcher import (
    load_matcher,
    match_occupations,
    enrich_matches,
)

from matching.scoring import (
    rank_matches,
)

from matching.recommendation import (
    build_recommendation,
)

from rag.retriever import (
    load_retriever,
    retrieve_occupation,
)

from rag.generator import (
    generate_answer,
)


# =========================================================
# Main
# =========================================================

def main():

    resume_path = (
        Path.home()
        / "Downloads"
        / "sample_resume.pdf"
    )

    # -----------------------------------------------------
    # 1. Analyze resume
    # -----------------------------------------------------

    result = analyze_resume_file(
        resume_path
    )

    analysis = result[
        "analysis"
    ]

    explicit_skills = [
        item["name"]
        for item in analysis.get(
            "explicit_skills",
            [],
        )
    ]

    capabilities = analysis.get(
        "capabilities",
        [],
    )

    print()
    print("=" * 70)
    print("RESUME → MATCHING PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # 2. Display extracted skills
    # -----------------------------------------------------

    print()
    print("Explicit skills:")
    print("-" * 70)

    for skill in explicit_skills:

        print(
            f"- {skill}"
        )

    # -----------------------------------------------------
    # 3. Display capabilities
    # -----------------------------------------------------

    print()
    print("Canonical O*NET capabilities:")
    print("-" * 70)

    for capability in capabilities:

        print(
            f"- {capability['name']}"
            f" | evidence strength: "
            f"{float(capability.get('evidence_strength', 0.0)):.3f}"
        )

        print(
            f"  Evidence: "
            f"{capability.get('evidence', '')}"
        )

    # -----------------------------------------------------
    # 4. Load matcher + O*NET data
    # -----------------------------------------------------

    model, collection = (
        load_matcher()
    )

    occupations = (
        build_occupation_dataset()
    )

    # -----------------------------------------------------
    # 5. Semantic retrieval
    # -----------------------------------------------------

    profile_text = result[
        "raw_text"
    ]

    matches = match_occupations(
        model,
        collection,
        profile_text,
        top_k=50,
    )

    # -----------------------------------------------------
    # 6. Enrichment
    # -----------------------------------------------------

    matches = enrich_matches(
        matches,
        explicit_skills,
        capabilities,
        occupations,
    )

    # -----------------------------------------------------
    # 7. Ranking
    # -----------------------------------------------------

    matches = rank_matches(
        matches
    )

    # -----------------------------------------------------
    # 8. Ranked occupations
    # -----------------------------------------------------

    print()
    print("Top occupations:")
    print("-" * 70)

    for rank, match in enumerate(
        matches[:10],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{match['title']}"
            f" | final: "
            f"{match['final_score']:.3f}"
            f" | semantic: "
            f"{match['semantic_score']:.3f}"
            f" | capability: "
            f"{match.get('capability_score', 0.0):.3f}"
            f" | technical: "
            f"{match.get('technical_score', 0.0):.3f}"
        )

    # -----------------------------------------------------
    # 9. Build actual recommendation
    # -----------------------------------------------------

    if not matches:

        print()
        print(
            "No occupation matches were found."
        )

        return

    recommendation = (
        build_recommendation(
            matches[0]
        )
    )

    # -----------------------------------------------------
    # 10. Recommendation summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    print()
    print(
        f"Occupation: "
        f"{recommendation['title']}"
    )

    print(
        f"Final score: "
        f"{recommendation['final_score']:.3f}"
    )

    print(
        f"Semantic score: "
        f"{recommendation['semantic_score']:.3f}"
    )

    print(
        f"Capability score: "
        f"{recommendation['capability_score']:.3f}"
    )

    print(
        f"Essential score: "
        f"{recommendation['essential_score']:.3f}"
    )

    print(
        f"Transferable score: "
        f"{recommendation['transferable_score']:.3f}"
    )

    print(
        f"Technical score: "
        f"{recommendation['technical_score']:.3f}"
    )

    # -----------------------------------------------------
    # 11. Why this matches
    # -----------------------------------------------------

    print()
    print("Why this matches:")

    for reason in recommendation[
        "reasons"
    ]:

        print(
            f"- {reason}"
        )

    # -----------------------------------------------------
    # 12. Strongest capabilities
    # -----------------------------------------------------

    print()
    print("Top capability strengths:")
    print("-" * 70)

    strengths = recommendation[
        "top_capability_strengths"
    ]

    if strengths:

        for capability in strengths:

            print(
                f"- {capability['name']}"
                f" | importance: "
                f"{capability['importance']:.2f}"
                f" | evidence strength: "
                f"{capability['evidence_strength']:.3f}"
            )

            for evidence in capability[
                "evidence"
            ]:

                print(
                    f"  Evidence: "
                    f"{evidence}"
                )

    else:

        print(
            "- None"
        )

    # -----------------------------------------------------
    # 13. Technical strengths
    # -----------------------------------------------------

    print()
    print("Top technical strengths:")
    print("-" * 70)

    technical_strengths = recommendation[
        "top_technical_strengths"
    ]

    if technical_strengths:

        for skill in technical_strengths:

            print(
                f"- {skill['name']}"
                f" -> O*NET: "
                f"{skill['onet_name']}"
                f" | specificity: "
                f"{skill['specificity']:.3f}"
            )

    else:

        print(
            "- None"
        )

    # -----------------------------------------------------
    # 14. Potential capability gaps
    # -----------------------------------------------------

    print()
    print("Potential capability gaps:")
    print("-" * 70)

    capability_gaps = recommendation[
        "potential_capability_gaps"
    ]

    if capability_gaps:

        for gap in capability_gaps:

            print(
                f"- {gap['name']}"
                f" | type: "
                f"{gap['type']}"
                f" | importance: "
                f"{gap['importance']:.2f}"
                f" | priority: "
                f"{gap['priority']:.3f}"
                f" | status: "
                f"{gap['evidence_status']}"
            )

    else:

        print(
            "- None"
        )

    # -----------------------------------------------------
    # 15. Technical mismatches
    # -----------------------------------------------------

    print()
    print("Technical mismatches:")
    print("-" * 70)

    technical_mismatches = recommendation[
        "technical_mismatches"
    ]

    if technical_mismatches:

        for mismatch in technical_mismatches:

            print(
                f"- {mismatch}"
            )

    else:

        print(
            "- None"
        )

    # =====================================================
    # RAG CAREER GUIDANCE
    # =====================================================

    # -----------------------------------------------------
    # 16. Load RAG retriever
    # -----------------------------------------------------

    rag_model, rag_collection = (
        load_retriever()
    )

    # -----------------------------------------------------
    # 17. Get recommended occupation
    # -----------------------------------------------------

    recommended_occupation = (
        recommendation["title"]
    )

    recommended_code = (
        recommendation.get("code")
    )

    # -----------------------------------------------------
    # 18. Retrieve exact occupation document
    # -----------------------------------------------------

    occupation_document = (
        retrieve_occupation(
            collection=rag_collection,
            occupation_code=recommended_code,
            occupation_title=recommended_occupation,
        )
    )

    # -----------------------------------------------------
    # 19. Generate personalized RAG guidance
    # -----------------------------------------------------

    question = (
        "What should I focus on learning next "
        "to become more suitable for this career?"
    )

    rag_answer = generate_answer(
        occupation=recommended_occupation,
        question=question,
        occupation_document=occupation_document,
        capability_gaps=capability_gaps,
        technical_mismatches=technical_mismatches,
        capability_strengths=strengths,
        technical_strengths=technical_strengths,
    )

    # -----------------------------------------------------
    # 20. Display RAG guidance
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("RAG-BASED CAREER GUIDANCE")
    print("=" * 70)

    print()

    if occupation_document:

        print(
            f"Knowledge source: "
            f"{occupation_document['metadata'].get('title')}"
        )

        print(
            f"O*NET code: "
            f"{occupation_document['metadata'].get('code')}"
        )

    print()
    print(rag_answer)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()