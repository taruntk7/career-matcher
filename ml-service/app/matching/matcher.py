from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer
import chromadb


MODEL_NAME = "all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHROMA_DIR = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
    / "chroma"
)


# ---------------------------------------------------------
# Load matcher
# ---------------------------------------------------------

def load_matcher():
    """
    Load the embedding model and ChromaDB collection.
    """

    model = SentenceTransformer(
        MODEL_NAME
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name="occupations"
    )

    return model, collection


# ---------------------------------------------------------
# Semantic matching
# ---------------------------------------------------------

def match_occupations(
    model,
    collection,
    user_text,
    top_k=10,
):
    """
    Return the most semantically similar occupations.
    """

    user_embedding = model.encode(
        user_text,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[
            user_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    for i in range(
        len(results["ids"][0])
    ):

        distance = float(
            results["distances"][0][i]
        )

        matches.append(
            {
                "code": results["ids"][0][i],

                "title": results[
                    "documents"
                ][0][i],

                "distance": distance,

                "similarity": (
                    1.0 - distance
                ),
            }
        )

    return matches


# ---------------------------------------------------------
# Load occupation data once
# ---------------------------------------------------------

def _load_occupations_by_code():

    from data.onet_loader import (
        build_occupation_dataset
    )

    occupations = (
        build_occupation_dataset()
    )

    return {
        occupation["code"]: occupation
        for occupation in occupations
    }


# ---------------------------------------------------------
# Capability enrichment
# ---------------------------------------------------------

def enrich_with_capabilities(
    matches,
    user_capabilities,
):
    """
    Match inferred user capabilities against:

        - O*NET essential skills
        - O*NET transferable skills

    Capability matching is kept separate from
    technical/software skill matching.
    """

    from matching.capability_matching import (
        calculate_capability_match
    )

    occupations_by_code = (
        _load_occupations_by_code()
    )

    enriched_matches = []

    for match in matches:

        occupation = (
            occupations_by_code.get(
                match["code"]
            )
        )

        if occupation is None:

            enriched_matches.append(
                match
            )

            continue

        result = (
            calculate_capability_match(
                user_capabilities,
                occupation,
            )
        )

        # -------------------------------------------------
        # Capability score
        # -------------------------------------------------
        #
        # Combine essential and transferable capability
        # coverage into one capability signal.
        #
        # If neither exists for the occupation, score = 0.
        # -------------------------------------------------

        essential_score = float(
            result.get(
                "essential_score",
                0.0,
            )
        )

        transferable_score = float(
            result.get(
                "transferable_score",
                0.0,
            )
        )

        if (
            essential_score > 0
            or transferable_score > 0
        ):

            capability_score = (
                essential_score
                + transferable_score
            ) / 2.0

        else:

            capability_score = 0.0

        enriched_match = {
            **match,

            "matched_essential_skills":
                result.get(
                    "matched_essential_skills",
                    [],
                ),

            "matched_transferable_skills":
                result.get(
                    "matched_transferable_skills",
                    [],
                ),

            "essential_score":
                essential_score,

            "transferable_score":
                transferable_score,

            "capability_score":
                capability_score,
        }

        enriched_matches.append(
            enriched_match
        )

    return enriched_matches


# ---------------------------------------------------------
# Backwards-compatible essential enrichment
# ---------------------------------------------------------

def enrich_with_skill_gaps(
    matches,
    user_skills,
    user_capabilities=None,
):
    """
    Backwards-compatible wrapper.

    Existing test code can still call this function.

    If user_capabilities are supplied, use them for
    capability matching.

    Otherwise, preserve the old behaviour by using
    user_skills as the capability evidence.
    """

    if user_capabilities is None:

        user_capabilities = (
            user_skills
        )

    enriched = enrich_with_capabilities(
        matches,
        user_capabilities,
    )

    final_matches = []

    for match in enriched:

        final_matches.append(
            {
                **match,

                # Old names retained so existing code
                # doesn't immediately break.
                "matched_skills":
                    match.get(
                        "matched_essential_skills",
                        [],
                    ),

                # We intentionally do not label unknown
                # capabilities as missing here.
                "missing_skills": [],

                "skill_coverage":
                    match.get(
                        "essential_score",
                        0.0,
                    ),
            }
        )

    return final_matches


# ---------------------------------------------------------
# Technical enrichment
# ---------------------------------------------------------

def enrich_with_technical_skills(
    matches,
    user_skills,
    occupations,
):
    """
    Match explicit technical/software skills.

    Technical skills are treated separately from broader
    capabilities.
    """

    from matching.technical_match import (
        calculate_technical_match
    )

    occupations_by_code = (
        _load_occupations_by_code()
    )

    enriched_matches = []

    for match in matches:

        occupation = (
            occupations_by_code.get(
                match["code"]
            )
        )

        if occupation is None:

            enriched_matches.append(
                match
            )

            continue

        technical = (
            calculate_technical_match(
                user_skills,
                occupation,
                occupations,
            )
        )

        enriched_match = {
            **match,

            "matched_technical_skills":
                technical[
                    "matched_skills"
                ],

            "missing_technical_skills":
                technical[
                    "missing_skills"
                ],

            "technical_coverage":
                technical[
                    "coverage_score"
                ],

            "technical_applicable":
                technical[
                    "technical_applicable"
                ],

            "technical_user_skills":
                technical[
                    "technical_user_skills"
                ],
        }

        enriched_matches.append(
            enriched_match
        )

    return enriched_matches


# ---------------------------------------------------------
# Combined enrichment
# ---------------------------------------------------------

def enrich_matches(
    matches,
    user_skills,
    user_capabilities,
    occupations,
):
    """
    Apply all matching signals.

    user_skills:
        Explicit skills extracted from the resume.

    user_capabilities:
        Capabilities explicitly supplied or inferred from
        resume evidence.
    """

    if user_capabilities is None:

        user_capabilities = (
            user_skills
        )

    # -----------------------------------------------------
    # Capability matching
    # -----------------------------------------------------

    matches = enrich_with_capabilities(
        matches,
        user_capabilities,
    )

    # -----------------------------------------------------
    # Technical skill matching
    # -----------------------------------------------------

    matches = enrich_with_technical_skills(
        matches,
        user_skills,
        occupations,
    )

    return matches


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    from matching.scoring import (
        rank_matches
    )

    model, collection = (
        load_matcher()
    )

    user_profile = """
    Python programming, machine learning, statistics,
    data analysis, SQL, pandas, NumPy, scikit-learn,
    building predictive models and working with data.
    """

    user_skills = [
        "Python",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
    ]

    user_capabilities = [
        "Mathematics",
        "Critical Thinking",
        "Programming",
        "Complex Problem Solving",
    ]

    print()
    print("User profile:")
    print(user_profile.strip())

    print()
    print("Explicit technical skills:")

    for skill in user_skills:
        print(f"- {skill}")

    print()
    print("Capability evidence:")

    for capability in user_capabilities:
        print(f"- {capability}")

    print()
    print("Finding occupations...")

    matches = match_occupations(
        model,
        collection,
        user_profile,
        top_k=10,
    )

    matches = enrich_matches(
        matches,
        user_skills,
        user_capabilities,
    )

    matches = rank_matches(
        matches
    )

    print()
    print("Ranked matches:")
    print("-" * 80)

    for rank, match in enumerate(
        matches,
        start=1,
    ):

        print(
            f"{rank}. {match['title']}"
            f" | final: "
            f"{match['final_score']:.3f}"
            f" | semantic: "
            f"{match['semantic_score']:.3f}"
            f" | technical: "
            f"{match.get('technical_score', 0.0):.3f}"
            f" | essential: "
            f"{match.get('essential_score', 0.0):.3f}"
            f" | transferable: "
            f"{match.get('transferable_score', 0.0):.3f}"
            f" | capability: "
            f"{match.get('capability_score', 0.0):.3f}"
        )

        print(
            "   Technical matches:"
        )

        for skill in match.get(
            "matched_technical_skills",
            [],
        ):

            print(
                f"      - {skill['name']}"
                f" -> O*NET: "
                f"{skill['onet_name']}"
            )

        print(
            "   Essential evidence:"
        )

        for skill in match.get(
            "matched_essential_skills",
            [],
        ):

            print(
                f"      - {skill['name']}"
                f" (importance: "
                f"{skill['importance']})"
            )

        print(
            "   Transferable evidence:"
        )

        for skill in match.get(
            "matched_transferable_skills",
            [],
        ):

            print(
                f"      - {skill['name']}"
                f" (importance: "
                f"{skill['importance']})"
            )

        print()