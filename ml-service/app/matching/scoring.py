from typing import Dict


# =========================================================
# Default scoring weights
# =========================================================
#
# Semantic:
#   How well the user's overall profile matches the
#   occupation semantically.
#
# Capability:
#   How well the user's evidenced capabilities align with
#   the occupation's essential + transferable capabilities.
#
# Technical:
#   How well the user's explicit technical/software skills
#   align with the occupation.
#
# Technical is optional because many occupations do not have
# meaningful software/technology evidence.
# =========================================================

DEFAULT_WEIGHTS = {
    "semantic": 0.50,
    "capability": 0.25,
    "technical": 0.25,
}


# =========================================================
# Calculate final score
# =========================================================

def calculate_final_score(
    match: Dict,
    weights: Dict = DEFAULT_WEIGHTS,
) -> Dict:
    """
    Combine semantic, capability, and technical signals.

    Capability already combines:
        - essential capability alignment
        - transferable capability alignment

    Technical matching is optional.

    If technical matching is not applicable, its weight is
    redistributed proportionally between semantic and
    capability matching.
    """

    # -----------------------------------------------------
    # Semantic score
    # -----------------------------------------------------

    semantic = float(
        match.get(
            "similarity",
            match.get(
                "semantic_score",
                0.0,
            ),
        )
    )

    # -----------------------------------------------------
    # Capability score
    # -----------------------------------------------------
    #
    # This is the IMPORTANT change.
    #
    # Do not use the old "skill_coverage" field as the
    # capability score.
    #
    # capability_score is produced by capability_matching.py.
    # -----------------------------------------------------

    capability = float(
        match.get(
            "capability_score",
            0.0,
        )
    )

    # -----------------------------------------------------
    # Technical score
    # -----------------------------------------------------

    technical = float(
        match.get(
            "technical_coverage",
            0.0,
        )
    )

    technical_applicable = bool(
        match.get(
            "technical_applicable",
            False,
        )
    )

    # -----------------------------------------------------
    # Read weights
    # -----------------------------------------------------

    semantic_weight = float(
        weights.get(
            "semantic",
            0.0,
        )
    )

    capability_weight = float(
        weights.get(
            "capability",
            0.0,
        )
    )

    technical_weight = float(
        weights.get(
            "technical",
            0.0,
        )
    )

    # -----------------------------------------------------
    # Handle non-applicable technical matching
    # -----------------------------------------------------

    if not technical_applicable:

        applicable_weight = (
            semantic_weight
            + capability_weight
        )

        if applicable_weight <= 0:

            raise ValueError(
                "At least one applicable scoring weight "
                "must be greater than 0."
            )

        semantic_weight = (
            semantic_weight
            / applicable_weight
        )

        capability_weight = (
            capability_weight
            / applicable_weight
        )

        technical_weight = 0.0

    else:

        total_weight = (
            semantic_weight
            + capability_weight
            + technical_weight
        )

        if total_weight <= 0:

            raise ValueError(
                "At least one scoring weight "
                "must be greater than 0."
            )

        semantic_weight /= total_weight
        capability_weight /= total_weight
        technical_weight /= total_weight

    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    final_score = (
        semantic * semantic_weight
        + capability * capability_weight
        + technical * technical_weight
    )

    # -----------------------------------------------------
    # Return all individual signals
    # -----------------------------------------------------

    return {
        "semantic_score": semantic,

        "capability_score": capability,

        "technical_score": technical,

        # Keep essential + transferable visible.
        # These are NOT independently weighted here.
        "essential_score": float(
            match.get(
                "essential_score",
                0.0,
            )
        ),

        "transferable_score": float(
            match.get(
                "transferable_score",
                0.0,
            )
        ),

        "final_score": final_score,
    }


# =========================================================
# Rank matches
# =========================================================

def rank_matches(
    matches,
    weights: Dict = DEFAULT_WEIGHTS,
):
    """
    Calculate final scores and rank occupations.
    """

    scored_matches = []

    for match in matches:

        score = calculate_final_score(
            match,
            weights,
        )

        scored_match = {
            **match,
            **score,
        }

        scored_matches.append(
            scored_match
        )

    scored_matches.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return scored_matches


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    test_matches = [

        {
            "title": "Data Scientists",
            "similarity": 0.471,
            "capability_score": 0.105,
            "essential_score": 0.121,
            "transferable_score": 0.089,
            "technical_coverage": 1.000,
            "technical_applicable": True,
        },

        {
            "title": "Project Management Specialists",
            "similarity": 0.626,
            "capability_score": 0.137,
            "essential_score": 0.000,
            "transferable_score": 0.274,
            "technical_coverage": 0.000,
            "technical_applicable": False,
        },

        {
            "title": "Lawyers",
            "similarity": 0.677,
            "capability_score": 0.040,
            "essential_score": 0.000,
            "transferable_score": 0.080,
            "technical_coverage": 0.000,
            "technical_applicable": False,
        },
    ]

    ranked = rank_matches(
        test_matches
    )

    print()
    print("Scoring test:")
    print("-" * 80)

    for rank, match in enumerate(
        ranked,
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
            f"{match['capability_score']:.3f}"
            f" | essential: "
            f"{match['essential_score']:.3f}"
            f" | transferable: "
            f"{match['transferable_score']:.3f}"
            f" | technical: "
            f"{match['technical_score']:.3f}"
        )