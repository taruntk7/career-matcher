from pathlib import Path
import sys
import math


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

from matching.evaluation import (
    EVALUATION_PROFILES,
)

from matching.matcher import (
    load_matcher,
    match_occupations,
)

from matching.scoring import (
    rank_matches,
)


# =========================================================
# Metrics
# =========================================================

def precision_at_k(
    ranked_occupations,
    relevant_occupations,
    k,
):
    """
    Precision@K =
    relevant recommendations in top K / K
    """

    top_k = ranked_occupations[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1
        for occupation in top_k
        if occupation in relevant_occupations
    )

    return relevant_count / len(top_k)


def dcg_at_k(
    ranked_occupations,
    relevant_occupations,
    k,
):
    """
    Binary relevance DCG@K.
    """

    score = 0.0

    for index, occupation in enumerate(
        ranked_occupations[:k],
        start=1,
    ):

        if occupation in relevant_occupations:

            score += 1.0 / math.log2(
                index + 1
            )

    return score


def ndcg_at_k(
    ranked_occupations,
    relevant_occupations,
    k,
):
    """
    NDCG@K using binary relevance.
    """

    actual_dcg = dcg_at_k(
        ranked_occupations,
        relevant_occupations,
        k,
    )

    ideal_count = min(
        len(relevant_occupations),
        k,
    )

    ideal_dcg = sum(
        1.0 / math.log2(index + 1)
        for index in range(
            1,
            ideal_count + 1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# =========================================================
# Evaluation
# =========================================================

def evaluate():
    """
    Evaluate the current semantic recommendation pipeline.
    """

    model, collection = load_matcher()

    precision_scores = {
        1: [],
        3: [],
        5: [],
        10: [],
    }

    ndcg_scores = {
        5: [],
        10: [],
    }

    print()
    print("=" * 70)
    print("CAREER RECOMMENDATION EVALUATION")
    print("=" * 70)

    for profile in EVALUATION_PROFILES:

        profile_id = profile[
            "profile_id"
        ]

        description = profile[
            "description"
        ]

        relevant = set(
            profile[
                "relevant_occupations"
            ]
        )

        # -------------------------------------------------
        # Retrieve candidates
        # -------------------------------------------------

        matches = match_occupations(
            model,
            collection,
            description,
            top_k=50,
        )

        # -------------------------------------------------
        # Rank using the current ranking pipeline
        # -------------------------------------------------

        matches = rank_matches(
            matches
        )

        ranked_occupations = [
            match["title"]
            for match in matches
        ]

        # -------------------------------------------------
        # Calculate metrics
        # -------------------------------------------------

        p1 = precision_at_k(
            ranked_occupations,
            relevant,
            1,
        )

        p3 = precision_at_k(
            ranked_occupations,
            relevant,
            3,
        )

        p5 = precision_at_k(
            ranked_occupations,
            relevant,
            5,
        )

        p10 = precision_at_k(
            ranked_occupations,
            relevant,
            10,
        )

        n5 = ndcg_at_k(
            ranked_occupations,
            relevant,
            5,
        )

        n10 = ndcg_at_k(
            ranked_occupations,
            relevant,
            10,
        )

        precision_scores[1].append(p1)
        precision_scores[3].append(p3)
        precision_scores[5].append(p5)
        precision_scores[10].append(p10)

        ndcg_scores[5].append(n5)
        ndcg_scores[10].append(n10)

        # -------------------------------------------------
        # Display profile result
        # -------------------------------------------------

        print()
        print(
            f"Profile: {profile_id}"
        )

        print(
            f"Relevant occupations: "
            f"{', '.join(relevant)}"
        )

        print(
            "Top 5 recommendations:"
        )

        for rank, match in enumerate(
            matches[:5],
            start=1,
        ):

            marker = (
                "✓"
                if match["title"] in relevant
                else " "
            )

            print(
                f"  {rank}. "
                f"[{marker}] "
                f"{match['title']}"
                f" | score: "
                f"{match['final_score']:.3f}"
            )

        print(
            f"Precision@1:  {p1:.3f}"
        )

        print(
            f"Precision@3:  {p3:.3f}"
        )

        print(
            f"Precision@5:  {p5:.3f}"
        )

        print(
            f"NDCG@5:       {n5:.3f}"
        )

    # =====================================================
    # Overall results
    # =====================================================

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    for k in [1, 3, 5, 10]:

        average = (
            sum(
                precision_scores[k]
            )
            / len(
                precision_scores[k]
            )
        )

        print(
            f"Precision@{k}: "
            f"{average:.3f}"
        )

    for k in [5, 10]:

        average = (
            sum(
                ndcg_scores[k]
            )
            / len(
                ndcg_scores[k]
            )
        )

        print(
            f"NDCG@{k}:      "
            f"{average:.3f}"
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    evaluate()