from typing import Dict, List, Any
from pathlib import Path
import re
import math


# =========================================================
# Utilities
# =========================================================

def normalize(text: str) -> str:
    """
    Normalize text for capability-name comparison.
    """

    text = str(text).lower().strip()

    text = text.replace(
        "-",
        " ",
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp value into [minimum, maximum].
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# =========================================================
# User capability parsing
# =========================================================

def _capability_name(
    capability: Any,
) -> str:
    """
    Accept both:

        "Critical Thinking"

    and:

        {
            "name": "Critical Thinking",
            ...
        }
    """

    if isinstance(
        capability,
        str,
    ):

        return capability

    if isinstance(
        capability,
        dict,
    ):

        return str(
            capability.get(
                "name",
                capability.get(
                    "capability",
                    "",
                ),
            )
        )

    return ""


def _capability_strength(
    capability: Any,
) -> float:
    """
    Resolve the strongest available evidence-strength field.

    Priority:

        evidence_strength
        similarity
        confidence
        default
    """

    if not isinstance(
        capability,
        dict,
    ):

        return 0.50

    if (
        "evidence_strength"
        in capability
    ):

        return clamp(
            capability[
                "evidence_strength"
            ]
        )

    if (
        "similarity"
        in capability
    ):

        return clamp(
            capability[
                "similarity"
            ]
        )

    if (
        "confidence"
        in capability
    ):

        confidence = capability[
            "confidence"
        ]

        if isinstance(
            confidence,
            (int, float),
        ):

            return clamp(
                confidence
            )

    return 0.50


def _normalize_user_capabilities(
    user_capabilities: List[Any],
) -> Dict[str, Dict]:
    """
    Convert all user capability formats into a canonical
    dictionary keyed by normalized capability name.
    """

    normalized = {}

    for capability in (
        user_capabilities or []
    ):

        name = _capability_name(
            capability
        ).strip()

        if not name:
            continue

        key = normalize(
            name
        )

        if isinstance(
            capability,
            dict,
        ):

            evidence = capability.get(
                "evidence",
                [],
            )

            if isinstance(
                evidence,
                str,
            ):

                evidence = [
                    evidence
                ]

            canonical = {
                **capability,

                "name":
                    name,

                "evidence":
                    evidence,

                "evidence_strength":
                    _capability_strength(
                        capability
                    ),
            }

        else:

            canonical = {
                "name":
                    name,

                "evidence":
                    [],

                "confidence":
                    1.0,

                "evidence_strength":
                    1.0,

                "source":
                    "explicit",
            }

        existing = normalized.get(
            key
        )

        if (
            existing is None
            or canonical[
                "evidence_strength"
            ]
            > existing[
                "evidence_strength"
            ]
        ):

            normalized[key] = (
                canonical
            )

    return normalized


# =========================================================
# Occupation capability vocabulary
# =========================================================

def build_capability_vocabulary(
    occupations: List[Dict],
) -> List[str]:
    """
    Build the global O*NET capability vocabulary.
    """

    capabilities = set()

    for occupation in occupations:

        for skill in occupation.get(
            "essential_skills",
            [],
        ):

            name = skill.get(
                "name"
            )

            if name:
                capabilities.add(
                    name
                )

        for skill in occupation.get(
            "transferable_skills",
            [],
        ):

            name = skill.get(
                "name"
            )

            if name:
                capabilities.add(
                    name
                )

    return sorted(
        capabilities
    )


# =========================================================
# Capability frequency / specificity
# =========================================================

def build_capability_document_frequency(
    occupations: List[Dict],
) -> Dict[str, int]:
    """
    Count how many occupations contain each capability.

    This lets us distinguish common capabilities from
    occupation-specific capabilities.

    Example:

        Programming
            appears in many occupations
            -> low specificity

        Negotiation
            appears in fewer occupations
            -> higher specificity
    """

    document_frequency = {}

    for occupation in occupations:

        occupation_capabilities = set()

        for skill in occupation.get(
            "essential_skills",
            [],
        ):

            name = skill.get(
                "name"
            )

            if name:

                occupation_capabilities.add(
                    normalize(name)
                )

        for skill in occupation.get(
            "transferable_skills",
            [],
        ):

            name = skill.get(
                "name"
            )

            if name:

                occupation_capabilities.add(
                    normalize(name)
                )

        for capability_name in (
            occupation_capabilities
        ):

            document_frequency[
                capability_name
            ] = (
                document_frequency.get(
                    capability_name,
                    0,
                )
                + 1
            )

    return document_frequency


def calculate_capability_specificity(
    capability_name: str,
    document_frequency: Dict[str, int],
    occupation_count: int,
) -> float:
    """
    Calculate a normalized specificity weight.

    Uses an IDF-like measure:

        log(N / df)

    and normalizes it into approximately [0, 1].

    Common capabilities receive a smaller weight.
    Rare capabilities receive a larger weight.

    A minimum specificity is retained so common capabilities
    do not become completely irrelevant.
    """

    if occupation_count <= 0:
        return 1.0

    normalized = normalize(
        capability_name
    )

    df = document_frequency.get(
        normalized,
        1,
    )

    df = max(
        1,
        min(
            df,
            occupation_count,
        )
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

    # Keep a floor so extremely common capabilities still
    # contribute somewhat.
    specificity = (
        0.35
        + 0.65 * normalized_idf
    )

    return clamp(
        specificity
    )


# =========================================================
# Occupation capability extraction
# =========================================================

def get_occupation_capabilities(
    occupation: Dict,
) -> Dict:
    """
    Return essential and transferable capabilities in a
    common structure.
    """

    essential = [
        {
            "name":
                skill["name"],

            "importance":
                float(
                    skill["importance"]
                ),

            "type":
                "essential",
        }

        for skill in occupation.get(
            "essential_skills",
            [],
        )

        if skill.get(
            "name"
        )
    ]

    transferable = [
        {
            "name":
                skill["name"],

            "importance":
                float(
                    skill["importance"]
                ),

            "type":
                "transferable",
        }

        for skill in occupation.get(
            "transferable_skills",
            [],
        )

        if skill.get(
            "name"
        )
    ]

    return {
        "essential":
            essential,

        "transferable":
            transferable,
    }


# =========================================================
# Capability matching
# =========================================================

def calculate_capability_match(
    user_capabilities: List[Dict],
    occupation: Dict,
    all_occupations: List[Dict] | None = None,
) -> Dict:
    """
    Compare user capabilities with an occupation's O*NET
    essential and transferable capabilities.

    Improvements over the previous implementation:

    1. Essential and transferable scores remain separate.

    2. Evidence strength from the LLM is respected.

    3. Capability specificity is considered:
       capabilities common across many occupations receive
       less discriminative weight.

    4. The final capability score is a weighted combination:

           70% essential
           30% transferable

    This prevents broad transferable capabilities from
    dominating occupation-specific essential requirements.
    """

    user_by_name = (
        _normalize_user_capabilities(
            user_capabilities
        )
    )

    # -----------------------------------------------------
    # Build global capability frequency information.
    # -----------------------------------------------------

    if all_occupations is None:

        # Safe fallback if caller only provides one occupation.
        all_occupations = [
            occupation
        ]

    occupation_count = max(
        1,
        len(all_occupations),
    )

    document_frequency = (
        build_capability_document_frequency(
            all_occupations
        )
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    matched_essential = []
    matched_transferable = []

    # -----------------------------------------------------
    # Essential capability score
    # -----------------------------------------------------

    total_essential_weight = 0.0
    matched_essential_weight = 0.0

    for skill in occupation.get(
        "essential_skills",
        [],
    ):

        name = skill.get(
            "name",
            "",
        ).strip()

        if not name:
            continue

        importance = float(
            skill.get(
                "importance",
                0.0,
            )
        )

        specificity = (
            calculate_capability_specificity(
                name,
                document_frequency,
                occupation_count,
            )
        )

        weighted_importance = (
            importance
            * specificity
        )

        total_essential_weight += (
            weighted_importance
        )

        user_capability = (
            user_by_name.get(
                normalize(name)
            )
        )

        if user_capability:

            evidence_strength = clamp(
                user_capability[
                    "evidence_strength"
                ]
            )

            contribution = (
                weighted_importance
                * evidence_strength
            )

            matched_essential_weight += (
                contribution
            )

            matched_essential.append(
                {
                    "name":
                        name,

                    "importance":
                        importance,

                    "specificity":
                        specificity,

                    "weighted_importance":
                        weighted_importance,

                    "evidence_strength":
                        evidence_strength,

                    "evidence_similarity":
                        evidence_strength,

                    "evidence":
                        user_capability.get(
                            "evidence",
                            [],
                        ),

                    "confidence":
                        user_capability.get(
                            "confidence",
                            evidence_strength,
                        ),
                }
            )

    # -----------------------------------------------------
    # Transferable capability score
    # -----------------------------------------------------

    total_transferable_weight = 0.0
    matched_transferable_weight = 0.0

    for skill in occupation.get(
        "transferable_skills",
        [],
    ):

        name = skill.get(
            "name",
            "",
        ).strip()

        if not name:
            continue

        importance = float(
            skill.get(
                "importance",
                0.0,
            )
        )

        specificity = (
            calculate_capability_specificity(
                name,
                document_frequency,
                occupation_count,
            )
        )

        weighted_importance = (
            importance
            * specificity
        )

        total_transferable_weight += (
            weighted_importance
        )

        user_capability = (
            user_by_name.get(
                normalize(name)
            )
        )

        if user_capability:

            evidence_strength = clamp(
                user_capability[
                    "evidence_strength"
                ]
            )

            contribution = (
                weighted_importance
                * evidence_strength
            )

            matched_transferable_weight += (
                contribution
            )

            matched_transferable.append(
                {
                    "name":
                        name,

                    "importance":
                        importance,

                    "specificity":
                        specificity,

                    "weighted_importance":
                        weighted_importance,

                    "evidence_strength":
                        evidence_strength,

                    "evidence_similarity":
                        evidence_strength,

                    "evidence":
                        user_capability.get(
                            "evidence",
                            [],
                        ),

                    "confidence":
                        user_capability.get(
                            "confidence",
                            evidence_strength,
                        ),
                }
            )

    # -----------------------------------------------------
    # Normalize essential score
    # -----------------------------------------------------

    if (
        total_essential_weight
        > 0
    ):

        essential_score = (
            matched_essential_weight
            / total_essential_weight
        )

    else:

        essential_score = 0.0

    # -----------------------------------------------------
    # Normalize transferable score
    # -----------------------------------------------------

    if (
        total_transferable_weight
        > 0
    ):

        transferable_score = (
            matched_transferable_weight
            / total_transferable_weight
        )

    else:

        transferable_score = 0.0

    essential_score = clamp(
        essential_score
    )

    transferable_score = clamp(
        transferable_score
    )

    # -----------------------------------------------------
    # Overall capability score
    # -----------------------------------------------------
    #
    # Essential capabilities describe what is fundamentally
    # required.
    #
    # Transferable capabilities describe broader abilities.
    #
    # Therefore essential gets the larger contribution.
    # -----------------------------------------------------

    capability_score = (
        0.70 * essential_score
        + 0.30 * transferable_score
    )

    capability_score = clamp(
        capability_score
    )

    return {
        "matched_essential_skills":
            matched_essential,

        "matched_transferable_skills":
            matched_transferable,

        "essential_score":
            essential_score,

        "transferable_score":
            transferable_score,

        "capability_score":
            capability_score,
    }


# =========================================================
# Standalone tests
# =========================================================

if __name__ == "__main__":

    from pathlib import Path
    import sys

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
            / "ml-service"
            / "app"
        ),
    )

    from data.onet_loader import (
        build_occupation_dataset,
    )

    occupations = (
        build_occupation_dataset()
    )

    # -----------------------------------------------------
    # Test Data Scientists
    # -----------------------------------------------------

    data_scientist = next(
        occupation
        for occupation in occupations
        if occupation[
            "title"
        ]
        == "Data Scientists"
    )

    data_science_capabilities = [

        {
            "name":
                "Programming",

            "evidence":
                [
                    "Developed a Python-based analytics service."
                ],

            "confidence":
                0.96,

            "evidence_strength":
                0.864,
        },

        {
            "name":
                "Critical Thinking",

            "evidence":
                [
                    "Compared model performance and improved feature selection."
                ],

            "confidence":
                0.95,

            "evidence_strength":
                0.855,
        },

        {
            "name":
                "Complex Problem Solving",

            "evidence":
                [
                    "Investigated data-quality issues and process bottlenecks."
                ],

            "confidence":
                0.96,

            "evidence_strength":
                0.864,
        },

        {
            "name":
                "Operations Analysis",

            "evidence":
                [
                    "Analyzed transaction data and operational patterns."
                ],

            "confidence":
                0.95,

            "evidence_strength":
                0.855,
        },
    ]

    print()
    print("=" * 70)
    print("CAPABILITY MATCHING TEST")
    print("=" * 70)

    result = calculate_capability_match(
        data_science_capabilities,
        data_scientist,
        occupations,
    )

    print()
    print(
        "Occupation:",
        data_scientist["title"],
    )

    print()
    print(
        "Essential score:",
        f"{result['essential_score']:.3f}",
    )

    print(
        "Transferable score:",
        f"{result['transferable_score']:.3f}",
    )

    print(
        "Capability score:",
        f"{result['capability_score']:.3f}",
    )

    print()
    print(
        "Matched essential capabilities:"
    )

    for capability in result[
        "matched_essential_skills"
    ]:

        print(
            f"- {capability['name']}"
            f" | importance: "
            f"{capability['importance']:.2f}"
            f" | specificity: "
            f"{capability['specificity']:.3f}"
            f" | evidence: "
            f"{capability['evidence_strength']:.3f}"
        )

    print()
    print(
        "Matched transferable capabilities:"
    )

    for capability in result[
        "matched_transferable_skills"
    ]:

        print(
            f"- {capability['name']}"
            f" | importance: "
            f"{capability['importance']:.2f}"
            f" | specificity: "
            f"{capability['specificity']:.3f}"
            f" | evidence: "
            f"{capability['evidence_strength']:.3f}"
        )