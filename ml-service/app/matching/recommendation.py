from typing import Any, Dict, List


# =========================================================
# Safe numeric conversion
# =========================================================

def to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a float.

    Handles:
        int
        float
        numeric strings
        dictionaries containing a numeric score

    This function ALWAYS returns a float.
    """

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if isinstance(
        value,
        str,
    ):

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    if isinstance(
        value,
        dict,
    ):

        # Try common score keys.
        for key in (
            "score",
            "value",
            "final_score",
            "semantic_score",
            "capability_score",
            "technical_score",
        ):

            if key not in value:
                continue

            nested = value[key]

            if isinstance(
                nested,
                (int, float),
            ):

                return float(nested)

            if isinstance(
                nested,
                str,
            ):

                try:
                    return float(nested)

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

    return default


# =========================================================
# Evidence helpers
# =========================================================

def clean_evidence(
    evidence: Any,
) -> List[str]:
    """
    Convert evidence into a clean list of strings.
    """

    if evidence is None:
        return []

    if isinstance(
        evidence,
        str,
    ):

        text = evidence.strip()

        return (
            [text]
            if text
            else []
        )

    if isinstance(
        evidence,
        list,
    ):

        result = []

        for item in evidence:

            text = str(
                item
            ).strip()

            if text:
                result.append(
                    text
                )

        return result

    text = str(
        evidence
    ).strip()

    return (
        [text]
        if text
        else []
    )


# =========================================================
# Gap priority
# =========================================================

def calculate_gap_priority(
    importance: float,
) -> float:
    """
    Estimate the priority of a potential capability gap.

    O*NET importance is treated as a 0-5 scale and
    normalized into [0, 1].

    This is a prioritization signal, NOT a probability
    that the candidate must learn the capability.
    """

    importance = max(
        0.0,
        min(
            5.0,
            float(importance),
        ),
    )

    return importance / 5.0


# =========================================================
# Sort helpers
# =========================================================

def sort_by_importance(
    items: List[Dict],
) -> List[Dict]:
    """
    Sort capabilities by O*NET importance.
    """

    return sorted(
        items,
        key=lambda item: to_float(
            item.get(
                "importance",
                0.0,
            )
        ),
        reverse=True,
    )


def sort_by_evidence(
    items: List[Dict],
) -> List[Dict]:
    """
    Sort capabilities by evidence strength and then
    O*NET importance.
    """

    return sorted(
        items,
        key=lambda item: (
            to_float(
                item.get(
                    "evidence_strength",
                    0.0,
                )
            ),
            to_float(
                item.get(
                    "importance",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )


# =========================================================
# Reason generation
# =========================================================

def build_reasons(
    match: Dict,
) -> List[str]:
    """
    Generate explanation text from already-computed scores.
    """

    semantic = to_float(
        match.get(
            "semantic_score",
            match.get(
                "similarity",
                0.0,
            ),
        )
    )

    capability = to_float(
        match.get(
            "capability_score",
            0.0,
        )
    )

    essential = to_float(
        match.get(
            "essential_score",
            0.0,
        )
    )

    transferable = to_float(
        match.get(
            "transferable_score",
            0.0,
        )
    )

    technical = to_float(
        match.get(
            "technical_score",
            match.get(
                "technical_coverage",
                0.0,
            ),
        )
    )

    reasons = []

    # -----------------------------------------------------
    # Semantic
    # -----------------------------------------------------

    if semantic >= 0.65:

        reasons.append(
            "The overall resume content has a strong "
            "semantic match with this occupation."
        )

    elif semantic >= 0.50:

        reasons.append(
            "The overall resume content has a good "
            "semantic match with this occupation."
        )

    elif semantic >= 0.35:

        reasons.append(
            "The resume has a moderate semantic "
            "relationship with this occupation."
        )

    else:

        reasons.append(
            "The semantic relationship between the resume "
            "and this occupation is relatively weak."
        )

    # -----------------------------------------------------
    # Essential capabilities
    # -----------------------------------------------------

    if essential >= 0.25:

        reasons.append(
            "Several important capabilities required by "
            "the occupation are supported by the resume."
        )

    elif essential > 0:

        reasons.append(
            "The resume provides evidence for some of "
            "the occupation's essential capabilities."
        )

    else:

        reasons.append(
            "No occupation-specific essential capabilities "
            "were directly matched."
        )

    # -----------------------------------------------------
    # Transferable capabilities
    # -----------------------------------------------------

    if transferable >= 0.25:

        reasons.append(
            "The resume also demonstrates several "
            "transferable capabilities relevant to "
            "the occupation."
        )

    elif transferable > 0:

        reasons.append(
            "The resume demonstrates some transferable "
            "capabilities relevant to the occupation."
        )

    # -----------------------------------------------------
    # Technical
    # -----------------------------------------------------

    if technical >= 0.65:

        reasons.append(
            "A substantial portion of the candidate's "
            "explicit technical skills is relevant "
            "to this occupation."
        )

    elif technical >= 0.35:

        reasons.append(
            "Several of the candidate's explicit technical "
            "skills are relevant to this occupation."
        )

    elif technical > 0:

        reasons.append(
            "Some explicit technical skills are relevant "
            "to this occupation."
        )

    else:

        reasons.append(
            "No explicit technical/software skills "
            "were matched."
        )

    return reasons


# =========================================================
# Build one recommendation
# =========================================================

def build_recommendation(
    match: Dict,
) -> Dict:
    """
    Convert one ranked occupation match into a structured
    recommendation.

    This function does not change the ranking.
    """

    # -----------------------------------------------------
    # Scores
    # -----------------------------------------------------

    title = str(
        match.get(
            "title",
            "Unknown occupation",
        )
    )

    final_score = to_float(
        match.get(
            "final_score",
            0.0,
        )
    )

    semantic_score = to_float(
        match.get(
            "semantic_score",
            match.get(
                "similarity",
                0.0,
            ),
        )
    )

    capability_score = to_float(
        match.get(
            "capability_score",
            0.0,
        )
    )

    essential_score = to_float(
        match.get(
            "essential_score",
            0.0,
        )
    )

    transferable_score = to_float(
        match.get(
            "transferable_score",
            0.0,
        )
    )

    technical_score = to_float(
        match.get(
            "technical_score",
            match.get(
                "technical_coverage",
                0.0,
            ),
        )
    )

    # -----------------------------------------------------
    # Evidence
    # -----------------------------------------------------

    matched_technical = list(
        match.get(
            "matched_technical_skills",
            match.get(
                "matched_skills",
                [],
            ),
        )
        or []
    )

    missing_technical = list(
        match.get(
            "missing_technical_skills",
            match.get(
                "missing_skills",
                [],
            ),
        )
        or []
    )

    matched_essential = list(
        match.get(
            "matched_essential_skills",
            [],
        )
        or []
    )

    matched_transferable = list(
        match.get(
            "matched_transferable_skills",
            [],
        )
        or []
    )

    # -----------------------------------------------------
    # Capability strengths
    # -----------------------------------------------------

    ordered_essential = sort_by_evidence(
        matched_essential
    )

    ordered_transferable = sort_by_evidence(
        matched_transferable
    )

    top_capability_strengths = []

    seen = set()

    for capability in (
        ordered_essential
        + ordered_transferable
    ):

        name = str(
            capability.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            continue

        key = name.lower()

        if key in seen:
            continue

        seen.add(
            key
        )

        top_capability_strengths.append(
            {
                "name":
                    name,

                "importance":
                    to_float(
                        capability.get(
                            "importance",
                            0.0,
                        )
                    ),

                "evidence_strength":
                    to_float(
                        capability.get(
                            "evidence_strength",
                            0.0,
                        )
                    ),

                "evidence":
                    clean_evidence(
                        capability.get(
                            "evidence",
                            [],
                        )
                    ),
            }
        )

        if len(
            top_capability_strengths
        ) >= 5:

            break

    # -----------------------------------------------------
    # Technical strengths
    # -----------------------------------------------------

    top_technical_strengths = []

    for skill in matched_technical:

        top_technical_strengths.append(
            {
                "name":
                    str(
                        skill.get(
                            "name",
                            "",
                        )
                    ),

                "onet_name":
                    str(
                        skill.get(
                            "onet_name",
                            "",
                        )
                    ),

                "specificity":
                    to_float(
                        skill.get(
                            "specificity",
                            0.0,
                        )
                    ),
            }
        )

    top_technical_strengths.sort(
        key=lambda item:
            item["specificity"],
        reverse=True,
    )

    top_technical_strengths = (
        top_technical_strengths[:8]
    )

    # -----------------------------------------------------
    # Potential capability gaps
    # -----------------------------------------------------

    matched_names = {
        str(
            item.get(
                "name",
                "",
            )
        ).strip().lower()

        for item in (
            matched_essential
            + matched_transferable
        )

        if item.get(
            "name"
        )
    }

    occupation_essential = list(
        match.get(
            "essential_skills",
            [],
        )
        or []
    )

    occupation_transferable = list(
        match.get(
            "transferable_skills",
            [],
        )
        or []
    )

    potential_capability_gaps = []

    # -----------------------------------------------------
    # Essential potential gaps
    # -----------------------------------------------------

    for skill in occupation_essential:

        name = str(
            skill.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            continue

        if name.lower() in matched_names:
            continue

        importance = to_float(
            skill.get(
                "importance",
                0.0,
            )
        )

        potential_capability_gaps.append(
            {
                "name":
                    name,

                "type":
                    "essential",

                "importance":
                    importance,

                "evidence_status":
                    "not_evidenced",

                "priority":
                    calculate_gap_priority(
                        importance
                    ),
            }
        )

    # -----------------------------------------------------
    # Transferable potential gaps
    # -----------------------------------------------------

    for skill in occupation_transferable:

        name = str(
            skill.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            continue

        if name.lower() in matched_names:
            continue

        importance = to_float(
            skill.get(
                "importance",
                0.0,
            )
        )

        potential_capability_gaps.append(
            {
                "name":
                    name,

                "type":
                    "transferable",

                "importance":
                    importance,

                "evidence_status":
                    "not_evidenced",

                "priority":
                    calculate_gap_priority(
                        importance
                    ),
            }
        )

    # -----------------------------------------------------
    # Sort potential gaps
    #
    # Essential capabilities remain ahead of
    # transferable capabilities.
    #
    # Within each type, higher priority comes first.
    # -----------------------------------------------------

    potential_capability_gaps.sort(
        key=lambda item: (
            0
            if item["type"] == "essential"
            else 1,

            -item["priority"],
        )
    )

    potential_capability_gaps = (
        potential_capability_gaps[:8]
    )

    # -----------------------------------------------------
    # Technical mismatches
    #
    # These are candidate technical skills that were
    # not found in the occupation's O*NET software examples.
    #
    # They are NOT automatically learning requirements.
    # -----------------------------------------------------

    technical_mismatches = []

    for skill in missing_technical:

        if isinstance(
            skill,
            dict,
        ):

            name = str(
                skill.get(
                    "name",
                    "",
                )
            ).strip()

        else:

            name = str(
                skill
            ).strip()

        if name:
            technical_mismatches.append(
                name
            )

    technical_mismatches = (
        technical_mismatches[:8]
    )

    # -----------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------

    recommendation = {

        "title":
            title,

        "final_score":
            final_score,

        "semantic_score":
            semantic_score,

        "capability_score":
            capability_score,

        "essential_score":
            essential_score,

        "transferable_score":
            transferable_score,

        "technical_score":
            technical_score,

        "reasons":
            build_reasons(
                {
                    "semantic_score":
                        semantic_score,

                    "capability_score":
                        capability_score,

                    "essential_score":
                        essential_score,

                    "transferable_score":
                        transferable_score,

                    "technical_score":
                        technical_score,
                }
            ),

        "matched_technical_skills":
            matched_technical,

        "matched_essential_capabilities":
            matched_essential,

        "matched_transferable_capabilities":
            matched_transferable,

        "top_capability_strengths":
            top_capability_strengths,

        "top_technical_strengths":
            top_technical_strengths,

        # New, clearer names
        "potential_capability_gaps":
            potential_capability_gaps,

        "technical_mismatches":
            technical_mismatches,

        # Backwards-compatible names
        "capability_gaps":
            potential_capability_gaps,

        "technical_gaps":
            technical_mismatches,

        # Keep the raw technical mismatch data too
        # for existing consumers.
        "missing_technical_skills":
            missing_technical,
    }

    return recommendation


# =========================================================
# Multiple recommendations
# =========================================================

def build_recommendations(
    matches: List[Dict],
) -> List[Dict]:
    """
    Build recommendations for ranked matches.
    """

    return [
        build_recommendation(
            match
        )
        for match in matches
    ]


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    test_match = {

        "title":
            "Data Scientists",

        "final_score":
            0.517,

        "semantic_score":
            0.613,

        "capability_score":
            0.308,

        "essential_score":
            0.285,

        "transferable_score":
            0.362,

        "technical_score":
            0.535,

        "matched_technical_skills": [

            {
                "name":
                    "Python",

                "onet_name":
                    "Python",

                "specificity":
                    0.506,
            },

            {
                "name":
                    "SQL",

                "onet_name":
                    "Structured query language SQL",

                "specificity":
                    0.493,
            },

            {
                "name":
                    "Pandas",

                "onet_name":
                    "pandas",

                "specificity":
                    0.930,
            },

            {
                "name":
                    "NumPy",

                "onet_name":
                    "NumPy",

                "specificity":
                    1.000,
            },

            {
                "name":
                    "Scikit-learn",

                "onet_name":
                    "Scikit-learn",

                "specificity":
                    0.930,
            },
        ],

        "missing_technical_skills": [
            {
                "name":
                    "Apache Spark",
            },
        ],

        "matched_essential_skills": [

            {
                "name":
                    "Critical Thinking",

                "importance":
                    4.25,

                "evidence_strength":
                    0.810,

                "evidence": [
                    "Compared model performance "
                    "using statistical metrics."
                ],
            },

            {
                "name":
                    "Speaking",

                "importance":
                    3.62,

                "evidence_strength":
                    0.525,

                "evidence": [
                    "Presented recommendations "
                    "to project managers."
                ],
            },
        ],

        "matched_transferable_skills": [

            {
                "name":
                    "Programming",

                "importance":
                    3.75,

                "evidence_strength":
                    0.855,

                "evidence": [
                    "Developed a Python-based "
                    "analytics service."
                ],
            },

            {
                "name":
                    "Complex Problem Solving",

                "importance":
                    3.75,

                "evidence_strength":
                    0.837,

                "evidence": [
                    "Built predictive models "
                    "and investigated bottlenecks."
                ],
            },
        ],

        "essential_skills": [

            {
                "name":
                    "Critical Thinking",

                "importance":
                    4.25,
            },

            {
                "name":
                    "Mathematics",

                "importance":
                    4.38,
            },

            {
                "name":
                    "Speaking",

                "importance":
                    3.62,
            },

            {
                "name":
                    "Writing",

                "importance":
                    3.50,
            },
        ],

        "transferable_skills": [

            {
                "name":
                    "Programming",

                "importance":
                    3.75,
            },

            {
                "name":
                    "Complex Problem Solving",

                "importance":
                    3.75,
            },

            {
                "name":
                    "Operations Analysis",

                "importance":
                    1.50,
            },
        ],
    }

    recommendation = (
        build_recommendation(
            test_match
        )
    )

    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    print()
    print(
        "Occupation:",
        recommendation["title"],
    )

    print(
        "Final score:",
        f"{recommendation['final_score']:.3f}",
    )

    print(
        "Semantic score:",
        f"{recommendation['semantic_score']:.3f}",
    )

    print(
        "Capability score:",
        f"{recommendation['capability_score']:.3f}",
    )

    print(
        "Essential score:",
        f"{recommendation['essential_score']:.3f}",
    )

    print(
        "Transferable score:",
        f"{recommendation['transferable_score']:.3f}",
    )

    print(
        "Technical score:",
        f"{recommendation['technical_score']:.3f}",
    )

    print()
    print("Why this matches:")

    for reason in recommendation[
        "reasons"
    ]:

        print(
            f"- {reason}"
        )

    print()
    print("Top capability strengths:")

    for capability in recommendation[
        "top_capability_strengths"
    ]:

        print(
            f"- {capability['name']}"
            f" | importance: "
            f"{capability['importance']:.2f}"
            f" | evidence: "
            f"{capability['evidence_strength']:.3f}"
        )

        for evidence in capability[
            "evidence"
        ]:

            print(
                f"  Evidence: {evidence}"
            )

    print()
    print("Top technical strengths:")

    for skill in recommendation[
        "top_technical_strengths"
    ]:

        print(
            f"- {skill['name']}"
            f" -> {skill['onet_name']}"
            f" | specificity: "
            f"{skill['specificity']:.3f}"
        )

    print()
    print("Potential capability gaps:")

    if recommendation[
        "potential_capability_gaps"
    ]:

        for gap in recommendation[
            "potential_capability_gaps"
        ]:

            print(
                f"- {gap['name']}"
                f" | type: {gap['type']}"
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

    print()
    print("Technical mismatches:")

    if recommendation[
        "technical_mismatches"
    ]:

        for mismatch in recommendation[
            "technical_mismatches"
        ]:

            print(
                f"- {mismatch}"
            )

    else:

        print(
            "- None"
        )