"""
Map user-provided skills to O*NET essential skills.

This mapping is intentionally explicit for now.
Later, this can be replaced/enhanced with semantic
skill matching or a proper skill taxonomy.
"""

import re


# Common user skill -> O*NET essential skill mappings.
#
# Keys are normalized user-provided skills.
# Values are O*NET essential skill names.
SKILL_MAPPING = {
    # Technical / analytical
    "machine learning": [
        "Critical Thinking",
        "Active Learning",
    ],
    "statistics": [
        "Mathematics",
        "Critical Thinking",
    ],
    "data analysis": [
        "Mathematics",
        "Critical Thinking",
    ],
    "data analytics": [
        "Mathematics",
        "Critical Thinking",
    ],
    "predictive modeling": [
        "Mathematics",
        "Critical Thinking",
    ],

    # Management / leadership
    "leadership": [
        "Coordination",
    ],
    "team leadership": [
        "Coordination",
        "Social Perceptiveness",
    ],
    "team management": [
        "Coordination",
        "Social Perceptiveness",
    ],
    "people management": [
        "Coordination",
        "Social Perceptiveness",
    ],
    "project management": [
        "Coordination",
        "Critical Thinking",
    ],
    "operations management": [
        "Coordination",
        "Critical Thinking",
    ],
    "planning": [
        "Critical Thinking",
    ],
    "strategic planning": [
        "Critical Thinking",
    ],

    # Communication
    "communication": [
        "Speaking",
        "Writing",
        "Active Listening",
    ],
    "verbal communication": [
        "Speaking",
        "Active Listening",
    ],
    "written communication": [
        "Writing",
    ],
    "presentation": [
        "Speaking",
    ],
    "presentations": [
        "Speaking",
    ],

    # People / interpersonal
    "stakeholder management": [
        "Active Listening",
        "Speaking",
        "Social Perceptiveness",
    ],
    "negotiation": [
        "Speaking",
        "Social Perceptiveness",
    ],
    "conflict resolution": [
        "Active Listening",
        "Social Perceptiveness",
    ],

    # Decision making / problem solving
    "decision making": [
        "Critical Thinking",
    ],
    "decision-making": [
        "Critical Thinking",
    ],
    "problem solving": [
        "Critical Thinking",
    ],
    "problem-solving": [
        "Critical Thinking",
    ],

    # Learning
    "learning": [
        "Active Learning",
    ],
    "continuous learning": [
        "Active Learning",
        "Learning Strategies",
    ],
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill for lookup."""

    skill = skill.lower().strip()
    skill = re.sub(r"[^a-z0-9\s-]", "", skill)
    skill = re.sub(r"\s+", " ", skill)

    return skill


def map_user_skills_to_onet(user_skills):
    """
    Map user skills to a set of O*NET essential skills.

    Unknown skills are ignored for now.

    Returns:
        set[str]: O*NET essential skill names.
    """

    mapped_skills = set()

    for skill in user_skills:
        normalized = normalize_skill(skill)

        mapped_skills.update(
            SKILL_MAPPING.get(normalized, [])
        )

    return mapped_skills


if __name__ == "__main__":

    technical_profile = [
        "Python",
        "Machine Learning",
        "Statistics",
        "Data Analysis",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
    ]

    management_profile = [
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

    print("Technical profile mapping:")
    for skill in sorted(
        map_user_skills_to_onet(technical_profile)
    ):
        print(f"- {skill}")

    print()

    print("Management profile mapping:")
    for skill in sorted(
        map_user_skills_to_onet(management_profile)
    ):
        print(f"- {skill}")