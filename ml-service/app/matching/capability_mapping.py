import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    """Normalize text for simple rule-based matching."""

    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ---------------------------------------------------------
# Capability rules
# ---------------------------------------------------------
#
# These are intentionally conservative.
#
# We distinguish:
#
#   explicit skill
#       Something directly stated by the user.
#
#   inferred capability
#       A capability reasonably supported by the
#       user's skills/profile.
#
# We do NOT infer every soft skill from every technical
# skill.
# ---------------------------------------------------------

CAPABILITY_RULES = {

    "Programming": {
        "skills": {
            "python",
            "java",
            "c",
            "c++",
            "c#",
            "javascript",
            "typescript",
            "go",
            "ruby",
            "php",
            "r",
            "scala",
            "kotlin",
            "swift",
            "sql",
            "structured query language sql",
            "programming",
            "software development",
        }
    },

    "Complex Problem Solving": {
        "skills": {
            "machine learning",
            "data analysis",
            "data analytics",
            "predictive modeling",
            "software development",
            "programming",
            "debugging",
            "troubleshooting",
            "research",
            "legal research",
            "case analysis",
            "legal analysis",
            "optimization",
        }
    },

    "Critical Thinking": {
        "skills": {
            "data analysis",
            "data analytics",
            "statistics",
            "research",
            "legal research",
            "case analysis",
            "legal analysis",
            "machine learning",
            "predictive modeling",
            "problem solving",
            "problem-solving",
            "debugging",
            "troubleshooting",
        }
    },
    "Writing": {
        "skills": {
            "legal research",
            "contract drafting",
            "legal writing",
            "technical writing",
            "report writing",
            "documentation",
        }
    },

    "Reading Comprehension": {
        "skills": {
            "legal research",
            "case analysis",
            "research",
            "literature review",
        }
    },

    "Judgment and Decision Making": {
        "skills": {
            "decision making",
            "decision-making",
            "case analysis",
            "legal analysis",
            "risk analysis",
            "strategy",
            "strategic planning",
        }
    },

    "Systems Analysis": {
        "skills": {
            "software development",
            "programming",
            "systems analysis",
            "system design",
            "data engineering",
            "database management",
        }
    },

    "Systems Evaluation": {
        "skills": {
            "software testing",
            "testing",
            "debugging",
            "quality assurance",
            "qa",
            "model validation",
            "validation",
        }
    },

    "Operations Analysis": {
        "skills": {
            "data analysis",
            "data analytics",
            "business analysis",
            "operations analysis",
            "process improvement",
        }
    },

    "Judgment and Decision Making": {
        "skills": {
            "decision making",
            "decision-making",
            "business analysis",
            "risk analysis",
            "strategy",
            "strategic planning",
        }
    },

    "Coordination": {
        "skills": {
            "team leadership",
            "team management",
            "project management",
            "coordination",
            "operations management",
            "stakeholder management",
        }
    },

    "Negotiation": {
        "skills": {
            "negotiation",
            "stakeholder management",
            "contract negotiation",
        }
    },

    "Persuasion": {
        "skills": {
            "sales",
            "business development",
            "negotiation",
            "marketing",
            "stakeholder management",
        }
    },

    "Time Management": {
        "skills": {
            "project management",
            "planning",
            "project planning",
            "time management",
        }
    },

    "Management of Personnel Resources": {
        "skills": {
            "team leadership",
            "team management",
            "people management",
            "leadership",
            "human resource management",
        }
    },

    "Social Perceptiveness": {
        "skills": {
            "stakeholder management",
            "customer service",
            "client management",
            "team management",
            "people management",
        }
    },
}


def normalize_skill(skill: str) -> str:
    return normalize_text(skill)


def infer_capabilities(
    user_skills: List[str],
) -> List[Dict]:
    """
    Infer O*NET-style capabilities from explicit user skills.

    Evidence strength is determined by how many independent
    explicit skills support the capability.

    1 supporting skill  -> 0.60
    2 supporting skills -> 0.75
    3+ supporting skills -> 0.85

    These are evidence-strength heuristics, not semantic
    similarity scores.
    """

    normalized_skills = {
        normalize_skill(skill): skill
        for skill in user_skills
    }

    capabilities = []

    for capability, rule in CAPABILITY_RULES.items():

        evidence = []

        for normalized, original in normalized_skills.items():

            if normalized in rule["skills"]:
                evidence.append(original)

        if not evidence:
            continue

        evidence_count = len(evidence)

        if evidence_count == 1:
            evidence_strength = 0.60

        elif evidence_count == 2:
            evidence_strength = 0.75

        else:
            evidence_strength = 0.85

        capabilities.append(
            {
                "name": capability,
                "evidence": evidence,
                "confidence": "inferred",
                "evidence_strength": evidence_strength,
            }
        )

    return capabilities

def get_capability_names(
    user_skills: List[str],
) -> List[str]:
    """
    Return only inferred capability names.
    """

    return [
        capability["name"]
        for capability in infer_capabilities(user_skills)
    ]


if __name__ == "__main__":

    test_profiles = {

        "Data Science": [
            "Python",
            "Machine Learning",
            "Statistics",
            "Data Analysis",
            "SQL",
            "Pandas",
            "NumPy",
            "Scikit-learn",
        ],

        "Management": [
            "Leadership",
            "Project Management",
            "Communication",
            "Team Management",
            "Planning",
            "Stakeholder Management",
            "Negotiation",
            "Decision Making",
            "Operations Management",
        ],

        "Law": [
            "Legal Research",
            "Contract Drafting",
            "Case Analysis",
            "Negotiation",
            "Client Counseling",
        ],
    }

    for name, skills in test_profiles.items():

        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

        print()
        print("Explicit skills:")

        for skill in skills:
            print(f"- {skill}")

        print()
        print("Inferred capabilities:")

        capabilities = infer_capabilities(skills)

        for capability in capabilities:

            print(
                f"- {capability['name']}"
                f" | evidence: "
                f"{', '.join(capability['evidence'])}"
                f" | confidence: "
                f"{capability['confidence']}"
            )