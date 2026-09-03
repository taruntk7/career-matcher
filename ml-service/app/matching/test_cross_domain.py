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


# =========================================================
# Test profiles
# =========================================================

TEST_PROFILES = {

    "PURE SOFTWARE": {

        "profile": """
        Software engineering and application development.
        Strong experience with Python, Java, C++, JavaScript,
        TypeScript, React, Node.js, Express, REST APIs,
        WebSockets, PostgreSQL, MongoDB, Git, Docker,
        Linux, software architecture, debugging, testing,
        and building full-stack applications.
        """,

        "skills": [
            "Python",
            "Java",
            "C++",
            "JavaScript",
            "TypeScript",
            "React",
            "Node.js",
            "Express",
            "REST APIs",
            "WebSockets",
            "PostgreSQL",
            "MongoDB",
            "Git",
            "Docker",
            "Linux",
        ],

        "capabilities": [
            {
                "name": "Programming",
                "confidence": 0.98,
                "evidence_strength": 0.90,
                "evidence": [
                    "Built full-stack applications "
                    "using multiple programming languages."
                ],
            },
            {
                "name": "Systems Analysis",
                "confidence": 0.90,
                "evidence_strength": 0.80,
                "evidence": [
                    "Designed APIs, databases, "
                    "and application architecture."
                ],
            },
            {
                "name": "Technology Design",
                "confidence": 0.92,
                "evidence_strength": 0.83,
                "evidence": [
                    "Designed software architecture "
                    "and full-stack systems."
                ],
            },
            {
                "name": "Complex Problem Solving",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Debugged and resolved technical "
                    "issues during application development."
                ],
            },
            {
                "name": "Systems Evaluation",
                "confidence": 0.85,
                "evidence_strength": 0.75,
                "evidence": [
                    "Tested and evaluated software systems."
                ],
            },
            {
                "name": "Troubleshooting",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Diagnosed and resolved software problems."
                ],
            },
        ],
    },


    "PURE DATA SCIENCE": {

        "profile": """
        Data science, machine learning, statistics, data
        analysis, predictive modeling, Python, SQL, pandas,
        NumPy, scikit-learn, feature engineering, model
        evaluation, forecasting, experimentation, and
        working with large datasets.
        """,

        "skills": [
            "Python",
            "SQL",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Statistics",
            "Data Analysis",
        ],

        "capabilities": [
            {
                "name": "Critical Thinking",
                "confidence": 0.97,
                "evidence_strength": 0.87,
                "evidence": [
                    "Compared model performance and "
                    "iteratively improved feature selection."
                ],
            },
            {
                "name": "Complex Problem Solving",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Built predictive models and "
                    "investigated complex data problems."
                ],
            },
            {
                "name": "Programming",
                "confidence": 0.97,
                "evidence_strength": 0.87,
                "evidence": [
                    "Implemented analytical and machine "
                    "learning workflows using Python."
                ],
            },
            {
                "name": "Operations Analysis",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Analyzed datasets to identify trends "
                    "and operational patterns."
                ],
            },
            {
                "name": "Systems Evaluation",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Evaluated predictive models using "
                    "statistical performance metrics."
                ],
            },
        ],
    },


    "PURE MANAGEMENT": {

        "profile": """
        Business and team management with strong experience
        in leadership, project management, stakeholder
        management, planning, negotiation, decision making,
        coordinating teams, managing personnel, and
        improving business operations.
        """,

        "skills": [
            "Leadership",
            "Project Management",
            "Team Management",
            "Stakeholder Management",
            "Planning",
            "Negotiation",
            "Decision Making",
            "Operations Management",
        ],

        "capabilities": [
            {
                "name": "Coordination",
                "confidence": 0.96,
                "evidence_strength": 0.86,
                "evidence": [
                    "Coordinated teams, stakeholders, "
                    "projects, and business activities."
                ],
            },
            {
                "name": "Management of Personnel Resources",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Led teams and delegated "
                    "responsibilities to personnel."
                ],
            },
            {
                "name": "Judgment and Decision Making",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Made organizational decisions "
                    "and prioritized business issues."
                ],
            },
            {
                "name": "Negotiation",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Negotiated with stakeholders "
                    "and business partners."
                ],
            },
            {
                "name": "Social Perceptiveness",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Managed stakeholder relationships "
                    "and team interactions."
                ],
            },
            {
                "name": "Time Management",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Planned projects and managed deadlines."
                ],
            },
        ],
    },


    "PURE LEGAL": {

        "profile": """
        Legal practice involving legal research, case
        analysis, contract drafting, legal writing, client
        counseling, negotiation, document review, legal
        argumentation, and advising clients.
        """,

        "skills": [
            "Legal Research",
            "Case Analysis",
            "Contract Drafting",
            "Legal Writing",
            "Client Counseling",
            "Negotiation",
        ],

        "capabilities": [
            {
                "name": "Reading Comprehension",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Performed legal research and "
                    "reviewed case law and documents."
                ],
            },
            {
                "name": "Critical Thinking",
                "confidence": 0.96,
                "evidence_strength": 0.86,
                "evidence": [
                    "Analyzed cases and developed "
                    "legal arguments."
                ],
            },
            {
                "name": "Complex Problem Solving",
                "confidence": 0.94,
                "evidence_strength": 0.85,
                "evidence": [
                    "Analyzed complex legal matters "
                    "and formulated solutions."
                ],
            },
            {
                "name": "Writing",
                "confidence": 0.96,
                "evidence_strength": 0.86,
                "evidence": [
                    "Drafted contracts and legal documents."
                ],
            },
            {
                "name": "Negotiation",
                "confidence": 0.96,
                "evidence_strength": 0.86,
                "evidence": [
                    "Negotiated with clients and "
                    "opposing parties."
                ],
            },
            {
                "name": "Persuasion",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Developed arguments and advised "
                    "clients on legal positions."
                ],
            },
        ],
    },
    
    "TECHNICAL + BUSINESS": {

        "profile": """
        Technical and business-oriented professional with
        experience in Python, machine learning, SQL, data
        analysis, project management, stakeholder management,
        communication, decision making, coordinating teams,
        and translating business requirements into technical
        solutions.
        """,

        "skills": [
            "Python",
            "Machine Learning",
            "SQL",
            "Data Analysis",
            "Project Management",
            "Stakeholder Management",
            "Communication",
            "Decision Making",
        ],

        "capabilities": [
            {
                "name": "Programming",
                "confidence": 0.95,
                "evidence_strength": 0.86,
                "evidence": [
                    "Developed analytical solutions using Python."
                ],
            },
            {
                "name": "Complex Problem Solving",
                "confidence": 0.94,
                "evidence_strength": 0.85,
                "evidence": [
                    "Built machine-learning solutions for business problems."
                ],
            },
            {
                "name": "Critical Thinking",
                "confidence": 0.93,
                "evidence_strength": 0.84,
                "evidence": [
                    "Analyzed data and evaluated solutions."
                ],
            },
            {
                "name": "Coordination",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Coordinated technical and business stakeholders."
                ],
            },
            {
                "name": "Judgment and Decision Making",
                "confidence": 0.90,
                "evidence_strength": 0.81,
                "evidence": [
                    "Made project and business decisions."
                ],
            },
            {
                "name": "Operations Analysis",
                "confidence": 0.88,
                "evidence_strength": 0.79,
                "evidence": [
                    "Used data analysis to improve business processes."
                ],
            },
        ],
    },
}


# =========================================================
# Display helpers
# =========================================================

def print_profile(
    name,
    profile,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("Skills:")

    for skill in profile["skills"]:
        print(
            f"- {skill}"
        )

    print()
    print("Capabilities:")

    for capability in profile["capabilities"]:

        print(
            f"- {capability['name']}"
            f" | evidence strength: "
            f"{capability['evidence_strength']:.3f}"
        )


def print_results(
    matches,
):
    print()
    print("Top matches:")
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


# =========================================================
# Test one profile
# =========================================================

def run_profile(
    name,
    profile,
    model,
    collection,
    occupations,
):

    print_profile(
        name,
        profile,
    )

    # -----------------------------------------------------
    # Semantic retrieval
    # -----------------------------------------------------

    matches = match_occupations(
        model,
        collection,
        profile["profile"],
        top_k=50,
    )

    # -----------------------------------------------------
    # Enrichment
    # -----------------------------------------------------

    matches = enrich_matches(
        matches,
        profile["skills"],
        profile["capabilities"],
        occupations,
    )

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    matches = rank_matches(
        matches
    )

    print_results(
        matches
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 70)
    print("CROSS-DOMAIN VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load everything once
    # -----------------------------------------------------

    occupations = (
        build_occupation_dataset()
    )

    model, collection = (
        load_matcher()
    )

    # -----------------------------------------------------
    # Run all profiles
    # -----------------------------------------------------

    for name, profile in (
        TEST_PROFILES.items()
    ):

        run_profile(
            name,
            profile,
            model,
            collection,
            occupations,
        )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()