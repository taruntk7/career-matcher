# --------------------------------------------------
# roadmap.sh mappings
# --------------------------------------------------

ROADMAPS = {

    "Data Scientists": {
        "name": "AI and Data Scientist",
        "url": "https://roadmap.sh/ai-data-scientist",

        # Curated from the official roadmap.sh roadmap.
        # This is used as structured learning context
        # for personalized guidance.
        "topics": [
            {
                "name": "Mathematics",
                "subtopics": [
                    "Linear Algebra",
                    "Calculus",
                    "Mathematical Analysis",
                ],
            },

            {
                "name": "Statistics",
                "subtopics": [
                    "Statistics",
                    "Central Limit Theorem",
                    "Hypothesis Testing",
                    "Probability",
                    "Sampling",
                    "A/B Testing",
                    "Ratio Metrics",
                ],
            },

            {
                "name": "Econometrics",
                "subtopics": [
                    "Regression",
                    "Time Series",
                    "Fitting Distributions",
                ],
            },

            {
                "name": "Coding",
                "subtopics": [
                    "Python",
                    "Data Structures and Algorithms",
                    "SQL",
                ],
            },

            {
                "name": "Exploratory Data Analysis",
                "subtopics": [
                    "Data Understanding",
                    "Data Analysis",
                    "Data Visualization",
                ],
            },

            {
                "name": "Machine Learning",
                "subtopics": [
                    "Supervised Learning",
                    "Unsupervised Learning",
                    "Ensemble Methods",
                    "Neural Networks",
                ],
            },

            {
                "name": "Deep Learning",
                "subtopics": [
                    "Fully Connected Networks",
                    "CNN",
                    "RNN",
                    "LSTM",
                    "Transformers",
                    "Transfer Learning",
                ],
            },

            {
                "name": "MLOps",
                "subtopics": [
                    "Model Deployment",
                    "Deployment Models",
                    "CI/CD",
                ],
            },
        ],
    },


    "Data Analysts": {
        "name": "Data Analyst",
        "url": "https://roadmap.sh/data-analyst",

        "topics": [
            {
                "name": "Data Analysis",
                "subtopics": [
                    "Data Cleaning",
                    "Data Transformation",
                    "Exploratory Data Analysis",
                ],
            },
            {
                "name": "SQL",
                "subtopics": [
                    "SQL",
                    "Querying Data",
                    "Aggregations",
                    "Joins",
                ],
            },
            {
                "name": "Statistics",
                "subtopics": [
                    "Descriptive Statistics",
                    "Probability",
                    "Hypothesis Testing",
                ],
            },
            {
                "name": "Data Visualization",
                "subtopics": [
                    "Charts",
                    "Dashboards",
                    "Data Presentation",
                ],
            },
        ],
    },


    "Business Intelligence Analysts": {
        "name": "BI Analyst",
        "url": "https://roadmap.sh/bi-analyst",

        "topics": [
            {
                "name": "SQL",
                "subtopics": [
                    "SQL",
                    "Data Querying",
                    "Data Aggregation",
                ],
            },
            {
                "name": "Data Analysis",
                "subtopics": [
                    "Data Cleaning",
                    "Exploratory Analysis",
                    "Business Analysis",
                ],
            },
            {
                "name": "Data Visualization",
                "subtopics": [
                    "Dashboards",
                    "Reporting",
                    "Data Visualization",
                ],
            },
        ],
    },


    "Data Engineers": {
        "name": "Data Engineer",
        "url": "https://roadmap.sh/data-engineer",

        "topics": [
            {
                "name": "Programming",
                "subtopics": [
                    "Python",
                    "SQL",
                ],
            },
            {
                "name": "Databases",
                "subtopics": [
                    "Relational Databases",
                    "Database Design",
                ],
            },
            {
                "name": "Data Processing",
                "subtopics": [
                    "Data Pipelines",
                    "ETL",
                    "Data Processing",
                ],
            },
            {
                "name": "Big Data",
                "subtopics": [
                    "Distributed Data Processing",
                    "Data Infrastructure",
                ],
            },
        ],
    },


    "Machine Learning Engineers": {
        "name": "Machine Learning",
        "url": "https://roadmap.sh/machine-learning",

        "topics": [
            {
                "name": "Mathematics and Statistics",
                "subtopics": [
                    "Linear Algebra",
                    "Probability",
                    "Statistics",
                ],
            },
            {
                "name": "Machine Learning",
                "subtopics": [
                    "Supervised Learning",
                    "Unsupervised Learning",
                    "Model Evaluation",
                    "Ensemble Methods",
                ],
            },
            {
                "name": "Deep Learning",
                "subtopics": [
                    "Neural Networks",
                    "CNN",
                    "RNN",
                    "Transformers",
                ],
            },
            {
                "name": "MLOps",
                "subtopics": [
                    "Model Deployment",
                    "Monitoring",
                    "CI/CD",
                ],
            },
        ],
    },


    "AI Engineers": {
        "name": "AI Engineer",
        "url": "https://roadmap.sh/ai-engineer",

        "topics": [
            {
                "name": "Programming",
                "subtopics": [
                    "Python",
                    "Data Structures",
                ],
            },
            {
                "name": "Machine Learning",
                "subtopics": [
                    "Machine Learning",
                    "Model Evaluation",
                ],
            },
            {
                "name": "Deep Learning",
                "subtopics": [
                    "Neural Networks",
                    "Transformers",
                ],
            },
            {
                "name": "AI Systems",
                "subtopics": [
                    "Model Deployment",
                    "AI Applications",
                ],
            },
        ],
    },
}


# --------------------------------------------------
# Helper
# --------------------------------------------------

def get_roadmap(occupation):
    """
    Return the roadmap mapping for an occupation.

    Returns None when no roadmap is mapped.
    """

    return ROADMAPS.get(occupation)