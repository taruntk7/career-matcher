# =========================================================
# Evaluation benchmark
# =========================================================

EVALUATION_PROFILES = [

    {
        "profile_id": "data_001",

        "description": (
            "Data science candidate with Python, SQL, pandas, NumPy, "
            "scikit-learn, statistics, machine learning and analytics "
            "experience."
        ),

        "explicit_skills": [
            "Python",
            "SQL",
            "Pandas",
            "NumPy",
            "Scikit-learn",
        ],

        "capabilities": [
            {
                "name": "Programming",
                "evidence_strength": 0.90,
                "confidence": 0.90,
                "evidence_type": "explicit",
                "evidence": "Developed data analysis and machine learning solutions using Python."
            },
            {
                "name": "Mathematics",
                "evidence_strength": 0.80,
                "confidence": 0.80,
                "evidence_type": "explicit",
                "evidence": "Applied statistics and mathematical modeling."
            },
            {
                "name": "Critical Thinking",
                "evidence_strength": 0.85,
                "confidence": 0.85,
                "evidence_type": "explicit",
                "evidence": "Evaluated models and analyzed data to solve problems."
            },
        ],

        "relevant_occupations": [
            "Data Scientists",
            "Business Intelligence Analysts",
        ],
    },


    {
        "profile_id": "software_001",

        "description": (
            "Software development candidate with Java, Python, "
            "JavaScript, React, Node.js, REST APIs, databases "
            "and application development experience."
        ),

        "explicit_skills": [
            "Java",
            "Python",
            "JavaScript",
            "React",
            "Node.js",
            "REST APIs",
        ],

        "capabilities": [
            {
                "name": "Programming",
                "evidence_strength": 0.95,
                "confidence": 0.95,
                "evidence_type": "explicit",
                "evidence": "Developed software applications using multiple programming languages."
            },
            {
                "name": "Technology Design",
                "evidence_strength": 0.85,
                "confidence": 0.85,
                "evidence_type": "explicit",
                "evidence": "Designed and implemented application architectures."
            },
            {
                "name": "Systems Analysis",
                "evidence_strength": 0.80,
                "confidence": 0.80,
                "evidence_type": "explicit",
                "evidence": "Analyzed application and system requirements."
            },
        ],

        "relevant_occupations": [
            "Software Developers",
            "Web Developers",
        ],
    },


    {
        "profile_id": "database_001",

        "description": (
            "Database-focused candidate with SQL, PostgreSQL, "
            "MySQL, database design, data modeling and database "
            "administration experience."
        ),

        "explicit_skills": [
            "SQL",
            "PostgreSQL",
            "MySQL",
        ],

        "capabilities": [
            {
                "name": "Programming",
                "evidence_strength": 0.75,
                "confidence": 0.75,
                "evidence_type": "explicit",
                "evidence": "Used SQL to manage and query relational databases."
            },
            {
                "name": "Systems Analysis",
                "evidence_strength": 0.90,
                "confidence": 0.90,
                "evidence_type": "explicit",
                "evidence": "Designed database structures and analyzed data systems."
            },
            {
                "name": "Quality Control Analysis",
                "evidence_strength": 0.75,
                "confidence": 0.75,
                "evidence_type": "explicit",
                "evidence": "Validated database records and maintained data quality."
            },
        ],

        "relevant_occupations": [
            "Database Administrators",
            "Database Architects",
        ],
    },


    {
        "profile_id": "management_001",

        "description": (
            "Technology management candidate with project "
            "coordination, team leadership, planning, stakeholder "
            "communication and IT delivery experience."
        ),

        "explicit_skills": [
            "Project Management",
            "Leadership",
            "Planning",
        ],

        "capabilities": [
            {
                "name": "Coordination",
                "evidence_strength": 0.90,
                "confidence": 0.90,
                "evidence_type": "explicit",
                "evidence": "Coordinated project activities and cross-functional teams."
            },
            {
                "name": "Management of Personnel Resources",
                "evidence_strength": 0.90,
                "confidence": 0.90,
                "evidence_type": "explicit",
                "evidence": "Led teams and delegated responsibilities."
            },
            {
                "name": "Judgment and Decision Making",
                "evidence_strength": 0.85,
                "confidence": 0.85,
                "evidence_type": "explicit",
                "evidence": "Made project and delivery decisions based on business requirements."
            },
            {
                "name": "Speaking",
                "evidence_strength": 0.80,
                "confidence": 0.80,
                "evidence_type": "explicit",
                "evidence": "Presented project information to stakeholders."
            },
        ],

        "relevant_occupations": [
            "Computer and Information Systems Managers",
            "Information Technology Project Managers",
        ],
    },

]