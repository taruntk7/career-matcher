import os

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    "ml-service",
    ".env",
)

load_dotenv(ENV_PATH)

MODEL_NAME = "openai/gpt-oss-120b"


# --------------------------------------------------
# Roadmap mapping
# --------------------------------------------------

try:
    # When imported as rag.generator
    from rag.roadmap_mapper import get_roadmap

except ImportError:
    # When generator.py is run directly
    from roadmap_mapper import get_roadmap


# --------------------------------------------------
# Groq client
# --------------------------------------------------

api_key = os.getenv(
    "GROQ_API_KEY"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables."
    )

client = Groq(
    api_key=api_key
)


# --------------------------------------------------
# Context formatting
# --------------------------------------------------

def format_occupation_context(document):
    """
    Format the exact retrieved O*NET occupation document.
    """

    if not document:
        return (
            "No occupation document was retrieved."
        )

    metadata = document.get(
        "metadata",
        {}
    )

    document_text = document.get(
        "document",
        ""
    )

    title = metadata.get(
        "title",
        "Unknown Occupation"
    )

    code = metadata.get(
        "code",
        "Unknown Code"
    )

    return (
        f"Occupation: {title}\n"
        f"O*NET Code: {code}\n\n"
        f"{document_text}"
    )


def format_gaps(gaps):
    """
    Format capability gaps identified by the
    recommendation system.
    """

    if not gaps:
        return (
            "No potential capability gaps identified."
        )

    lines = []

    for gap in gaps:

        name = gap.get(
            "name",
            "Unknown"
        )

        gap_type = gap.get(
            "type",
            "unknown"
        )

        importance = gap.get(
            "importance",
            0
        )

        priority = gap.get(
            "priority",
            0
        )

        status = gap.get(
            "evidence_status",
            "unknown"
        )

        lines.append(
            f"- {name} "
            f"(type: {gap_type}, "
            f"importance: {importance:.2f}, "
            f"priority: {priority:.3f}, "
            f"status: {status})"
        )

    return "\n".join(lines)


def format_capability_strengths(strengths):
    """
    Format candidate capability strengths.
    """

    if not strengths:
        return (
            "No capability strengths identified."
        )

    lines = []

    for capability in strengths:

        name = capability.get(
            "name",
            "Unknown"
        )

        importance = capability.get(
            "importance",
            0
        )

        evidence_strength = capability.get(
            "evidence_strength",
            0
        )

        lines.append(
            f"- {name} "
            f"(O*NET importance: {importance:.2f}, "
            f"evidence strength: {evidence_strength:.3f})"
        )

        evidence = capability.get(
            "evidence",
            []
        )

        if isinstance(evidence, str):
            evidence = [evidence]

        for item in evidence:

            lines.append(
                f"  Evidence: {item}"
            )

    return "\n".join(lines)


def format_technical_strengths(strengths):
    """
    Format technical skills matched to O*NET
    associated software/technology.
    """

    if not strengths:
        return (
            "No technical strengths identified."
        )

    lines = []

    for skill in strengths:

        name = skill.get(
            "name",
            "Unknown"
        )

        onet_name = skill.get(
            "onet_name",
            "Unknown"
        )

        specificity = skill.get(
            "specificity",
            0
        )

        lines.append(
            f"- {name} "
            f"(O*NET associated technology: {onet_name}, "
            f"specificity: {specificity:.3f})"
        )

    return "\n".join(lines)


def format_technical_mismatches(mismatches):
    """
    Format candidate technical skills that were not
    matched to the occupation's associated technology.
    """

    if not mismatches:
        return (
            "No technical mismatches identified."
        )

    lines = []

    for skill in mismatches:

        if isinstance(skill, dict):

            name = skill.get(
                "name",
                "Unknown"
            )

        else:

            name = str(skill)

        lines.append(
            f"- {name}"
        )

    return "\n".join(lines)


def format_roadmap_context(roadmap):
    """
    Format the structured roadmap.sh information.

    The roadmap topics are passed explicitly to the LLM
    rather than only passing the roadmap URL.
    """

    if not roadmap:
        return (
            "No roadmap.sh roadmap is currently mapped "
            "to this occupation."
        )

    name = roadmap.get(
        "name",
        "Unknown roadmap"
    )

    url = roadmap.get(
        "url",
        ""
    )

    topics = roadmap.get(
        "topics",
        []
    )

    lines = [
        f"Roadmap name: {name}",
        f"Roadmap URL: {url}",
        "",
        "Roadmap topics:",
    ]

    if not topics:

        lines.append(
            "- No structured topics available."
        )

    else:

        for topic in topics:

            topic_name = topic.get(
                "name",
                "Unknown"
            )

            subtopics = topic.get(
                "subtopics",
                []
            )

            lines.append(
                f"- {topic_name}"
            )

            if subtopics:

                lines.append(
                    "  Subtopics: "
                    + ", ".join(subtopics)
                )

    return "\n".join(lines)


# --------------------------------------------------
# Prompt
# --------------------------------------------------

def build_prompt(
    occupation,
    question,
    occupation_document,
    capability_gaps,
    technical_mismatches,
    capability_strengths,
    technical_strengths,
    roadmap=None,
):
    """
    Build a grounded career-guidance prompt using:

    - O*NET occupation information
    - candidate strengths
    - candidate potential gaps
    - technical matches/mismatches
    - structured roadmap.sh learning topics
    """

    occupation_context = (
        format_occupation_context(
            occupation_document
        )
    )

    gap_context = format_gaps(
        capability_gaps
    )

    technical_mismatch_context = (
        format_technical_mismatches(
            technical_mismatches
        )
    )

    capability_strength_context = (
        format_capability_strengths(
            capability_strengths
        )
    )

    technical_strength_context = (
        format_technical_strengths(
            technical_strengths
        )
    )

    roadmap_context = (
        format_roadmap_context(
            roadmap
        )
    )

    return f"""
You are an AI career guidance assistant.

The recommendation system has already selected this occupation
as the candidate's recommended career.

RECOMMENDED OCCUPATION
======================
{occupation}


CANDIDATE CAPABILITY STRENGTHS
==============================
These are capabilities for which the resume contains supporting evidence.

{capability_strength_context}


CANDIDATE TECHNICAL STRENGTHS
=============================
These are technical skills from the resume that matched the
recommended occupation's associated O*NET software/technology list.

{technical_strength_context}


CANDIDATE POTENTIAL CAPABILITY GAPS
===================================
These are occupation capabilities for which the resume did not
provide sufficient evidence.

{gap_context}


CANDIDATE TECHNICAL MISMATCHES
==============================
These are technical skills present in the candidate's profile that
were not matched to the recommended occupation's associated O*NET
software/technology list.

{technical_mismatch_context}


O*NET OCCUPATION CONTEXT
========================
The following is the exact O*NET occupation document retrieved for
the recommended occupation.

{occupation_context}


ROADMAP.SH LEARNING ROADMAP
===========================
The following structured learning roadmap has been mapped to the
recommended occupation:

{roadmap_context}


IMPORTANT ROADMAP INTERPRETATION
================================

The roadmap.sh information above is a structured learning-path
reference.

Use the supplied roadmap topics directly.

Do NOT invent additional roadmap sections and claim that they
came from roadmap.sh.

Do NOT treat roadmap.sh topics as official O*NET requirements.

Instead:

1. Use O*NET to understand the occupation.
2. Use the candidate profile to understand current strengths.
3. Use the identified potential capability gaps to determine
   development priorities.
4. Use roadmap.sh topics to organize the learning sequence.
5. Personalize the sequence around what the candidate already knows.

The roadmap is a reference structure, not a mandatory checklist.


USER QUESTION
=============
{question}


IMPORTANT INTERPRETATION RULES
==============================

1. SOURCE OF OCCUPATION FACTS

Use the supplied O*NET occupation document as the source for
occupation-related facts.

Do not present information as an O*NET fact unless it is supported
by the supplied occupation document.


2. CAPABILITY GAPS

A potential capability gap means that the resume did not provide
enough evidence for that capability.

It does NOT prove that the candidate lacks the capability.

Use wording such as:

- "the system did not find evidence of..."
- "this is an area to strengthen"
- "consider developing..."

Do not say:

- "you do not have this skill"
- "you are missing this skill"

unless the supplied information explicitly proves that.


3. TECHNICAL STRENGTHS

A technical strength means the candidate's technical skill was
successfully matched to an associated O*NET technology.

Treat these as existing strengths.

Do not recommend learning them from scratch.


4. TECHNICAL MISMATCHES

A technical mismatch means only that the candidate's technical
skill was not matched to this occupation's associated O*NET
software/technology list.

It does NOT prove that:

- the skill is useless,
- the skill is incorrect,
- the candidate should stop using it,
- or the skill is unrelated to the career.

Do not automatically turn technical mismatches into learning gaps.


5. DO NOT INFER ABSENCE

Do not claim that a candidate lacks a skill merely because it
does not appear in the supplied strengths or mismatch lists.

Only use explicit evidence supplied by the recommendation system.


6. EXISTING STRENGTHS COME FIRST

When the candidate already demonstrates a relevant capability or
technical skill, build on that foundation.


7. ROADMAP.SH

Use only the roadmap topics supplied above when referring to
roadmap.sh.

Do not invent roadmap.sh sections.

You may combine roadmap topics into a personalized sequence.


8. AI-GENERATED RECOMMENDATIONS

You may suggest practical:

- exercises,
- projects,
- study methods,
- learning sequences,
- technologies,

but clearly distinguish these from official O*NET or roadmap.sh
content.


9. AVOID GENERIC ROADMAPS

Do not reproduce every roadmap topic.

Select only the topics relevant to:

- the recommended occupation,
- the candidate's strengths,
- the candidate's potential gaps,
- and the user's question.


10. PRIORITIZE

Prioritize development areas using:

- occupation relevance,
- candidate potential gaps,
- O*NET importance values,
- roadmap relevance,
- existing candidate strengths.


11. DO NOT OVERSTATE

Do not claim that improving a particular skill will mathematically
increase the recommendation score unless the recommendation system
explicitly provides such a relationship.


12. RESPONSE STYLE

Give practical, personalized, prioritized guidance.

Prefer a focused learning roadmap over a generic career article.

Answer the user's question directly.
""".strip()


# --------------------------------------------------
# Generation
# --------------------------------------------------

def generate_answer(
    occupation,
    question,
    occupation_document,
    capability_gaps=None,
    technical_mismatches=None,
    capability_strengths=None,
    technical_strengths=None,
):
    """
    Generate personalized career guidance using:

    - recommended occupation
    - capability gaps
    - technical mismatches
    - capability strengths
    - technical strengths
    - exact O*NET occupation document
    - structured roadmap.sh topics
    """

    capability_gaps = (
        capability_gaps or []
    )

    technical_mismatches = (
        technical_mismatches or []
    )

    capability_strengths = (
        capability_strengths or []
    )

    technical_strengths = (
        technical_strengths or []
    )

    if not occupation:

        raise ValueError(
            "Recommended occupation is required."
        )

    if not occupation_document:

        return (
            "I could not retrieve the O*NET document for "
            "the recommended occupation."
        )

    # --------------------------------------------------
    # Retrieve roadmap mapping
    # --------------------------------------------------

    roadmap = get_roadmap(
        occupation
    )

    # --------------------------------------------------
    # Build grounded prompt
    # --------------------------------------------------

    prompt = build_prompt(
        occupation=occupation,
        question=question,
        occupation_document=occupation_document,
        capability_gaps=capability_gaps,
        technical_mismatches=technical_mismatches,
        capability_strengths=capability_strengths,
        technical_strengths=technical_strengths,
        roadmap=roadmap,
    )

    # --------------------------------------------------
    # Groq generation
    # --------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a practical and factual career "
                    "guidance assistant. Clearly separate "
                    "O*NET facts, recommendation-system findings, "
                    "roadmap.sh references, and AI-generated "
                    "suggestions."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.2,
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# --------------------------------------------------
# Standalone test
# --------------------------------------------------

if __name__ == "__main__":

    from retriever import (
        load_retriever,
        retrieve_occupation,
    )

    model, collection = (
        load_retriever()
    )

    occupation = "Data Scientists"

    question = (
        "What should I focus on learning next "
        "to become more suitable for this career?"
    )

    # --------------------------------------------------
    # Retrieve exact occupation document
    # --------------------------------------------------

    occupation_document = (
        retrieve_occupation(
            collection=collection,
            occupation_title=occupation,
        )
    )

    # --------------------------------------------------
    # Example capability gaps
    # --------------------------------------------------

    capability_gaps = [

        {
            "name": "Mathematics",
            "type": "essential",
            "importance": 4.38,
            "priority": 0.876,
            "evidence_status": "not_evidenced",
        },

        {
            "name": "Reading Comprehension",
            "type": "essential",
            "importance": 3.88,
            "priority": 0.776,
            "evidence_status": "not_evidenced",
        },

        {
            "name": "Active Learning",
            "type": "essential",
            "importance": 3.62,
            "priority": 0.724,
            "evidence_status": "not_evidenced",
        },
    ]

    # --------------------------------------------------
    # Example technical mismatches
    # --------------------------------------------------

    technical_mismatches = [
        "Java",
        "TypeScript",
        "Excel",
    ]

    # --------------------------------------------------
    # Example capability strengths
    # --------------------------------------------------

    capability_strengths = [

        {
            "name": "Critical Thinking",
            "importance": 4.25,
            "evidence_strength": 0.810,
            "evidence": [
                "Designed and evaluated predictive "
                "models using scikit-learn."
            ],
        },

        {
            "name": "Programming",
            "importance": 3.75,
            "evidence_strength": 0.810,
            "evidence": [
                "Developed software and analytics "
                "services using Python."
            ],
        },
    ]

    # --------------------------------------------------
    # Example technical strengths
    # --------------------------------------------------

    technical_strengths = [

        {
            "name": "NumPy",
            "onet_name": "NumPy",
            "specificity": 1.000,
        },

        {
            "name": "Pandas",
            "onet_name": "pandas",
            "specificity": 0.930,
        },

        {
            "name": "Scikit-learn",
            "onet_name": "Scikit-learn",
            "specificity": 0.930,
        },
    ]

    # --------------------------------------------------
    # Generate guidance
    # --------------------------------------------------

    answer = generate_answer(
        occupation=occupation,
        question=question,
        occupation_document=occupation_document,
        capability_gaps=capability_gaps,
        technical_mismatches=technical_mismatches,
        capability_strengths=capability_strengths,
        technical_strengths=technical_strengths,
    )

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("PERSONALIZED RAG GUIDANCE")
    print("=" * 60)

    if occupation_document:

        print(
            f"Retrieved occupation: "
            f"{occupation_document['metadata'].get('title')}"
        )

        print(
            f"O*NET code: "
            f"{occupation_document['metadata'].get('code')}"
        )

    # --------------------------------------------------
    # Display roadmap mapping
    # --------------------------------------------------

    roadmap = get_roadmap(
        occupation
    )

    print()

    if roadmap:

        print(
            f"Roadmap.sh: "
            f"{roadmap.get('name', 'Unknown')}"
        )

        print(
            f"Roadmap URL: "
            f"{roadmap.get('url', '')}"
        )

        print(
            "Roadmap topics:"
        )

        for topic in roadmap.get(
            "topics",
            []
        ):

            print(
                f"- {topic.get('name', 'Unknown')}"
            )

    else:

        print(
            "Roadmap.sh: No mapped roadmap"
        )

    print("=" * 60)
    print(answer)