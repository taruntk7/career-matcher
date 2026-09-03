import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq


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


ENV_PATH = (
    PROJECT_ROOT
    / "ml-service"
    / ".env"
)

OCCUPATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "onet"
    / "processed"
    / "occupations.json"
)

load_dotenv(
    ENV_PATH
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "openai/gpt-oss-120b"

MAX_RESUME_CHARS = 14000

SKILL_MAX_TOKENS = 1800

CAPABILITY_MAX_TOKENS = 2200


# =========================================================
# O*NET capability vocabulary
# =========================================================

def load_onet_capabilities() -> List[str]:
    """
    Load the canonical O*NET capability vocabulary from
    essential and transferable skills.
    """

    if not OCCUPATIONS_PATH.exists():

        raise FileNotFoundError(
            "Could not find occupations.json:\n"
            f"{OCCUPATIONS_PATH}"
        )

    with open(
        OCCUPATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        occupations = json.load(f)

    capabilities = set()

    for occupation in occupations:

        for skill in occupation.get(
            "essential_skills",
            [],
        ):

            name = str(
                skill.get(
                    "name",
                    "",
                )
            ).strip()

            if name:
                capabilities.add(
                    name
                )

        for skill in occupation.get(
            "transferable_skills",
            [],
        ):

            name = str(
                skill.get(
                    "name",
                    "",
                )
            ).strip()

            if name:
                capabilities.add(
                    name
                )

    return sorted(
        capabilities
    )


# =========================================================
# Groq client
# =========================================================

def get_client() -> Groq:
    """
    Create the Groq client.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY was not found.\n"
            f"Expected it in:\n{ENV_PATH}"
        )

    return Groq(
        api_key=api_key
    )


# =========================================================
# Text preparation
# =========================================================

def prepare_resume_text(
    resume_text: str,
) -> str:
    """
    Conservatively clean and size-limit the resume text.
    """

    if not resume_text:

        raise ValueError(
            "Resume text is empty."
        )

    text = resume_text.strip()

    text = text.replace(
        "\x00",
        " ",
    )

    text = " ".join(
        text.split()
    )

    if len(text) > MAX_RESUME_CHARS:

        text = text[
            :MAX_RESUME_CHARS
        ]

    return text


# =========================================================
# Evidence strength
# =========================================================

def capability_evidence_strength(
    capability: Dict,
) -> float:
    """
    Convert LLM confidence + evidence type into a single
    evidence-strength score.
    """

    confidence = max(
        0.0,
        min(
            1.0,
            float(
                capability.get(
                    "confidence",
                    0.0,
                )
            ),
        ),
    )

    evidence_type = capability.get(
        "evidence_type",
        "weak_inference",
    )

    multiplier = {
        "explicit":
            1.00,

        "strong_inference":
            0.90,

        "moderate_inference":
            0.75,

        "weak_inference":
            0.55,
    }.get(
        evidence_type,
        0.55,
    )

    return max(
        0.0,
        min(
            1.0,
            confidence * multiplier,
        ),
    )


# =========================================================
# Explicit skills schema
# =========================================================

EXPLICIT_SKILLS_SCHEMA = {
    "type": "object",

    "properties": {

        "explicit_skills": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "name": {
                        "type": "string"
                    },

                    "evidence": {
                        "type": "string"
                    },

                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                },

                "required": [
                    "name",
                    "evidence",
                    "confidence",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "explicit_skills",
    ],

    "additionalProperties": False,
}


# =========================================================
# Capability schema
# =========================================================

def build_capability_schema(
    onet_capabilities: List[str],
) -> Dict:
    """
    Build a schema in which capability names can ONLY be
    canonical O*NET capability names.
    """

    return {
        "type": "object",

        "properties": {

            "capabilities": {
                "type": "array",

                "items": {
                    "type": "object",

                    "properties": {

                        "name": {
                            "type": "string",

                            "enum":
                                onet_capabilities,
                        },

                        "evidence": {
                            "type": "string"
                        },

                        "evidence_type": {
                            "type": "string",

                            "enum": [
                                "explicit",
                                "strong_inference",
                                "moderate_inference",
                                "weak_inference",
                            ],
                        },

                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },

                    "required": [
                        "name",
                        "evidence",
                        "evidence_type",
                        "confidence",
                    ],

                    "additionalProperties": False,
                },
            },
        },

        "required": [
            "capabilities",
        ],

        "additionalProperties": False,
    }


# =========================================================
# Explicit skill prompt
# =========================================================

SKILL_SYSTEM_PROMPT = """
You extract explicit skills from resumes.

Return ONLY skills that are directly stated or clearly
demonstrated in the resume.

Examples:
- Python
- SQL
- React
- PostgreSQL
- Legal Research
- Contract Drafting

Rules:

- Do not infer soft skills.
- Do not infer capabilities.
- Do not recommend careers.
- Do not rank occupations.
- Do not invent information.
- Every skill must have supporting resume evidence.
- Keep evidence short.
- Confidence must reflect how clearly the resume supports
  the skill.

A skill listed in a Technical Skills section is valid
explicit evidence.
A technology clearly used in a project or experience bullet
is also valid evidence.
"""


# =========================================================
# Capability prompt
# =========================================================

def build_capability_system_prompt(
    onet_capabilities: List[str],
) -> str:
    """
    Build the capability extraction prompt.
    """

    vocabulary = "\n".join(
        f"- {name}"
        for name in onet_capabilities
    )

    return f"""
You extract broader capabilities from resume evidence.

IMPORTANT:

You may ONLY use capability names from the O*NET vocabulary
provided below.

Do NOT invent capability names.

A capability must be supported by actual behavior,
responsibilities, project work, or demonstrated activity.

Examples:

"Built and debugged software"
    -> Programming
    -> Complex Problem Solving

"Compared model performance and improved feature selection"
    -> Critical Thinking

"Led a five-person team and delegated responsibilities"
    -> Coordination
    -> Management of Personnel Resources

"Negotiated a contract"
    -> Negotiation

"Presented findings to stakeholders"
    -> Speaking

Rules:

- Do not infer unrelated capabilities.
- Technical skill alone does not prove leadership,
  negotiation, or coordination.
- Prefer fewer strong capabilities over many speculative ones.
- Every capability must include actual resume evidence.
- Do not rank occupations.
- Do not recommend careers.
- Do not calculate career-fit scores.

Evidence type:

explicit
    capability itself is directly stated.

strong_inference
    strong behavioral/project evidence.

moderate_inference
    reasonable but less direct evidence.

weak_inference
    limited evidence.

Confidence reflects the quality of the evidence.

CANONICAL O*NET CAPABILITY VOCABULARY
-------------------------------------

{vocabulary}
"""


# =========================================================
# Call 1 — explicit skills
# =========================================================

def extract_explicit_skills(
    resume_text: str,
    client: Groq,
) -> List[Dict]:
    """
    First LLM call.

    Extracts only explicit skills.
    """

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SKILL_SYSTEM_PROMPT,
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            "Analyze this resume and "
                            "extract its explicit skills.\n\n"
                            "RESUME\n"
                            "------\n"
                            f"{resume_text}"
                        ),
                },
            ],

            temperature=0,

            max_tokens=SKILL_MAX_TOKENS,

            response_format={
                "type":
                    "json_schema",

                "json_schema": {
                    "name":
                        "explicit_skill_analysis",

                    "strict":
                        True,

                    "schema":
                        EXPLICIT_SKILLS_SCHEMA,
                },
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Groq returned an empty explicit-skill response."
        )

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON for explicit skills."
        ) from exc

    skills = result.get(
        "explicit_skills",
        [],
    )

    if not isinstance(
        skills,
        list,
    ):

        return []

    return skills


# =========================================================
# Call 2 — O*NET capabilities
# =========================================================

def extract_capabilities(
    resume_text: str,
    onet_capabilities: List[str],
    client: Groq,
) -> List[Dict]:
    """
    Second LLM call.

    Extracts only canonical O*NET capabilities.
    """

    schema = build_capability_schema(
        onet_capabilities
    )

    system_prompt = (
        build_capability_system_prompt(
            onet_capabilities
        )
    )

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt,
                },

                {
                    "role":
                        "user",

                    "content":
                        (
                            "Analyze this resume and "
                            "extract evidence-backed O*NET "
                            "capabilities.\n\n"
                            "RESUME\n"
                            "------\n"
                            f"{resume_text}"
                        ),
                },
            ],

            temperature=0,

            max_tokens=CAPABILITY_MAX_TOKENS,

            response_format={
                "type":
                    "json_schema",

                "json_schema": {
                    "name":
                        "onet_capability_analysis",

                    "strict":
                        True,

                    "schema":
                        schema,
                },
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Groq returned an empty capability response."
        )

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON for capabilities."
        ) from exc

    capabilities = result.get(
        "capabilities",
        [],
    )

    if not isinstance(
        capabilities,
        list,
    ):

        return []

    # -----------------------------------------------------
    # Calculate evidence strength locally.
    # -----------------------------------------------------

    for capability in capabilities:

        capability[
            "evidence_strength"
        ] = capability_evidence_strength(
            capability
        )

    return capabilities


# =========================================================
# Full resume analysis
# =========================================================

def analyze_resume_text(
    resume_text: str,
    client: Groq | None = None,
) -> Dict:
    """
    Run the two-stage resume analysis.

    Stage 1:
        explicit skills

    Stage 2:
        canonical O*NET capabilities
    """

    resume_text = prepare_resume_text(
        resume_text
    )

    if client is None:

        client = get_client()

    onet_capabilities = (
        load_onet_capabilities()
    )

    if not onet_capabilities:

        raise RuntimeError(
            "O*NET capability vocabulary is empty."
        )

    explicit_skills = (
        extract_explicit_skills(
            resume_text,
            client,
        )
    )

    capabilities = (
        extract_capabilities(
            resume_text,
            onet_capabilities,
            client,
        )
    )

    return {
        "explicit_skills":
            explicit_skills,

        "capabilities":
            capabilities,
    }


# =========================================================
# Resume file analysis
# =========================================================

def analyze_resume_file(
    resume_path: str | Path,
) -> Dict:
    """
    Parse a resume file and run the two-stage LLM analysis.
    """

    from parsing.resume_parser import (
        parse_resume,
    )

    resume = parse_resume(
        resume_path
    )

    analysis = analyze_resume_text(
        resume["raw_text"]
    )

    return {
        "filename":
            resume["filename"],

        "file_type":
            resume["file_type"],

        "raw_text":
            resume["raw_text"],

        "analysis":
            analysis,
    }


# =========================================================
# Printing
# =========================================================

def print_analysis(
    result: Dict,
) -> None:

    analysis = result[
        "analysis"
    ]

    print()
    print("=" * 70)
    print("LLM RESUME ANALYSIS")
    print("=" * 70)

    print()
    print("Resume:")
    print(
        result["filename"]
    )

    # -----------------------------------------------------
    # Explicit skills
    # -----------------------------------------------------

    print()
    print("Explicit skills:")
    print("-" * 70)

    explicit_skills = analysis.get(
        "explicit_skills",
        [],
    )

    if explicit_skills:

        for skill in explicit_skills:

            print(
                f"- {skill['name']}"
                f" | confidence: "
                f"{float(skill['confidence']):.2f}"
            )

            print(
                f"  Evidence: "
                f"{skill['evidence']}"
            )

    else:

        print("- None")

    # -----------------------------------------------------
    # O*NET capabilities
    # -----------------------------------------------------

    print()
    print("O*NET capabilities:")
    print("-" * 70)

    capabilities = analysis.get(
        "capabilities",
        [],
    )

    if capabilities:

        for capability in capabilities:

            print(
                f"- {capability['name']}"
                f" | type: "
                f"{capability['evidence_type']}"
                f" | confidence: "
                f"{float(capability['confidence']):.2f}"
                f" | evidence strength: "
                f"{float(capability.get('evidence_strength', 0.0)):.3f}"
            )

            print(
                f"  Evidence: "
                f"{capability['evidence']}"
            )

    else:

        print("- None")


# =========================================================
# Standalone test
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "ml-service/app/parsing/"
            "resume_analyzer.py "
            "<resume_path>"
        )

        raise SystemExit(1)

    resume_path = sys.argv[1]

    try:

        result = analyze_resume_file(
            resume_path
        )

        print_analysis(
            result
        )

    except Exception as exc:

        print()
        print("ERROR:")
        print(
            str(exc)
        )

        raise SystemExit(1)