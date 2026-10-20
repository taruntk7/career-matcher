"""Orchestrate resume analysis, occupation ranking, and RAG career guidance."""

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


# Add app/ to the import path because existing modules use imports such as
# `from matching...` and `from data...` rather than package-relative imports.
PIPELINE_FILE = Path(__file__).resolve()
APP_DIR = PIPELINE_FILE.parents[1]
PROJECT_ROOT = PIPELINE_FILE.parents[3]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from parsing.resume_analyzer import analyze_resume_file
from data.onet_loader import build_occupation_dataset
from matching.matcher import (
    load_matcher,
    match_occupations,
    enrich_matches,
)
from matching.scoring import rank_matches
from matching.recommendation import build_recommendation

logger = logging.getLogger("career-matcher.pipeline")


@lru_cache(maxsize=1)
def _load_matching_resources():
    """Load the embedding model and occupation Chroma collection once."""
    logger.info("Loading occupation matcher resources")
    return load_matcher()


@lru_cache(maxsize=1)
def _load_occupation_dataset():
    """Build/load the O*NET occupation data once per server process."""
    logger.info("Building occupation dataset from O*NET CSV files")
    return build_occupation_dataset()


@lru_cache(maxsize=1)
def _load_rag_resources():
    """Load RAG resources lazily so missing RAG setup doesn't prevent ranking."""
    from rag.retriever import load_retriever

    logger.info("Loading career-guidance RAG resources")
    return load_retriever()


def _build_recommendation(match: dict[str, Any], rank: int, occupation_by_code: dict[str, dict]) -> dict[str, Any]:
    recommendation = build_recommendation(match)
    code = match.get("code")
    occupation_record = occupation_by_code.get(str(code), {})

    # The existing builder omits these metadata fields, so attach them here
    # for the API/UI without changing the ranking implementation.
    recommendation["code"] = code
    recommendation["rank"] = rank
    recommendation["description"] = occupation_record.get("description", "")
    return recommendation


def run_career_pipeline(resume_path: str | Path, original_filename: str | None = None) -> dict[str, Any]:
    """Run analysis -> retrieval -> enrichment -> ranking -> recommendation -> RAG."""
    resume_path = Path(resume_path)
    logger.info(
        "PIPELINE START | resume_path=%s | exists=%s | original_filename=%s",
        resume_path,
        resume_path.is_file(),
        original_filename,
    )

    # 1. Extract resume text, explicit skills, and inferred capabilities.
    resume_result = analyze_resume_file(resume_path)
    if original_filename:
        resume_result["filename"] = original_filename

    analysis = resume_result.get("analysis") or {}
    explicit_skills = [
        item["name"]
        for item in analysis.get("explicit_skills", [])
        if isinstance(item, dict) and item.get("name")
    ]
    capabilities = analysis.get("capabilities", []) or []
    profile_text = resume_result.get("raw_text", "") or ""

    logger.info(
        "ANALYSIS COMPLETE | filename=%s | text_chars=%d | explicit_skills=%d | capabilities=%d",
        resume_result.get("filename"),
        len(profile_text),
        len(explicit_skills),
        len(capabilities),
    )

    base_response: dict[str, Any] = {
        "filename": resume_result.get("filename"),
        "file_type": resume_result.get("file_type"),
        # Retained for compatibility with the current API response.
        "raw_text": profile_text,
        "analysis": analysis,
        "explicit_skills": explicit_skills,
        "capabilities": capabilities,
    }

    # 2. Load the matcher / O*NET dataset, retrieve candidates, and enrich them.
    model, collection = _load_matching_resources()
    occupations = _load_occupation_dataset()
    occupation_by_code = {
        str(occupation.get("code")): occupation
        for occupation in occupations
        if occupation.get("code") is not None
    }

    logger.info("SEMANTIC RETRIEVAL START | top_k=50")
    matches = match_occupations(
        model,
        collection,
        profile_text,
        top_k=50,
    )
    logger.info("SEMANTIC RETRIEVAL COMPLETE | candidates=%d", len(matches))

    matches = enrich_matches(
        matches,
        explicit_skills,
        capabilities,
        occupations,
    )
    logger.info("MATCH ENRICHMENT COMPLETE | candidates=%d", len(matches))

    # 3. Calculate final scores and sort descending by final_score.
    matches = rank_matches(matches)
    logger.info("RANKING COMPLETE | ranked_candidates=%d", len(matches))

    if not matches:
        logger.warning("PIPELINE FINISHED WITHOUT OCCUPATION MATCHES")
        return {
            "filename": resume_result.get("filename"),
            "recommendations": [],
            "career_guidance": None,
            "career_guidance_status": "not_run",
            "career_guidance_source": None,
            "career_guidance_error": None,
            "message": "Resume analyzed, but no occupation matches were returned.",
        }

    # 4. Prepare detailed results for the top 10 ranked occupations.
    recommendations = [
        _build_recommendation(match, rank, occupation_by_code)
        for rank, match in enumerate(matches[:3], start=1)
    ]
    best_recommendation = recommendations[0]

    logger.info(
        "RECOMMENDATION BUILT | title=%s | code=%s | final_score=%s",
        best_recommendation.get("title"),
        best_recommendation.get("code"),
        best_recommendation.get("final_score"),
    )

    # 5. Generate RAG guidance for the top-ranked occupation.
    # If the separate RAG collection, roadmap mapping, or Groq call is not
    # available, preserve the ranked recommendations and report the issue.
    career_guidance = None
    guidance_status = "unavailable"
    guidance_source = None
    guidance_error = None

    try:
        from rag.retriever import retrieve_occupation
        from rag.generator import generate_answer

        _rag_model, rag_collection = _load_rag_resources()
        occupation_document = retrieve_occupation(
            collection=rag_collection,
            occupation_code=best_recommendation.get("code"),
            occupation_title=best_recommendation.get("title"),
        )

        if not occupation_document:
            guidance_error = (
                "No matching occupation document was found in the RAG collection."
            )
            logger.warning(
                "RAG DOCUMENT NOT FOUND | title=%s | code=%s",
                best_recommendation.get("title"),
                best_recommendation.get("code"),
            )
        else:
            question = (
                "What should I focus on learning next to become more suitable for this career?"
            )
            career_guidance = generate_answer(
                occupation=best_recommendation["title"],
                question=question,
                occupation_document=occupation_document,
                capability_gaps=best_recommendation.get("potential_capability_gaps", []),
                technical_mismatches=best_recommendation.get("technical_mismatches", []),
                capability_strengths=best_recommendation.get("top_capability_strengths", []),
                technical_strengths=best_recommendation.get("top_technical_strengths", []),
            )
            guidance_status = "success"
            metadata = occupation_document.get("metadata") or {}
            guidance_source = {
                "title": metadata.get("title"),
                "code": metadata.get("code"),
            }
            logger.info(
                "RAG GUIDANCE COMPLETE | occupation=%s | source=%s",
                best_recommendation.get("title"),
                guidance_source,
            )

    except Exception as exc:
        guidance_error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "RAG GUIDANCE FAILED | occupation=%s | error_type=%s",
            best_recommendation.get("title"),
            type(exc).__name__,
        )

    logger.info(
        "PIPELINE COMPLETE | filename=%s | recommendations=%d | rag_status=%s",
        resume_result.get("filename"),
        len(recommendations),
        guidance_status,
    )

    # Return one copy of the top-three detailed recommendations.
    # Avoid duplicating the same large objects under multiple keys,
    # and omit the extracted resume text from the API response.
    return {
        "filename": resume_result.get("filename"),
        "recommendations": recommendations,
        "career_guidance": career_guidance,
        "career_guidance_status": guidance_status,
        "career_guidance_source": guidance_source,
        "career_guidance_error": guidance_error,
        "message": (
            "Top career recommendations generated."
            if career_guidance_status_success(guidance_status)
            else "Top career recommendations generated; career guidance is unavailable."
        ),
    }


def career_guidance_status_success(status: str) -> bool:
    """Small helper to keep the response message readable."""
    return status == "success"
