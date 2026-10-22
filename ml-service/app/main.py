import logging
import re
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.career_pipeline import run_career_pipeline


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("career-matcher.api")
logger.setLevel(logging.INFO)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[2]
ML_SERVICE_DIR = APP_FILE.parents[1]

ENV_PATH = ML_SERVICE_DIR / ".env"

ONET_RAW_DIR = PROJECT_ROOT / "data" / "onet" / "raw"
ONET_PROCESSED_DIR = PROJECT_ROOT / "data" / "onet" / "processed"
ONET_JSON_PATH = ONET_PROCESSED_DIR / "occupations.json"

REQUIRED_CSV_FILES = [
    "occupation_data.csv",
    "task_statements.csv",
    "essential_skills.csv",
    "transferable_skills.csv",
    "software_skills.csv",
    "job_zones.csv",
]

app = FastAPI(
    title="Career Matcher ML Service",
    version="0.2.0",
)


# --------------------------------------------------
# Career guidance chatbot request models
# --------------------------------------------------

class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class CareerGuidanceChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    occupation: str = Field(min_length=1, max_length=200)
    occupation_code: str | None = None

    capability_gaps: list[dict[str, Any]] = Field(default_factory=list)
    technical_mismatches: list[Any] = Field(default_factory=list)
    capability_strengths: list[dict[str, Any]] = Field(default_factory=list)
    technical_strengths: list[dict[str, Any]] = Field(default_factory=list)
    resume_explicit_skills: list[str] = Field(default_factory=list)
    history: list[ChatHistoryMessage] = Field(default_factory=list)


# --------------------------------------------------
# Startup diagnostics: print important file paths
# --------------------------------------------------

@app.on_event("startup")
async def startup_diagnostics():
    logger.info("========== APPLICATION STARTUP ==========")
    logger.info("Working directory: %s", Path.cwd())
    logger.info("Main Python file: %s", APP_FILE)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("ML service directory: %s", ML_SERVICE_DIR)
    logger.info(".env path: %s | exists=%s", ENV_PATH, ENV_PATH.is_file())

    # Check both the expected folder and the possible misplaced folder.
    misplaced_raw_dir = ML_SERVICE_DIR / "data" / "onet" / "raw"

    logger.info(
        "Expected raw data directory: %s | exists=%s",
        ONET_RAW_DIR,
        ONET_RAW_DIR.is_dir(),
    )
    logger.info(
        "Alternative raw data directory: %s | exists=%s",
        misplaced_raw_dir,
        misplaced_raw_dir.is_dir(),
    )
    logger.info(
        "Processed data directory: %s | exists=%s",
        ONET_PROCESSED_DIR,
        ONET_PROCESSED_DIR.is_dir(),
    )
    logger.info(
        "Required JSON: %s | exists=%s",
        ONET_JSON_PATH,
        ONET_JSON_PATH.is_file(),
    )

    for filename in REQUIRED_CSV_FILES:
        expected = ONET_RAW_DIR / filename
        alternative = misplaced_raw_dir / filename

        logger.info(
            "DATA FILE | name=%s | expected=%s | exists=%s | "
            "alternative=%s | alternative_exists=%s",
            filename,
            expected,
            expected.is_file(),
            alternative,
            alternative.is_file(),
        )

    logger.info("========== STARTUP DIAGNOSTICS COMPLETE ==========")


# --------------------------------------------------
# HTTP request and response logging
# --------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "HTTP REQUEST START | id=%s | method=%s | path=%s | client=%s | content_type=%s",
        request_id,
        request.method,
        request.url.path,
        client_host,
        request.headers.get("content-type", "unknown"),
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "HTTP REQUEST CRASH | id=%s | method=%s | path=%s | error_type=%s | duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            type(exc).__name__,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id

    logger.info(
        "HTTP RESPONSE | id=%s | method=%s | path=%s | status=%s | response_type=%s | duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        response.headers.get("content-type", "unknown"),
        elapsed_ms,
    )

    return response


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    logger.info("Health check successful")
    return {"status": "ok", "service": "ml-service"}


# --------------------------------------------------
# RAG-based conversational career guidance
# --------------------------------------------------

def _generate_career_chat(payload: CareerGuidanceChatRequest) -> dict[str, Any]:
    """Retrieve occupation context and generate one conversational answer."""

    from app.services.career_pipeline import _load_rag_resources
    from rag.retriever import retrieve_documents, retrieve_occupation
    from rag.generator import generate_answer

    # Reuse the cached embedding model and persistent ChromaDB collection.
    model, collection = _load_rag_resources()

    # Retrieve the exact recommended occupation document first.
    occupation_document = retrieve_occupation(
        collection=collection,
        occupation_code=payload.occupation_code,
        occupation_title=payload.occupation,
    )

    if not occupation_document:
        raise HTTPException(
            status_code=503,
            detail=(
                "The O*NET document for this occupation could not be retrieved."
            ),
        )

    # Only retrieve other occupations for explicit comparison questions.
    # For ordinary career guidance, the exact occupation document plus the
    # candidate profile and curated roadmap provide more relevant grounding.
    target_metadata = occupation_document.get("metadata") or {}
    target_code = str(target_metadata.get("code") or payload.occupation_code or "")
    related_documents = []

    comparison_question = re.search(
        r"\b(compare|comparison|versus|\bvs\b|difference|differences|"
        r"alternative careers|other careers|similar occupations)\b",
        payload.question,
        flags=re.IGNORECASE,
    ) is not None

    if comparison_question:
        query = (
            f"Career comparison for: {payload.occupation}\n"
            f"User question: {payload.question.strip()}"
        )
        retrieved_documents = retrieve_documents(
            model=model,
            collection=collection,
            query=query,
            top_k=8,
        )

        for item in retrieved_documents:
            metadata = item.get("metadata") or {}
            retrieved_code = str(metadata.get("code") or "")
            retrieved_title = str(metadata.get("title") or "")

            if target_code and retrieved_code == target_code:
                continue
            if retrieved_title.casefold() == payload.occupation.casefold():
                continue

            related_documents.append(item)
            if len(related_documents) >= 3:
                break

    conversation_history = [
        message.model_dump()
        for message in payload.history[-8:]
    ]

    answer = generate_answer(
        occupation=payload.occupation,
        question=payload.question.strip(),
        occupation_document=occupation_document,
        capability_gaps=payload.capability_gaps,
        technical_mismatches=payload.technical_mismatches,
        capability_strengths=payload.capability_strengths,
        technical_strengths=payload.technical_strengths,
        resume_explicit_skills=payload.resume_explicit_skills,
        conversation_history=conversation_history,
        retrieved_documents=related_documents,
        chat_mode=True,
    )

    return {
        "answer": answer,
        "occupation": payload.occupation,
        "source": {
            "title": target_metadata.get("title"),
            "code": target_metadata.get("code"),
        },
        "retrieved_sources": [
            {
                "title": (item.get("metadata") or {}).get("title"),
                "code": (item.get("metadata") or {}).get("code"),
            }
            for item in related_documents
        ],
    }


@app.post("/api/career-guidance/chat")
async def career_guidance_chat(
    payload: CareerGuidanceChatRequest,
    request: Request,
):
    request_id = request.state.request_id

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    logger.info(
        "CAREER CHAT START | id=%s | occupation=%s | question_chars=%d",
        request_id,
        payload.occupation,
        len(payload.question),
    )

    try:
        result = await run_in_threadpool(_generate_career_chat, payload)
        logger.info(
            "CAREER CHAT COMPLETE | id=%s | occupation=%s",
            request_id,
            payload.occupation,
        )
        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "CAREER CHAT FAILED | id=%s | error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Career guidance chat failed.",
                "request_id": request_id,
            },
        ) from exc


# --------------------------------------------------
# Resume upload and analysis
# --------------------------------------------------

@app.post("/api/resume/analyze")
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
):
    request_id = request.state.request_id
    temp_path = None

    try:
        original_filename = Path(resume.filename or "resume").name
        extension = Path(original_filename).suffix.lower()

        logger.info(
            "UPLOAD RECEIVED | id=%s | filename=%s | extension=%s | content_type=%s",
            request_id,
            original_filename,
            extension,
            resume.content_type,
        )

        allowed_types = {".pdf", ".docx", ".txt"}
        if extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, DOCX, and TXT files are supported.",
            )

        content = await resume.read()
        logger.info("UPLOAD READ | id=%s | bytes=%d", request_id, len(content))

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded resume is empty.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        logger.info(
            "TEMP FILE CREATED | id=%s | path=%s | exists=%s | bytes=%d",
            request_id,
            temp_path,
            temp_path.is_file(),
            temp_path.stat().st_size,
        )
        logger.info(
            "ANALYZER START | id=%s | function=analyze_resume_file | argument_path=%s",
            request_id,
            temp_path,
        )

        result = await run_in_threadpool(
            run_career_pipeline,
            temp_path,
            original_filename,
        )

        analysis = result.get("analysis") or {}
        logger.info(
            "ANALYZER SUCCESS | id=%s | result_keys=%s | analysis_keys=%s | extracted_text_characters=%d",
            request_id,
            list(result.keys()),
            list(analysis.keys()) if isinstance(analysis, dict) else [],
            len(result.get("raw_text") or ""),
        )
        return result

    except HTTPException as exc:
        logger.warning(
            "HTTP ERROR | id=%s | method=%s | path=%s | status=%d | error_type=%s | detail=%r",
            request_id,
            request.method,
            request.url.path,
            exc.status_code,
            type(exc).__name__,
            exc.detail,
        )
        raise

    except Exception as exc:
        logger.exception(
            "RESUME ANALYSIS FAILED | id=%s | error_type=%s | filename=%s | temp_path=%s",
            request_id,
            type(exc).__name__,
            resume.filename,
            temp_path,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": type(exc).__name__,
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc

    finally:
        if temp_path is not None:
            try:
                logger.info(
                    "TEMP FILE CLEANUP | id=%s | path=%s | exists=%s",
                    request_id,
                    temp_path,
                    temp_path.exists(),
                )
                temp_path.unlink(missing_ok=True)
            except Exception:
                logger.exception(
                    "TEMP FILE CLEANUP FAILED | id=%s | path=%s",
                    request_id,
                    temp_path,
                )

        await resume.close()
