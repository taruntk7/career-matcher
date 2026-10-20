import logging
import time
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.services.career_pipeline import run_career_pipeline


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(message)s"
    ),
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
    version="0.1.0",
)


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
    logger.info(".env path: %s | exists=%s",
                ENV_PATH, ENV_PATH.is_file())

    # Check both the expected folder and the possible
    # misplaced folder inside ml-service.
    misplaced_raw_dir = (
        ML_SERVICE_DIR / "data" / "onet" / "raw"
    )

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

    # Report the status of every required CSV.
    for filename in REQUIRED_CSV_FILES:
        expected = ONET_RAW_DIR / filename
        alternative = misplaced_raw_dir / filename

        logger.info(
            "DATA FILE | name=%s | expected=%s | "
            "exists=%s | alternative=%s | alternative_exists=%s",
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
    client_host = (
        request.client.host if request.client else "unknown"
    )

    logger.info(
        "HTTP REQUEST START | id=%s | method=%s | path=%s | "
        "client=%s | content_type=%s",
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
            "HTTP REQUEST CRASH | id=%s | method=%s | path=%s | "
            "error_type=%s | duration_ms=%.2f",
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
        "HTTP RESPONSE | id=%s | method=%s | path=%s | "
        "status=%s | response_type=%s | duration_ms=%.2f",
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
    return {
        "status": "ok",
        "service": "ml-service",
    }


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
        original_filename = Path(
            resume.filename or "resume"
        ).name
        extension = Path(original_filename).suffix.lower()

        logger.info(
            "UPLOAD RECEIVED | id=%s | filename=%s | "
            "extension=%s | content_type=%s",
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

        logger.info(
            "UPLOAD READ | id=%s | bytes=%d",
            request_id,
            len(content),
        )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded resume is empty.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        logger.info(
            "TEMP FILE CREATED | id=%s | path=%s | "
            "exists=%s | bytes=%d",
            request_id,
            temp_path,
            temp_path.is_file(),
            temp_path.stat().st_size,
        )

        logger.info(
            "ANALYZER START | id=%s | function=analyze_resume_file | "
            "argument_path=%s",
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
            "ANALYZER SUCCESS | id=%s | result_keys=%s | "
            "analysis_keys=%s | extracted_text_characters=%d",
            request_id,
            list(result.keys()),
            list(analysis.keys()) if isinstance(analysis, dict) else [],
            len(result.get("raw_text") or ""),
        )

        return result

    except HTTPException as exc:
        logger.warning(
            "HTTP ERROR | id=%s | method=%s | path=%s | "
            "status=%d | error_type=%s | detail=%r",
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
            "RESUME ANALYSIS FAILED | id=%s | "
            "error_type=%s | filename=%s | temp_path=%s",
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