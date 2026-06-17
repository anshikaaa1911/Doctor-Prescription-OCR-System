"""FastAPI application for prescription OCR."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from src.extractor import extract_prescription_fields
from src.ocr_engine import OCRPipeline
from src.preprocessor import preprocess_array
from src.validator import validate_medicines

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Doctor Prescription OCR API", version="1.0.0")
BATCH_RESULTS: dict[str, dict[str, Any]] = {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration.

    Args:
        config_path: Optional config path.

    Returns:
        Parsed configuration dictionary.
    """
    path = config_path or Path.cwd() / "config.yaml"
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed; using default configuration.")
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def extension_for_upload(upload: UploadFile) -> str:
    """Get normalized file extension from an upload.

    Args:
        upload: Uploaded file.

    Returns:
        Lowercase extension without dot.
    """
    return Path(upload.filename or "").suffix.lower().lstrip(".")


def validate_upload_metadata(upload: UploadFile, config: dict[str, Any]) -> None:
    """Validate upload filename and extension.

    Args:
        upload: Uploaded file.
        config: Application configuration.

    Raises:
        HTTPException: If the extension is not allowed.
    """
    allowed = {str(value).lower() for value in config.get("api", {}).get("allowed_formats", ["jpg", "png", "pdf"])}
    extension = extension_for_upload(upload)
    normalized = "jpg" if extension == "jpeg" else extension
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {extension}")


async def read_limited_upload(upload: UploadFile, max_size_mb: int) -> bytes:
    """Read an upload while enforcing maximum size.

    Args:
        upload: Uploaded file.
        max_size_mb: Maximum accepted file size in megabytes.

    Returns:
        Uploaded bytes.

    Raises:
        HTTPException: If the file exceeds the configured size.
    """
    max_bytes = max_size_mb * 1024 * 1024
    content = await upload.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {max_size_mb} MB limit")
    return content


def decode_image(content: bytes, extension: str) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV image.

    Args:
        content: Uploaded file bytes.
        extension: File extension without dot.

    Returns:
        Decoded BGR image.

    Raises:
        HTTPException: If decoding fails.
    """
    if extension == "pdf":
        return decode_pdf_first_page(content)

    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")
    return image


def decode_pdf_first_page(content: bytes) -> np.ndarray:
    """Decode the first page of a PDF into an OpenCV image.

    Args:
        content: PDF bytes.

    Returns:
        First page as a BGR image.

    Raises:
        HTTPException: If PDF conversion is unavailable or fails.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PDF support requires pdf2image") from exc

    try:
        pages = convert_from_bytes(content, first_page=1, last_page=1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode PDF") from exc
    if not pages:
        raise HTTPException(status_code=400, detail="PDF contains no pages")
    rgb = np.array(pages[0].convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_image_array(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Run preprocessing, OCR, extraction, and validation.

    Args:
        image: Input BGR image.
        config: Application config.

    Returns:
        Full OCR response.
    """
    preprocessed = preprocess_array(image, config)
    ocr_result = OCRPipeline(config).recognize(preprocessed["image"])
    extracted = extract_prescription_fields(ocr_result["raw_text"])
    medicine_names = [medicine["name"] for medicine in extracted["medicines"]]
    validation = validate_medicines(medicine_names)
    return {
        "extracted_fields": extracted,
        "ocr": ocr_result,
        "preprocessing": preprocessed["metadata"],
        "medicine_validation": validation,
        "confidence": {
            "ocr": ocr_result["confidence"],
            "preprocessing": preprocessed["metadata"]["confidence"],
        },
    }


async def process_upload(upload: UploadFile, config: dict[str, Any]) -> dict[str, Any]:
    """Process a single uploaded prescription file.

    Args:
        upload: Uploaded image or PDF.
        config: Application config.

    Returns:
        Full OCR response.
    """
    validate_upload_metadata(upload, config)
    max_size_mb = int(config.get("api", {}).get("max_file_size_mb", 10))
    content = await read_limited_upload(upload, max_size_mb)
    extension = extension_for_upload(upload)
    extension = "jpg" if extension == "jpeg" else extension
    image = decode_image(content, extension)
    return await asyncio.to_thread(process_image_array, image, config)


def process_batch_job(job_id: str, files: list[tuple[str, bytes]], config: dict[str, Any]) -> None:
    """Run a batch OCR job in the background.

    Args:
        job_id: Batch job identifier.
        files: File name and bytes pairs.
        config: Application config.
    """
    BATCH_RESULTS[job_id] = {"status": "processing", "results": []}
    results: list[dict[str, Any]] = []
    for filename, content in files:
        extension = Path(filename).suffix.lower().lstrip(".")
        extension = "jpg" if extension == "jpeg" else extension
        try:
            image = decode_image(content, extension)
            result = process_image_array(image, config)
            results.append({"filename": filename, "result": result})
        except Exception as exc:
            logger.exception("Batch file failed: %s", filename)
            results.append({"filename": filename, "error": str(exc)})
    BATCH_RESULTS[job_id] = {"status": "completed", "results": results}


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health.

    Returns:
        Health status payload.
    """
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    """OCR a single uploaded prescription.

    Args:
        file: Multipart image or PDF upload.

    Returns:
        Extracted prescription data with confidence metadata.
    """
    config = load_config()
    return await process_upload(file, config)


@app.post("/batch")
async def batch_endpoint(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Start a background batch OCR job.

    Args:
        background_tasks: FastAPI background task manager.
        files: Multipart uploads.

    Returns:
        Accepted job identifier.
    """
    config = load_config()
    max_size_mb = int(config.get("api", {}).get("max_file_size_mb", 10))
    stored_files: list[tuple[str, bytes]] = []
    for upload in files:
        validate_upload_metadata(upload, config)
        content = await read_limited_upload(upload, max_size_mb)
        stored_files.append((upload.filename or f"{uuid4()}.bin", content))

    job_id = uuid4().hex
    BATCH_RESULTS[job_id] = {"status": "queued", "results": []}
    background_tasks.add_task(process_batch_job, job_id, stored_files, config)
    return {"job_id": job_id, "status": "queued"}


@app.get("/batch/{job_id}")
async def batch_status(job_id: str) -> dict[str, Any]:
    """Return batch job status.

    Args:
        job_id: Batch job identifier.

    Returns:
        Batch status and results if complete.

    Raises:
        HTTPException: If job id is unknown.
    """
    if job_id not in BATCH_RESULTS:
        raise HTTPException(status_code=404, detail="Batch job not found")
    return BATCH_RESULTS[job_id]


def save_temp_upload(content: bytes, suffix: str) -> Path:
    """Save bytes to a temporary file.

    Args:
        content: File bytes.
        suffix: File suffix.

    Returns:
        Temporary file path.
    """
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with handle:
        handle.write(content)
    return Path(handle.name)
