"""Language-model helpers for prescription extraction."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PRESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patient_name": {"type": ["string", "null"]},
        "patient_age": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "doctor_name": {"type": ["string", "null"]},
        "diagnosis": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "medicines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "dosage": {"type": ["string", "null"]},
                    "frequency": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "confidences": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "number"},
                            "dosage": {"type": "number"},
                            "frequency": {"type": "number"},
                            "duration": {"type": "number"},
                        },
                        "required": ["name", "dosage", "frequency", "duration"],
                    },
                },
                "required": ["name", "dosage", "frequency", "duration", "confidence", "confidences"],
            },
        },
        "confidences": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "patient_name": {"type": "number"},
                "patient_age": {"type": "number"},
                "date": {"type": "number"},
                "doctor_name": {"type": "number"},
                "diagnosis": {"type": "number"},
                "notes": {"type": "number"},
                "medicines": {"type": "number"},
            },
            "required": ["patient_name", "patient_age", "date", "doctor_name", "diagnosis", "notes", "medicines"],
        },
    },
    "required": [
        "patient_name",
        "patient_age",
        "date",
        "doctor_name",
        "diagnosis",
        "notes",
        "medicines",
        "confidences",
    ],
}


@lru_cache(maxsize=1)
def load_llm_pipeline() -> Any:
    """Load the local language pipeline used for entity extraction."""
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not installed; using blank English pipeline.")
            return spacy.blank("en")
    except ImportError:
        logger.warning("spaCy is not installed; language-model fallback is disabled.")
        return None


def extract_person_entities(text: str, llm: Any | None = None) -> list[tuple[str, int, int]]:
    """Extract person entities from text using the configured language pipeline."""
    pipeline = llm if llm is not None else load_llm_pipeline()
    if pipeline is None:
        return []

    doc = pipeline(text[:500])
    return [
        (entity.text, entity.start_char, entity.end_char)
        for entity in getattr(doc, "ents", [])
        if entity.label_ == "PERSON"
    ]


def should_use_openai(config: dict[str, Any] | None) -> bool:
    """Return whether OpenAI extraction is enabled by config and environment."""
    llm_config = (config or {}).get("llm", {})
    provider = str(llm_config.get("provider", "local")).lower()
    api_key_env = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
    return provider == "openai" and bool(os.getenv(api_key_env))


def extract_prescription_with_openai(
    raw_text: str,
    baseline: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract prescription fields with the OpenAI Responses API.

    Returns None when OpenAI is not configured or the request fails, allowing
    callers to keep using deterministic local extraction as a fallback.
    """
    llm_config = (config or {}).get("llm", {})
    api_key_env = str(llm_config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key:
        return None

    model = str(llm_config.get("model", "gpt-4.1-mini"))
    timeout_seconds = float(llm_config.get("timeout_seconds", 20))
    max_chars = int(llm_config.get("max_input_chars", 6000))

    prompt = (
        "Extract structured prescription fields from OCR text. "
        "Use null for missing values, keep medicine names concise, normalize common prescription "
        "frequency abbreviations into readable English, and return only data that is supported by "
        "the OCR text. Use the baseline extraction as a hint, not as proof.\n\n"
        f"Baseline JSON:\n{json.dumps(baseline, ensure_ascii=True)}\n\n"
        f"OCR text:\n{raw_text[:max_chars]}"
    )

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You extract medical prescription OCR into strict JSON. "
                    "Do not invent patient, doctor, or medicine details."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "prescription_fields",
                "strict": True,
                "schema": PRESCRIPTION_SCHEMA,
            }
        },
    }

    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = _extract_response_text(response.json())
        parsed = json.loads(content)
    except Exception as exc:
        logger.warning("OpenAI prescription extraction failed; using local extraction: %s", exc)
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_response_text(response_json: dict[str, Any]) -> str:
    """Get text content from a Responses API result."""
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]

    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("Responses API result did not include text output")
