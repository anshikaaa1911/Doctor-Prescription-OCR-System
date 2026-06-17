"""Structured prescription field extraction using regex and spaCy."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class Medicine(TypedDict):
    """Structured medicine entry."""

    name: str
    dosage: str
    frequency: str
    duration: str


class PrescriptionFields(TypedDict):
    """Structured prescription fields."""

    patient_name: str
    patient_age: str
    date: str
    doctor_name: str
    medicines: list[Medicine]
    diagnosis: str
    notes: str


PATIENT_PATTERNS = [
    re.compile(r"(?:patient|pt\.?|name)\s*[:\-]\s*(?P<value>[A-Z][A-Za-z .'-]{1,80})", re.IGNORECASE),
]
AGE_PATTERN = re.compile(r"(?:age|aged)\s*[:\-]?\s*(?P<value>\d{1,3})(?:\s*(?:yrs?|years?|y/o))?", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"(?:date|dt\.?)\s*[:\-]?\s*(?P<value>\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})",
    re.IGNORECASE,
)
DOCTOR_PATTERN = re.compile(r"(?:dr\.?|doctor|physician)\s*[:\-]?\s*(?P<value>(?:Dr\.?\s*)?[A-Z][A-Za-z .'-]{1,80})", re.IGNORECASE)
DIAGNOSIS_PATTERN = re.compile(r"(?:diagnosis|dx|assessment)\s*[:\-]\s*(?P<value>.+)", re.IGNORECASE)
NOTES_PATTERN = re.compile(r"(?:notes?|advice|remarks?)\s*[:\-]\s*(?P<value>.+)", re.IGNORECASE)
RX_LINE_PATTERN = re.compile(
    r"^(?:rx\s*[:\-]?\s*)?(?:tab\.?|cap\.?|syp\.?|inj\.?|ointment|cream|drop)?\s*"
    r"(?P<name>[A-Z][A-Za-z0-9+\- ]{2,50}?)"
    r"(?=\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)|\s+(?:once|twice|thrice|daily|od|bd|bid|tds|tid|qid|qhs|hs|sos|prn|q\d+h|[01]-[01]-[01])|\s+(?:for\s*)?\d+\s*(?:days?|weeks?|months?)|$)"
    r"(?:\s+(?P<dosage>\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)))?"
    r"(?:\s+(?P<frequency>(?:once|twice|thrice|daily|od|bd|bid|tds|tid|qid|qhs|hs|sos|prn|q\d+h|[01]-[01]-[01])))?"
    r"(?:\s+(?:for\s*)?(?P<duration>\d+\s*(?:days?|weeks?|months?)))?",
    re.IGNORECASE,
)


def _load_spacy_model() -> Any:
    """Load a spaCy model with a lightweight fallback.

    Returns:
        spaCy language pipeline.
    """
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not installed; using blank English pipeline.")
            return spacy.blank("en")
    except ImportError:
        logger.warning("spaCy is not installed; named-entity fallback is disabled.")
        return None


def normalize_text(text: str) -> str:
    """Normalize OCR text for extraction.

    Args:
        text: Raw OCR text.

    Returns:
        Whitespace-normalized text.
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def extract_first(patterns: list[re.Pattern[str]], text: str) -> str:
    """Extract the first named regex group value.

    Args:
        patterns: Regex patterns containing a value group.
        text: Text to search.

    Returns:
        Matched value or an empty string.
    """
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return clean_field(match.group("value"))
    return ""


def clean_field(value: str) -> str:
    """Clean extracted field text.

    Args:
        value: Extracted field value.

    Returns:
        Cleaned field value.
    """
    return re.sub(r"\s+", " ", value).strip(" .:-")


def extract_patient_name(text: str, nlp: Any | None = None) -> str:
    """Extract patient name.

    Args:
        text: Normalized OCR text.
        nlp: Optional spaCy pipeline.

    Returns:
        Patient name or an empty string.
    """
    regex_value = extract_first(PATIENT_PATTERNS, text)
    if regex_value:
        return regex_value
    if nlp is None:
        return ""
    doc = nlp(text[:500])
    people = [entity.text for entity in getattr(doc, "ents", []) if entity.label_ == "PERSON"]
    return clean_field(people[0]) if people else ""


def extract_doctor_name(text: str, nlp: Any | None = None) -> str:
    """Extract doctor name.

    Args:
        text: Normalized OCR text.
        nlp: Optional spaCy pipeline.

    Returns:
        Doctor name or an empty string.
    """
    match = DOCTOR_PATTERN.search(text)
    if match:
        value = clean_field(match.group("value"))
        return value if value.lower().startswith("dr") else f"Dr. {value}"
    if nlp is None:
        return ""
    for entity in getattr(nlp(text[:500]), "ents", []):
        if entity.label_ == "PERSON" and "dr" in text[max(0, entity.start_char - 10) : entity.start_char].lower():
            return clean_field(entity.text)
    return ""


def extract_age(text: str) -> str:
    """Extract patient age.

    Args:
        text: Normalized OCR text.

    Returns:
        Age string or an empty string.
    """
    match = AGE_PATTERN.search(text)
    return clean_field(match.group("value")) if match else ""


def extract_date(text: str) -> str:
    """Extract prescription date.

    Args:
        text: Normalized OCR text.

    Returns:
        Date string or an empty string.
    """
    match = DATE_PATTERN.search(text)
    return clean_field(match.group("value")) if match else ""


def extract_diagnosis(text: str) -> str:
    """Extract diagnosis text.

    Args:
        text: Normalized OCR text.

    Returns:
        Diagnosis or an empty string.
    """
    match = DIAGNOSIS_PATTERN.search(text)
    return clean_field(match.group("value")) if match else ""


def extract_notes(text: str) -> str:
    """Extract notes or advice.

    Args:
        text: Normalized OCR text.

    Returns:
        Notes or an empty string.
    """
    values = [clean_field(match.group("value")) for match in NOTES_PATTERN.finditer(text)]
    return " ".join(value for value in values if value)


def is_medicine_candidate(line: str) -> bool:
    """Determine whether a line likely contains a medicine.

    Args:
        line: OCR text line.

    Returns:
        True when the line resembles a medicine instruction.
    """
    lowered = line.lower()
    markers = ["tab", "cap", "syp", "inj", "rx", "mg", "ml", "daily", "bd", "tds", "bid", "tid", "days"]
    blocked = ["patient", "date", "doctor", "diagnosis", "notes", "advice", "age"]
    return any(marker in lowered for marker in markers) and not any(term in lowered for term in blocked)


def extract_medicines(text: str) -> list[Medicine]:
    """Extract medicine instructions.

    Args:
        text: Normalized OCR text.

    Returns:
        List of structured medicine entries.
    """
    medicines: list[Medicine] = []
    for line in text.splitlines():
        if not is_medicine_candidate(line):
            continue
        cleaned_line = re.sub(r"^\s*(?:\d+[\).\-]\s*)", "", line).strip()
        match = RX_LINE_PATTERN.match(cleaned_line)
        if not match:
            continue
        name = clean_medicine_name(match.group("name") or "")
        if not name:
            continue
        medicines.append(
            {
                "name": name,
                "dosage": clean_field(match.group("dosage") or ""),
                "frequency": clean_field(match.group("frequency") or ""),
                "duration": clean_field(match.group("duration") or ""),
            }
        )
    return medicines


def clean_medicine_name(value: str) -> str:
    """Clean a medicine name.

    Args:
        value: Raw medicine name.

    Returns:
        Cleaned medicine name.
    """
    name = re.sub(r"^(?:rx|tab|cap|syp|inj|tablet|capsule)\.?\s*", "", value, flags=re.IGNORECASE)
    name = re.sub(r"\b(?:mg|mcg|g|ml|daily|od|bd|bid|tds|tid|qid|days?|weeks?)\b.*$", "", name, flags=re.IGNORECASE)
    return clean_field(name)


def extract_prescription_fields(raw_text: str) -> PrescriptionFields:
    """Extract structured fields from OCR text.

    Args:
        raw_text: Raw OCR output.

    Returns:
        Structured prescription JSON-compatible dictionary.
    """
    text = normalize_text(raw_text)
    nlp = _load_spacy_model()
    return {
        "patient_name": extract_patient_name(text, nlp),
        "patient_age": extract_age(text),
        "date": extract_date(text),
        "doctor_name": extract_doctor_name(text, nlp),
        "medicines": extract_medicines(text),
        "diagnosis": extract_diagnosis(text),
        "notes": extract_notes(text),
    }
