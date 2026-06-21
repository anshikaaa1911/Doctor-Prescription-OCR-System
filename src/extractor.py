"""Structured prescription field extraction using regex and local LLM helpers."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from src.llm import extract_person_entities, extract_prescription_with_openai, load_llm_pipeline, should_use_openai

logger = logging.getLogger(__name__)


class Medicine(TypedDict):
    """Structured medicine entry."""

    name: str | None
    dosage: str | None
    frequency: str | None
    duration: str | None
    confidence: float
    confidences: dict[str, float]


class PrescriptionFields(TypedDict):
    """Structured prescription fields."""

    patient_name: str | None
    patient_age: str | None
    date: str | None
    doctor_name: str | None
    medicines: list[Medicine]
    diagnosis: str | None
    notes: str | None
    confidences: dict[str, float]


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

FREQUENCY_MAP = {
    "OD": "Once daily",
    "BD": "Twice daily",
    "BID": "Twice daily",
    "TDS": "Three times daily",
    "TID": "Three times daily",
    "QID": "Four times daily",
    "SOS": "As needed",
    "PRN": "As needed"
}


def normalize_text(text: str) -> str:
    """Normalize OCR text for extraction.

    Args:
        text: Raw OCR text.

    Returns:
        Whitespace-normalized text.
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def extract_first(patterns: list[re.Pattern[str]], text: str) -> str | None:
    """Extract the first named regex group value.

    Args:
        patterns: Regex patterns containing a value group.
        text: Text to search.

    Returns:
        Matched value or None.
    """
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            val = clean_field(match.group("value"))
            return val if val else None
    return None


def clean_field(value: str) -> str:
    """Clean extracted field text.

    Args:
        value: Extracted field value.

    Returns:
        Cleaned field value.
    """
    return re.sub(r"\s+", " ", value).strip(" .:-")


def extract_patient_name(text: str, llm: Any | None = None) -> str | None:
    """Extract patient name.

    Args:
        text: Normalized OCR text.
        llm: Optional language pipeline.

    Returns:
        Patient name or None.
    """
    regex_value = extract_first(PATIENT_PATTERNS, text)
    if regex_value:
        return regex_value
    people = [entity[0] for entity in extract_person_entities(text, llm)]
    val = clean_field(people[0]) if people else None
    return val if val else None


def extract_doctor_name(text: str, llm: Any | None = None) -> str | None:
    """Extract doctor name.

    Args:
        text: Normalized OCR text.
        llm: Optional language pipeline.

    Returns:
        Doctor name or None.
    """
    match = DOCTOR_PATTERN.search(text)
    if match:
        value = clean_field(match.group("value"))
        val = value if value.lower().startswith("dr") else f"Dr. {value}"
        return val if val else None
    for entity_text, start_char, _end_char in extract_person_entities(text, llm):
        if "dr" in text[max(0, start_char - 10) : start_char].lower():
            val = clean_field(entity_text)
            return val if val else None
    return None


def extract_age(text: str) -> str | None:
    """Extract patient age.

    Args:
        text: Normalized OCR text.

    Returns:
        Age string or None.
    """
    match = AGE_PATTERN.search(text)
    val = clean_field(match.group("value")) if match else None
    return val if val else None


def extract_date(text: str) -> str | None:
    """Extract prescription date.

    Args:
        text: Normalized OCR text.

    Returns:
        Date string or None.
    """
    match = DATE_PATTERN.search(text)
    val = clean_field(match.group("value")) if match else None
    return val if val else None


def extract_diagnosis(text: str) -> str | None:
    """Extract diagnosis text.

    Args:
        text: Normalized OCR text.

    Returns:
        Diagnosis or None.
    """
    match = DIAGNOSIS_PATTERN.search(text)
    val = clean_field(match.group("value")) if match else None
    return val if val else None


def extract_notes(text: str) -> str | None:
    """Extract notes or advice.

    Args:
        text: Normalized OCR text.

    Returns:
        Notes or None.
    """
    values = [clean_field(match.group("value")) for match in NOTES_PATTERN.finditer(text)]
    val = " ".join(value for value in values if value)
    return val if val else None


def is_medicine_candidate(line: str) -> bool:
    """Determine whether a line likely contains a medicine.

    Args:
        line: OCR text line.

    Returns:
        True when the line resembles a medicine instruction.
    """
    lowered = line.lower()
    markers = ["tab", "cap", "syp", "inj", "rx", "mg", "ml", "daily", "bd", "tds", "bid", "tid", "qid", "days", "weeks", "months"]
    blocked = ["patient", "date", "doctor", "diagnosis", "notes", "advice", "age"]
    
    # Check if starts with list marker (digits followed by . or - or ), or bullet points like *, -, •)
    has_list_marker = bool(re.match(r"^\s*(?:\d+[\).\-]\s*|[*•\-]\s*)", line))
    
    if has_list_marker and not any(term in lowered for term in blocked):
        cleaned = re.sub(r"^\s*(?:\d+[\).\-]\s*|[*•\-]\s*)", "", line).strip()
        if len(cleaned) >= 3:
            return True
            
    return any(marker in lowered for marker in markers) and not any(term in lowered for term in blocked)


def map_frequency(freq: str | None) -> str | None:
    """Map abbreviated frequency to human-readable format.

    Args:
        freq: Frequency abbreviation.

    Returns:
        Human-readable frequency or the original if not in map.
    """
    if not freq:
        return None
    normalized = freq.replace(".", "").strip().upper()
    return FREQUENCY_MAP.get(normalized, freq)


def _calculate_medicine_confidences(
    name: str | None,
    dosage: str | None,
    frequency_raw: str | None,
    duration: str | None
) -> dict[str, float]:
    """Calculate confidence scores for medicine fields based on regex match quality.

    Args:
        name: Medicine name.
        dosage: Dosage.
        frequency_raw: Raw frequency.
        duration: Duration.

    Returns:
        Dict of confidence scores.
    """
    conf = {"name": 0.0, "dosage": 0.0, "frequency": 0.0, "duration": 0.0}

    # Name confidence
    if name:
        score = 0.9
        if len(name) < 3 or len(name) > 30:
            score -= 0.15
        if any(c in name for c in "@#%_*"):
            score -= 0.2
        conf["name"] = max(0.1, min(score, 1.0))

    # Dosage confidence
    if dosage:
        if re.search(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)", dosage, re.IGNORECASE):
            score = 0.95
        else:
            score = 0.6
        conf["dosage"] = max(0.1, min(score, 1.0))

    # Frequency confidence
    if frequency_raw:
        normalized = frequency_raw.replace(".", "").strip().upper()
        if normalized in FREQUENCY_MAP:
            score = 0.95
        elif normalized in ["ONCE", "TWICE", "THRICE", "DAILY"]:
            score = 0.9
        else:
            score = 0.7
        conf["frequency"] = max(0.1, min(score, 1.0))

    # Duration confidence
    if duration:
        if re.search(r"\d+\s*(?:days?|weeks?|months?)", duration, re.IGNORECASE):
            score = 0.95
        else:
            score = 0.7
        conf["duration"] = max(0.1, min(score, 1.0))

    return {k: round(v, 2) for k, v in conf.items()}


def _calculate_field_confidences(fields: dict[str, Any], text: str) -> dict[str, float]:
    """Calculate confidence scores for top-level fields based on regex match quality.

    Args:
        fields: Dict of extracted fields.
        text: Normalized OCR text.

    Returns:
        Dict of confidence scores.
    """
    conf = {
        "patient_name": 0.0,
        "patient_age": 0.0,
        "date": 0.0,
        "doctor_name": 0.0,
        "diagnosis": 0.0,
        "notes": 0.0,
    }

    # patient_name
    val = fields.get("patient_name")
    if val:
        has_regex = any(p.search(text) for p in PATIENT_PATTERNS)
        score = 0.9 if has_regex else 0.75
        if len(val) < 3 or len(val) > 40:
            score -= 0.15
        if any(c.isdigit() for c in val):
            score -= 0.3
        conf["patient_name"] = max(0.1, min(score, 1.0))

    # patient_age
    val = fields.get("patient_age")
    if val:
        score = 0.95
        if not val.isdigit():
            score = 0.6
        else:
            age_int = int(val)
            if age_int <= 0 or age_int > 120:
                score -= 0.4
        conf["patient_age"] = max(0.1, min(score, 1.0))

    # date
    val = fields.get("date")
    if val:
        score = 0.95
        if not re.search(r"\d", val):
            score = 0.5
        conf["date"] = max(0.1, min(score, 1.0))

    # doctor_name
    val = fields.get("doctor_name")
    if val:
        has_regex = bool(DOCTOR_PATTERN.search(text))
        score = 0.9 if has_regex else 0.75
        if val.lower().startswith("dr"):
            score += 0.05
        if len(val) < 3 or len(val) > 40:
            score -= 0.15
        conf["doctor_name"] = max(0.1, min(score, 1.0))

    # diagnosis
    val = fields.get("diagnosis")
    if val:
        score = 0.85
        conf["diagnosis"] = max(0.1, min(score, 1.0))

    # notes
    val = fields.get("notes")
    if val:
        score = 0.85
        conf["notes"] = max(0.1, min(score, 1.0))

    return {k: round(v, 2) for k, v in conf.items()}


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
        cleaned_line = re.sub(r"^\s*(?:\d+[\).\-]\s*|[*•\-]\s*|rx\b\.?\s*[:\-]?\s*)", "", line, flags=re.IGNORECASE).strip()
        match = RX_LINE_PATTERN.match(cleaned_line)
        if not match:
            continue
        name = clean_medicine_name(match.group("name") or "")
        if not name:
            continue

        dosage = clean_field(match.group("dosage") or "")
        frequency = clean_field(match.group("frequency") or "")
        duration = clean_field(match.group("duration") or "")

        mapped_frequency = map_frequency(frequency)

        name_val = name if name else None
        dosage_val = dosage if dosage else None
        frequency_val = mapped_frequency if mapped_frequency else None
        duration_val = duration if duration else None

        med_confidences = _calculate_medicine_confidences(name_val, dosage_val, frequency, duration_val)
        avg_conf = sum(med_confidences.values()) / len(med_confidences) if med_confidences else 0.0

        medicines.append(
            {
                "name": name_val,
                "dosage": dosage_val,
                "frequency": frequency_val,
                "duration": duration_val,
                "confidence": round(avg_conf, 2),
                "confidences": med_confidences
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


def _normalize_openai_result(result: dict[str, Any], fallback: PrescriptionFields) -> PrescriptionFields:
    """Normalize an OpenAI extraction result to the expected prescription shape."""
    medicines: list[Medicine] = []
    for medicine in result.get("medicines", []):
        if not isinstance(medicine, dict):
            continue
        confidences = medicine.get("confidences") if isinstance(medicine.get("confidences"), dict) else {}
        med_conf = {
            "name": float(confidences.get("name", 0.0)),
            "dosage": float(confidences.get("dosage", 0.0)),
            "frequency": float(confidences.get("frequency", 0.0)),
            "duration": float(confidences.get("duration", 0.0)),
        }
        medicines.append(
            {
                "name": clean_field(str(medicine["name"])) if medicine.get("name") else None,
                "dosage": clean_field(str(medicine["dosage"])) if medicine.get("dosage") else None,
                "frequency": clean_field(str(medicine["frequency"])) if medicine.get("frequency") else None,
                "duration": clean_field(str(medicine["duration"])) if medicine.get("duration") else None,
                "confidence": round(max(0.0, min(float(medicine.get("confidence", 0.0)), 1.0)), 2),
                "confidences": {key: round(max(0.0, min(value, 1.0)), 2) for key, value in med_conf.items()},
            }
        )

    confidences = result.get("confidences") if isinstance(result.get("confidences"), dict) else {}
    field_conf = {
        "patient_name": float(confidences.get("patient_name", 0.0)),
        "patient_age": float(confidences.get("patient_age", 0.0)),
        "date": float(confidences.get("date", 0.0)),
        "doctor_name": float(confidences.get("doctor_name", 0.0)),
        "diagnosis": float(confidences.get("diagnosis", 0.0)),
        "notes": float(confidences.get("notes", 0.0)),
        "medicines": float(confidences.get("medicines", 0.0)),
    }

    return {
        "patient_name": clean_field(str(result["patient_name"])) if result.get("patient_name") else None,
        "patient_age": clean_field(str(result["patient_age"])) if result.get("patient_age") else None,
        "date": clean_field(str(result["date"])) if result.get("date") else None,
        "doctor_name": clean_field(str(result["doctor_name"])) if result.get("doctor_name") else None,
        "medicines": medicines or fallback["medicines"],
        "diagnosis": clean_field(str(result["diagnosis"])) if result.get("diagnosis") else None,
        "notes": clean_field(str(result["notes"])) if result.get("notes") else None,
        "confidences": {key: round(max(0.0, min(value, 1.0)), 2) for key, value in field_conf.items()},
    }


def extract_prescription_fields(raw_text: str, config: dict[str, Any] | None = None) -> PrescriptionFields:
    """Extract structured fields from OCR text.

    Args:
        raw_text: Raw OCR output.
        config: Optional application configuration.

    Returns:
        Structured prescription JSON-compatible dictionary.
    """
    text = normalize_text(raw_text)
    llm = load_llm_pipeline()

    patient_name = extract_patient_name(text, llm)
    patient_age = extract_age(text)
    date = extract_date(text)
    doctor_name = extract_doctor_name(text, llm)
    medicines = extract_medicines(text)
    diagnosis = extract_diagnosis(text)
    notes = extract_notes(text)

    fields = {
        "patient_name": patient_name if patient_name else None,
        "patient_age": patient_age if patient_age else None,
        "date": date if date else None,
        "doctor_name": doctor_name if doctor_name else None,
        "diagnosis": diagnosis if diagnosis else None,
        "notes": notes if notes else None,
    }

    field_conf = _calculate_field_confidences(fields, text)
    med_confs = [m["confidence"] for m in medicines]
    field_conf["medicines"] = round(sum(med_confs) / len(med_confs) if med_confs else 0.0, 2)

    local_result: PrescriptionFields = {
        "patient_name": fields["patient_name"],
        "patient_age": fields["patient_age"],
        "date": fields["date"],
        "doctor_name": fields["doctor_name"],
        "medicines": medicines,
        "diagnosis": fields["diagnosis"],
        "notes": fields["notes"],
        "confidences": field_conf,
    }

    if should_use_openai(config):
        openai_result = extract_prescription_with_openai(raw_text, local_result, config)
        if openai_result:
            return _normalize_openai_result(openai_result, local_result)

    return local_result
