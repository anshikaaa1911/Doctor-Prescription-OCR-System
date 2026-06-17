"""Medicine validation using the OpenFDA drug label API."""

from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import Any, TypedDict
from urllib.parse import quote

import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationResult(TypedDict):
    """Medicine validation result."""

    valid: bool
    suggestions: list[str]
    flagged: list[str]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration.

    Args:
        config_path: Optional config path.

    Returns:
        Parsed config dictionary.
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


def normalize_drug_name(name: str) -> str:
    """Normalize a drug name for API lookup.

    Args:
        name: Raw drug name.

    Returns:
        Normalized drug name.
    """
    return " ".join(name.strip().split())


def score_candidate(query: str, candidate: str) -> float:
    """Score similarity between a query and candidate drug name.

    Args:
        query: Queried drug name.
        candidate: Candidate drug name from OpenFDA.

    Returns:
        Similarity score from 0.0 to 1.0.
    """
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    if query_lower == candidate_lower:
        return 1.0
    if query_lower in candidate_lower or candidate_lower in query_lower:
        return 0.85
    matches = get_close_matches(query_lower, [candidate_lower], n=1, cutoff=0.0)
    return 0.75 if matches else 0.0


def search_openfda(
    medicine_name: str,
    client: httpx.Client,
    base_url: str,
    timeout_seconds: float,
) -> tuple[float, list[str]]:
    """Search OpenFDA for a medicine name.

    Args:
        medicine_name: Medicine name to validate.
        client: HTTP client.
        base_url: OpenFDA endpoint URL.
        timeout_seconds: Request timeout.

    Returns:
        Best confidence score and candidate suggestions.
    """
    query = quote(f'openfda.brand_name:"{medicine_name}" openfda.generic_name:"{medicine_name}"')
    url = f"{base_url}?search={query}&limit=5"
    try:
        response = client.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return 0.0, []
        logger.warning("OpenFDA returned HTTP %s for %s", exc.response.status_code, medicine_name)
        return 0.0, []
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("OpenFDA lookup failed for %s: %s", medicine_name, exc)
        return 0.0, []

    suggestions: list[str] = []
    scores: list[float] = []
    for result in payload.get("results", []):
        openfda = result.get("openfda", {})
        names = openfda.get("brand_name", []) + openfda.get("generic_name", [])
        for candidate in names:
            candidate_text = normalize_drug_name(str(candidate))
            if candidate_text and candidate_text not in suggestions:
                suggestions.append(candidate_text)
                scores.append(score_candidate(medicine_name, candidate_text))

    return (max(scores) if scores else 0.0), suggestions[:5]


def validate_medicines(
    medicine_names: list[str],
    config_path: Path | None = None,
    client: httpx.Client | None = None,
) -> ValidationResult:
    """Validate medicine names against OpenFDA.

    Args:
        medicine_names: Medicine names to validate.
        config_path: Optional config path.
        client: Optional injected HTTP client for tests.

    Returns:
        Validation status, suggestions, and flagged names.
    """
    config = load_config(config_path)
    settings = config.get("validator", {})
    threshold = float(settings.get("confidence_threshold", 0.7))
    base_url = str(settings.get("openfda_base_url", "https://api.fda.gov/drug/label.json"))
    timeout_seconds = float(settings.get("timeout_seconds", 5))

    suggestions: list[str] = []
    flagged: list[str] = []
    close_client = client is None
    http_client = client or httpx.Client()
    try:
        for raw_name in medicine_names:
            name = normalize_drug_name(raw_name)
            if not name:
                continue
            score, candidates = search_openfda(name, http_client, base_url, timeout_seconds)
            suggestions.extend(candidate for candidate in candidates if candidate not in suggestions)
            if score < threshold:
                flagged.append(name)
    finally:
        if close_client:
            http_client.close()

    return {"valid": len(flagged) == 0, "suggestions": suggestions, "flagged": flagged}
