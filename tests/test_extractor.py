"""Tests for prescription field extraction."""

from __future__ import annotations

import pytest

from src.extractor import extract_prescription_fields


@pytest.fixture()
def mock_ocr_outputs() -> list[str]:
    """Create mock OCR outputs.

    Returns:
        Five representative OCR text samples.
    """
    return [
        """
        Patient: John Doe
        Age: 45
        Date: 12/05/2026
        Dr. Alice Smith
        Diagnosis: Hypertension
        Rx Tab Amlodipine 5mg OD for 30 days
        Notes: Review after one month
        """,
        """
        Name - Maria Gomez
        Aged 32 years
        Dt: 2026-06-01
        Doctor: Brian Clark
        Dx: Acute bronchitis
        1. Cap Amoxicillin 500mg TDS 7 days
        Advice: Take after food
        """,
        """
        Pt: Rakesh Kumar
        Age 60 y/o
        Date 01-06-2026
        Physician: Dr Nina Patel
        Assessment: Diabetes mellitus
        Tab Metformin 500mg BD for 90 days
        Remarks: Monitor fasting sugar
        """,
        """
        Patient Name: Emily Stone
        Age: 8
        Date: 6.2.2026
        Dr Robert Hall
        Diagnosis: Fever
        Syp Paracetamol 120ml TID 5 days
        Notes: Plenty of fluids
        """,
        """
        Patient: Omar Khan
        Age: 54 yrs
        Date: 15/06/26
        Dr. Sara Lee
        Diagnosis: Gastritis
        Rx Pantoprazole 40mg OD 14 days
        Rx Domperidone 10mg BD 5 days
        Advice: Avoid spicy food
        """,
    ]


def test_extract_patient_name_from_mock_outputs(mock_ocr_outputs: list[str]) -> None:
    """Test patient-name regex extraction across sample OCR text."""
    expected = ["John Doe", "Maria Gomez", "Rakesh Kumar", "Emily Stone", "Omar Khan"]
    for text, name in zip(mock_ocr_outputs, expected):
        assert extract_prescription_fields(text)["patient_name"] == name


def test_extract_age_from_mock_outputs(mock_ocr_outputs: list[str]) -> None:
    """Test age regex extraction across sample OCR text."""
    expected = ["45", "32", "60", "8", "54"]
    for text, age in zip(mock_ocr_outputs, expected):
        assert extract_prescription_fields(text)["patient_age"] == age


def test_extract_date_from_mock_outputs(mock_ocr_outputs: list[str]) -> None:
    """Test date regex extraction across sample OCR text."""
    expected = ["12/05/2026", "2026-06-01", "01-06-2026", "6.2.2026", "15/06/26"]
    for text, date in zip(mock_ocr_outputs, expected):
        assert extract_prescription_fields(text)["date"] == date


def test_extract_diagnosis_and_notes(mock_ocr_outputs: list[str]) -> None:
    """Test diagnosis and notes extraction."""
    first = extract_prescription_fields(mock_ocr_outputs[0])
    assert first["diagnosis"] == "Hypertension"
    assert first["notes"] == "Review after one month"


def test_extract_medicines_from_mock_outputs(mock_ocr_outputs: list[str]) -> None:
    """Test medicine extraction from all mock OCR outputs."""
    extracted = [extract_prescription_fields(text) for text in mock_ocr_outputs]
    assert extracted[0]["medicines"][0]["name"] == "Amlodipine"
    assert extracted[1]["medicines"][0]["dosage"] == "500mg"
    assert extracted[2]["medicines"][0]["frequency"].lower() == "bd"
    assert extracted[3]["medicines"][0]["duration"] == "5 days"
    assert len(extracted[4]["medicines"]) == 2
