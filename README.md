# Doctor Prescription OCR System

> Digitizes handwritten and printed medical prescriptions into validated, structured JSON using a hybrid NLP + OCR pipeline — with optional LLM augmentation and a production-ready REST API.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red?style=flat)
![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5?style=flat&logo=spacy&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi&logoColor=white)
![OpenFDA](https://img.shields.io/badge/OpenFDA-Drug_Validation-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## The Problem

Every day, millions of handwritten prescriptions are issued by doctors worldwide. Despite healthcare going digital, the prescription pad remains stubbornly paper-based — and that gap has real consequences:

- **Pharmacists misread** handwriting and dispense the wrong drug or dosage
- **Patients lose prescriptions** and have no digital record to fall back on
- **Hospitals can't auto-populate** EHR systems from paper — staff transcribe manually
- **Researchers and analysts** can't easily aggregate prescription data for insights
- **Refill automation** is blocked because prescription data lives in an image, not a database

Manual transcription is slow, error-prone, and doesn't scale. A single illegible prescription can cause a medication error — one of the leading causes of preventable patient harm globally.

---

## The Solution

This project automates the full pipeline from prescription image to validated, structured data:

```
Raw Image → Preprocessing → OCR → NLP Extraction → Drug Validation → Structured JSON
```

It combines classical computer vision, NLP, and optional LLM augmentation into a single fault-tolerant system — deployable as a CLI tool or a REST API. No cloud dependency by default; the entire pipeline runs locally.

---

## Project Summary

This repository combines image preprocessing, OCR, and structured extraction to convert prescription images into validated JSON. It supports both printed and handwritten prescriptions, and includes tools for environment verification and sample input generation.

| | |
|---|---|
| **Input** | JPG / PNG of a handwritten or printed prescription |
| **Output** | Validated JSON with patient info, medications, dosage, diagnosis, confidence scores |
| **Core stack** | OpenCV · Tesseract · spaCy · FastAPI · OpenFDA |
| **LLM support** | Optional OpenAI augmentation for complex handwriting |
| **Deployment** | CLI or REST API (single image or batch) |
| **Config** | Fully driven by `config.yaml` — no code changes needed |

---

## Key Features

- Image preprocessing using OpenCV for better OCR accuracy (denoise, deskew, binarize, contrast enhance)
- OCR extraction via Tesseract with EasyOCR automatic fallback
- Structured prescription field parsing using spaCy NER and Regex
- Fault-tolerant pipeline — automatic fallback chain, no single point of failure
- Real-time drug validation against the OpenFDA database
- Confidence scoring per field and overall for every result
- Optional OpenAI LLM refinement for difficult or ambiguous handwriting
- Production-ready REST API (FastAPI) with single and batch endpoints
- Fully configurable via `config.yaml` — no code changes needed for new environments
- Sample image generation and setup verification included

---

## Use Cases

- **Pharmacy automation** — reduce manual data entry at dispensing counters
- **Digital health records** — auto-populate EHR systems from paper prescriptions
- **Clinical data pipelines** — feed structured prescription data into analytics systems
- **Patient apps** — make prescription information readable and searchable for patients
- **Drug safety research** — aggregate and analyse prescription patterns at scale

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Install Tesseract OCR**
```bash
# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — download from https://github.com/UB-Mannheim/tesseract/wiki
```

**3. Install spaCy model**
```bash
python -m spacy download en_core_web_sm
```

**4. Generate a sample prescription and run**
```bash
python create_sample_prescription.py
python main.py
```

**5. Verify setup**
```bash
python test_project.py
```

---

## Project Structure

```
├── main.py                         # Entry point
├── main_enhanced.py                # Core OCR workflow
├── preprocess.py                   # OpenCV image cleaning
├── ocr.py                          # Tesseract + EasyOCR extraction
├── create_sample_prescription.py   # Test image generator
├── test_project.py                 # Environment and setup validator
├── config.yaml                     # Runtime configuration
├── requirements.txt
├── src/
│   ├── api.py                      # FastAPI app — single + batch endpoints
│   ├── extractor.py                # spaCy NER + Regex field parser
│   ├── llm_refiner.py              # OpenAI augmentation layer
│   └── validator.py                # OpenFDA drug validation
├── images/                         # Input prescription images
└── output/                         # Extracted JSON results
```

---

## Usage

**CLI**
```bash
python main.py
```
Processes all images in `images/` and saves results to `output/`.

**API**
```bash
uvicorn src.api:app --reload
```
```bash
# Single image
curl -X POST http://localhost:8000/ocr -F "file=@prescription.jpg"

# Batch
curl -X POST http://localhost:8000/ocr/batch -F "files=@rx1.jpg" -F "files=@rx2.jpg"
```

---

## Optional LLM Augmentation

By default the pipeline runs entirely locally — no API calls, no cost, no data leaving your machine. To enable OpenAI refinement for difficult or ambiguous handwriting:

```yaml
# config.yaml
llm:
  provider: openai
  api_key_env: OPENAI_API_KEY
  fallback_on_error: true
```

If the LLM call fails or times out, the system automatically falls back to spaCy extraction — no interruption to the pipeline.

---

## Testing

```bash
python test_project.py
```

Validates: Python packages · Tesseract installation · spaCy model · required files and folders · module imports · sample image availability · basic OpenCV functionality.

---

## License

MIT — free to use, modify, and distribute.