#  Doctor Prescription OCR System

> Digitizes handwritten and printed medical prescriptions into structured, machine-readable JSON using a classical NLP + Computer Vision pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What it does

Takes a prescription image like this and returns structured JSON:

```
Image → OpenCV preprocessing → Tesseract OCR → spaCy/Regex extraction → JSON
```

```json
{
  "patient_name": "John Doe",
  "patient_age": "34",
  "date": "12/06/2025",
  "doctor_name": "Dr. Sharma",
  "medicines": [
    {
      "name": "Paracetamol",
      "dosage": "500mg",
      "frequency": "Twice daily",
      "duration": "5 days",
      "confidence": 0.92
    }
  ],
  "diagnosis": "Viral fever",
  "notes": "Take with food"
}
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Image Processing | OpenCV | Grayscale, CLAHE, deskew, denoise, threshold |
| OCR Engine | Tesseract + EasyOCR | Text extraction with fallback |
| NLP Extraction | Regex + spaCy | Field parsing and entity recognition |
| Drug Validation | OpenFDA API + rapidfuzz | Medicine name verification |
| API | FastAPI | REST endpoints for integration |
| Testing | pytest | Unit and integration tests |

---

## Project Structure

```
DoctorPrescriptionOCR/
│
├── src/
│   ├── preprocessor.py      # OpenCV image cleaning pipeline
│   ├── ocr_engine.py        # Tesseract + EasyOCR abstraction
│   ├── extractor.py         # Regex + spaCy field extraction
│   ├── validator.py         # OpenFDA drug name validation
│   └── api.py               # FastAPI REST endpoints
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_extractor.py
│   └── test_api.py
│
├── images/                  # Input prescription images
├── output/                  # Extracted JSON output
├── config.yaml              # All tunable parameters
├── main_enhanced.py         # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DoctorPrescriptionOCR.git
cd DoctorPrescriptionOCR
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 5. Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

## Usage

### CLI

```bash
# Single image
python main_enhanced.py --image images/prescription.jpg

# Batch folder
python main_enhanced.py --folder images/
```

### API

```bash
uvicorn src.api:app --reload
```

```bash
# Extract from image
curl -X POST http://localhost:8000/ocr \
  -F "file=@prescription.jpg"

# Health check
curl http://localhost:8000/health

# Batch (up to 10 images)
curl -X POST http://localhost:8000/ocr/batch \
  -F "files=@rx1.jpg" -F "files=@rx2.jpg"
```

---

## Configuration

All parameters live in `config.yaml` — no hardcoded values:

```yaml
ocr:
  tesseract_psm: 6
  confidence_threshold: 60
  fallback_engine: easyocr

preprocessing:
  clahe_clip_limit: 2.0
  deskew: true
  denoise: true

validation:
  fuzzy_threshold: 80
  cache_ttl_hours: 24

api:
  rate_limit_per_minute: 20
  max_batch_size: 10
  max_file_size_mb: 10
```

---

## Extraction Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│     preprocessor.py         │
│  • Auto-rotation correction │
│  • CLAHE contrast enhance   │
│  • Deskew (Hough transform) │
│  • Adaptive thresholding    │
│  • Noise removal            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       ocr_engine.py         │
│  • DPI / blur quality check │
│  • Tesseract PSM auto-select│
│  • EasyOCR fallback < 60%   │
│  • Returns word bounding box│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       extractor.py          │
│  • Regex field extraction   │
│  • spaCy NER fallback       │
│  • Frequency normalization  │
│  • Per-field confidence score│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       validator.py          │
│  • Fuzzy name matching      │
│  • OpenFDA API lookup       │
│  • 24hr local cache         │
│  • Top-3 suggestions        │
└────────────┬────────────────┘
             │
             ▼
        Structured JSON
```

---

## Confidence Scores

Every extracted field includes a confidence score (0.0–1.0):

| Score | Meaning |
|---|---|
| `0.9 – 1.0` | High confidence — strong regex or known abbreviation |
| `0.7 – 0.9` | Medium — partial match or spaCy NER fallback |
| `< 0.7` | Low — flagged for manual review |

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Limitations

- Works best on printed prescriptions; handwritten support is partial
- OCR accuracy depends on image quality (min 150 DPI recommended)
- Drug validation requires internet access for OpenFDA API

---

## Roadmap

- [ ] LLM fallback layer for low-confidence extractions
- [ ] TrOCR integration for handwritten prescriptions
- [ ] Fine-tuned NER model on medical prescription dataset
- [ ] Docker container + docker-compose
- [ ] Web UI for image upload and result review
- [ ] FHIR-compatible output format

---

## Author

Built as a learning project covering OCR, computer vision, classical NLP, and REST API design.

---

## License

MIT License — free to use, modify, and distribute.d interface for image uploads
* Improved OCR accuracy using deep learning models
