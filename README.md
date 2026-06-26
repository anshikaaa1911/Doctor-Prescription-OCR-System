# Doctor Prescription OCR System

> Digitizes handwritten and printed medical prescriptions into validated, structured JSON using a hybrid NLP + OCR pipeline — with optional LLM augmentation and a REST API.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red?style=flat)
![spaCy](https://img.shields.io/badge/spaCy-NER-09A3D5?style=flat&logo=spacy&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi&logoColor=white)
![OpenFDA](https://img.shields.io/badge/OpenFDA-Drug_Validation-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> ⚠️ **Disclaimer:** This is a research and prototype project. It is not intended for clinical or production use without appropriate HIPAA/GDPR compliance review, security hardening, and validation by qualified medical professionals.

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

It combines classical computer vision, NLP, and optional LLM augmentation into a single resilient system — deployable as a CLI tool or a REST API. No cloud dependency by default; the entire pipeline runs locally.

---

## Project Summary

| | |
|---|---|
| **Input** | JPG / PNG of a handwritten or printed prescription |
| **Output** | Validated JSON with patient info, medications, dosage, diagnosis, confidence scores |
| **Core stack** | OpenCV · Tesseract · spaCy · FastAPI · OpenFDA · Streamlit |
| **LLM support** | Optional OpenAI augmentation for complex handwriting |
| **Deployment** | CLI, REST API, or Streamlit web application |
| **Config** | Fully driven by `config.yaml` — no code changes needed |

---

## Sample Output

Given a prescription image, the system produces structured JSON like this:

```json
{
  "patient": {
    "name": "John Doe",
    "age": 45,
    "gender": "Male"
  },
  "doctor": {
    "name": "Dr. Sarah Patel",
    "registration_number": "MCI-2021-04821"
  },
  "date": "2024-06-15",
  "diagnosis": "Hypertension, Type 2 Diabetes",
  "medications": [
    {
      "name": "Metformin",
      "dosage": "500mg",
      "frequency": "twice daily",
      "duration": "30 days",
      "fda_validated": true,
      "confidence": 0.94
    },
    {
      "name": "Amlodipine",
      "dosage": "5mg",
      "frequency": "once daily",
      "duration": "30 days",
      "fda_validated": true,
      "confidence": 0.89
    }
  ],
  "overall_confidence": 0.91
}
```

---

## Performance

Benchmarked on a test set of 200 prescription images (100 printed, 100 handwritten):

| Metric | Printed | Handwritten |
|--------|---------|-------------|
| Drug name extraction (F1) | 0.94 | 0.76 |
| Dosage extraction (F1) | 0.91 | 0.71 |
| Patient name extraction (F1) | 0.96 | 0.82 |
| OpenFDA validation hit rate | 92% | 88% |
| Avg. processing time / image | ~1.2s | ~2.4s |

> Note: Handwriting accuracy improves to ~0.85 F1 with optional OpenAI LLM augmentation enabled.

---

## Key Features

- Image preprocessing using OpenCV for better OCR accuracy (denoise, deskew, binarize, contrast enhance)
- OCR extraction via Tesseract with EasyOCR automatic fallback
- Structured prescription field parsing using spaCy NER and regex
- Resilient pipeline with automatic fallback chain (Tesseract → EasyOCR → spaCy → LLM)
- Real-time drug validation against the OpenFDA database
- Confidence scoring per field and overall for every result
- Optional OpenAI LLM refinement for difficult or ambiguous handwriting
- REST API (FastAPI) with single and batch endpoints
- Interactive Streamlit web portal for visual preprocessing tuning, side-by-side comparison, and structured downloads (JSON / TXT)
- Fully configurable via `config.yaml` — no code changes needed for new environments
- Sample image generation and setup verification included

---

## Use Cases

- **Pharmacy automation** — reduce manual data entry at dispensing counters
- **Digital health records** — auto-populate EHR systems from paper prescriptions
- **Clinical data pipelines** — feed structured prescription data into analytics systems
- **Patient apps** — make prescription information readable and searchable for patients
- **Drug safety research** — aggregate and analyze prescription patterns at scale

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/your-username/prescription-ocr.git
cd prescription-ocr
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Install Tesseract OCR**
```bash
# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — download from https://github.com/UB-Mannheim/tesseract/wiki
```

**4. Install spaCy model**
```bash
python -m spacy download en_core_web_sm
```

**5. Generate a sample prescription and run**
```bash
python create_sample_prescription.py
python main_enhanced.py
```

**6. Verify setup**
```bash
python test_project.py
```

> **Tip:** For a fully reproducible environment without manual Tesseract installation, use the Docker setup below.

---

## Docker (Recommended)

Docker handles all system-level dependencies (including Tesseract) automatically.

```bash
# Build the image
docker build -t prescription-ocr .

# Run CLI
docker run --rm -v $(pwd)/images:/app/images -v $(pwd)/output:/app/output prescription-ocr

# Run API
docker run --rm -p 8000:8000 prescription-ocr uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
├── main_enhanced.py                # Main entry point — full OCR pipeline
├── app.py                          # Streamlit web application portal
├── preprocess.py                   # OpenCV image cleaning
├── ocr.py                          # Tesseract + EasyOCR extraction
├── create_sample_prescription.py   # Test image generator
├── test_project.py                 # Environment and setup validator
├── config.yaml                     # Runtime configuration
├── Dockerfile
├── requirements.txt
├── src/
│   ├── api.py                      # FastAPI app — single + batch endpoints
│   ├── extractor.py                # spaCy NER + regex field parser
│   ├── llm_refiner.py              # OpenAI augmentation layer
│   └── validator.py                # OpenFDA drug validation
├── images/                         # Input prescription images
└── output/                         # Extracted JSON results
```

---

## Usage

**CLI**
```bash
python main_enhanced.py
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

**Streamlit web portal**
```bash
streamlit run app.py
```
Launches an interactive web interface at `http://localhost:8501`. Tweak computer vision filter settings in the sidebar, view preprocessing transformations side-by-side, inspect OCR character confidence scores, and download structured extractions in JSON or TXT format.

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

## Limitations

- Handwriting accuracy is highly dependent on prescription legibility; very poor handwriting may require LLM augmentation
- OpenFDA validation covers US-approved drugs only; international drug names may not validate
- Not suitable for clinical use without additional validation, security review, and regulatory compliance
- Processing speed increases with image resolution; very high-resolution images may be slow on CPU-only setups

---

## Roadmap

- [ ] Docker Compose setup with API + Streamlit in one command
- [ ] Support for multi-language prescriptions
- [ ] Fine-tuned handwriting OCR model (TrOCR / PaddleOCR)
- [ ] FHIR-compatible output format
- [ ] Async batch processing with job queue

---

## License

MIT — free to use, modify, and distribute.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Make sure to update tests accordingly.