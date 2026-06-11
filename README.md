# Doctor Prescription OCR System

A Python-based OCR application that extracts and digitizes text from prescription images using OpenCV and Tesseract OCR.

## Features

* Extracts text from prescription images
* Image preprocessing using OpenCV
* OCR-based text recognition using Tesseract
* Generates machine-readable text output

## Tech Stack

* Python
* OpenCV
* Tesseract OCR
* Pillow

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd DoctorPrescriptionOCR
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR and ensure it is added to your system PATH.

## Usage

1. Place prescription images inside the `images/` folder.

2. Run the application:

```bash
python main.py
```

3. Extracted text will be displayed in the terminal and saved to the `output/` folder.

## Project Structure

```text
DoctorPrescriptionOCR/
│
├── images/
├── output/
├── main.py
├── preprocess.py
├── ocr.py
├── requirements.txt
└── README.md
```

## Future Enhancements

* Handwritten prescription recognition
* Medicine name and dosage extraction
* Web-based interface for image uploads
* Improved OCR accuracy using deep learning models
