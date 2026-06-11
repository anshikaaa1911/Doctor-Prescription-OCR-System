# ⚡ QUICK START GUIDE

**Get the Doctor Prescription OCR project running in 5 minutes!**

---

## 🎯 TL;DR - Three Commands

```powershell
# 1. Install Python packages
pip install -r requirements.txt

# 2. Create a sample prescription image for testing
python create_sample_prescription.py

# 3. Run the OCR
python main.py
```

Done! Check the `output/` folder for extracted text.

---

## ✅ Quick Verification

Before running, check that you have everything:

```powershell
# Run this to verify setup
python test_project.py
```

This will tell you if anything is missing.

---

## 🔧 One-Time Setup: Install Tesseract

You only need to do this once.

### Windows:
1. Go to: https://github.com/UB-Mannheim/tesseract/wiki
2. Download and run the installer
3. Use default installation path
4. Done!

### macOS:
```bash
brew install tesseract
```

### Linux:
```bash
sudo apt-get install tesseract-ocr
```

---

## 📁 Your First Run

### Option 1: Use Sample Image (Recommended for Testing)

```powershell
# Generate a sample prescription image
python create_sample_prescription.py

# Run OCR on it
python main.py
```

**Output:** Check `output/prescription_*.txt` for results

### Option 2: Use Your Own Image

1. Get a prescription image (JPG, PNG, BMP, or TIFF)
2. Copy it to the `images/` folder
3. Run: `python main.py`

---

## 📊 Expected Output

You should see something like this in terminal:

```
============================================================
DOCTOR PRESCRIPTION OCR
Automatic Text Extraction from Prescription Images
============================================================

Searching for images in 'images' folder...
✓ Found 1 image(s) to process

[1/1] PREPROCESSING IMAGE...
✓ Image converted to grayscale
✓ Gaussian blur applied
✓ Thresholding applied
✓ Morphological operations applied

[2/2] EXTRACTING TEXT WITH OCR...
✓ Text extraction completed successfully!

============================================================
PRESCRIPTION TEXT
============================================================
Dr. Michael Johnson
Patient Name: James Smith
Amoxicillin 500mg
Take one tablet three times daily
...
============================================================
✓ Text saved to: output/prescription_20240612_143022.txt
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'cv2'" | Run: `pip install -r requirements.txt` |
| "Tesseract not found" | Install from: https://github.com/UB-Mannheim/tesseract/wiki |
| "No images found" | Run: `python create_sample_prescription.py` first |
| Poor OCR results | Try a clearer image or adjust thresholds in `preprocess.py` |

---

## 📚 Next Steps

1. **Explore the code**: Read comments in `main.py`, `preprocess.py`, `ocr.py`
2. **Understand OCR**: See "How It Works" in `README.md`
3. **Modify parameters**: Experiment with blur size and threshold values
4. **Try different images**: Test with various prescription formats
5. **Learn more**: Check links in `README.md` for OpenCV and Tesseract docs

---

## 📂 Project Files Explained

```
HandWriting/
├── main.py              👈 Run this (python main.py)
├── preprocess.py        📷 Image preprocessing
├── ocr.py               📖 Text extraction
├── test_project.py      ✓ Verify setup
├── create_sample...     📋 Create test image
├── images/              📁 Put images here
├── output/              📁 Results saved here
├── README.md            📚 Full documentation
├── SETUP.md             🔧 Detailed setup
├── QUICK_START.md       ⚡ This file
└── requirements.txt     📦 Dependencies
```

---

## 🚀 You're Ready!

**Run this now:**
```powershell
python create_sample_prescription.py
python main.py
```

**Questions?** Check `README.md` for detailed explanations.

**Happy OCR-ing! 🎉**
