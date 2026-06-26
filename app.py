"""
app.py
======
Premium Streamlit Web Application for the Doctor Prescription OCR System.
Features:
- Image & PDF Uploader with orientation and DPI correction
- Interactive Preprocessing Control Sandbox in the sidebar
- Real-time side-by-side comparison of original vs processed binary image
- Structured field visualizer (Patient, Doctor, Date, Diagnosis, Notes)
- Visually rich table of medicines with OpenFDA validation status and fuzzy matches
"""

import io
import json
import logging
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from src.preprocessor import preprocess_array
from src.ocr_engine import OCRPipeline
from src.extractor import extract_prescription_fields
from src.validator import validate_medicines
from src.tesseract_config import configure_tesseract

# Configure Tesseract path if needed
configure_tesseract()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Rx Asshirwaad OCR Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    /* Core Font Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Header layout */
    .banner {
        background: linear-gradient(135deg, #00B4DB, #0083B0);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.2);
    }
    .banner h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .banner p {
        margin: 5px 0 0 0;
        font-size: 1.05rem;
        opacity: 0.9;
    }
    
    /* Metrics Row */
    .metric-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: rgba(0, 180, 219, 0.04);
        border: 1px solid rgba(0, 180, 219, 0.12);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 25px;
    }
    .metric-box {
        text-align: center;
        flex: 1;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-box:last-child {
        border-right: none;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #00B4DB;
        font-family: 'Outfit', sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #A0AEC0;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tables and list styling */
    .med-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .med-table th {
        background: rgba(0, 180, 219, 0.1);
        color: #00B4DB;
        text-align: left;
        padding: 12px;
        font-weight: 600;
        border-bottom: 2px solid rgba(0, 180, 219, 0.2);
    }
    .med-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success {
        background: rgba(72, 187, 120, 0.15);
        color: #48BB78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .badge-warning {
        background: rgba(236, 201, 75, 0.15);
        color: #ECC94B;
        border: 1px solid rgba(236, 201, 75, 0.3);
    }
    .badge-error {
        background: rgba(245, 101, 101, 0.15);
        color: #F56565;
        border: 1px solid rgba(245, 101, 101, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Banner
st.markdown(
    """
    <div class="banner">
        <h1>🏥 Rx Asshirwaad OCR Portal</h1>
        <p>Professional Medical Prescription OCR, Advanced Preprocessing Sandbox, and FDA Validation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Preprocessing Control Sandbox
st.sidebar.markdown("### ⚙️ Preprocessing Sandbox")
st.sidebar.write("Fine-tune computer vision parameters below to adapt the thresholding to your lighting/contrast conditions.")

resize_width = st.sidebar.slider("Standard Resize Width (px)", min_value=600, max_value=2400, value=1600, step=100)
clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
clahe_tile_grid = st.sidebar.slider("CLAHE Grid Size (NxN)", min_value=2, max_value=16, value=8, step=1)
denoise_enabled = st.sidebar.checkbox("Non-Local Means Denoising", value=True)
deskew_enabled = st.sidebar.checkbox("Auto-Deskew (Hough Lines)", value=True)
adaptive_block_size = st.sidebar.slider("Adaptive Threshold Block Size", min_value=3, max_value=99, value=31, step=2)
# Ensure block size is odd
if adaptive_block_size % 2 == 0:
    adaptive_block_size += 1

adaptive_c = st.sidebar.slider("Adaptive Threshold Constant (C)", min_value=1, max_value=25, value=11, step=1)
morphology_kernel = st.sidebar.slider("Morphology Kernel Size (NxN)", min_value=1, max_value=5, value=2, step=1)

# Sidebar LLM Refinement Layer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 LLM Refinement Layer")
st.sidebar.write("Optionally run a generative AI model to clean up raw OCR inaccuracies and structure ambiguous fields.")
enable_llm = st.sidebar.checkbox("Enable LLM Refinement", value=False)

llm_provider = "local"
llm_model = "gpt-4o-mini"
llm_key = ""
api_key_env_var = "OPENAI_API_KEY"

if enable_llm:
    llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "NVIDIA NIM"])
    llm_key = st.sidebar.text_input("API Key", type="password", help="Enter your provider API Key")
    if llm_provider == "OpenAI":
        llm_model = st.sidebar.text_input("Model Name", value="gpt-4o-mini")
        api_key_env_var = "OPENAI_API_KEY"
    else:
        llm_model = st.sidebar.text_input("Model Name", value="meta/llama-3.1-8b-instruct")
        api_key_env_var = "NVIDIA_API_KEY"

# Build pipeline configuration
pipeline_config = {
    "ocr": {
        "tesseract_psm": 6,
        "confidence_threshold": 60,
        "fallback_engine": "easyocr",
        "language": "eng",
        "easyocr_languages": ["en"],
    },
    "preprocessing": {
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_grid_size": [clahe_tile_grid, clahe_tile_grid],
        "deskew": deskew_enabled,
        "denoise": denoise_enabled,
        "resize_width": resize_width,
        "adaptive_block_size": adaptive_block_size,
        "adaptive_c": adaptive_c,
        "morphology_kernel_size": [morphology_kernel, morphology_kernel],
    },
    "validator": {
        "openfda_base_url": "https://api.fda.gov/drug/label.json",
        "confidence_threshold": 0.7,
        "timeout_seconds": 5,
    },
    "validation": {
        "fuzzy_threshold": 80,
        "cache_ttl_hours": 24,
    },
    "quality_check": {
        "min_dpi": 150,
        "blur_threshold": 100,
    }
}

if enable_llm and llm_key:
    os.environ[api_key_env_var] = llm_key
    pipeline_config["llm"] = {
        "provider": "openai" if llm_provider == "OpenAI" else "nvidia",
        "model": llm_model,
        "api_key_env": api_key_env_var,
        "timeout_seconds": 20,
        "max_input_chars": 6000
    }

# Decode uploaded file helper
def decode_uploaded_file(uploaded_file):
    content = uploaded_file.read()
    extension = uploaded_file.name.split(".")[-1].lower()
    
    if extension == "pdf":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(content, first_page=1, last_page=1)
            if not pages:
                raise ValueError("PDF contains no pages")
            rgb = np.array(pages[0].convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return image, 200.0
        except Exception as e:
            st.error(f"Failed to process PDF: {e}. Please ensure `pdf2image` and Poppler are installed correctly.")
            st.stop()
    else:
        # Image file decoding
        dpi = 200.0  # Default fallback
        try:
            with Image.open(io.BytesIO(content)) as img:
                dpi_tuple = img.info.get("dpi")
                if dpi_tuple:
                    dpi = float(dpi_tuple[0])
        except Exception:
            pass
            
        try:
            with Image.open(io.BytesIO(content)) as pil_img:
                corrected_pil = ImageOps.exif_transpose(pil_img)
                rgb = np.array(corrected_pil.convert("RGB"))
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return image, dpi
        except Exception as e:
            st.error(f"Error decoding image: {e}")
            st.stop()

# Layout
st.markdown("### 📤 Upload Prescription")
uploaded_file = st.file_uploader(
    "Choose a prescription image (PNG, JPG, JPEG) or PDF first page...", 
    type=["jpg", "png", "jpeg", "pdf"]
)

if uploaded_file is not None:
    # Decode image BGR and extract estimated/actual DPI
    original_bgr, dpi = decode_uploaded_file(uploaded_file)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Run preprocessing array instantly to demonstrate Sandbox
    with st.spinner("Applying preprocessing sandbox settings..."):
        try:
            preprocess_res = preprocess_array(original_bgr, pipeline_config)
            preprocessed_binary = preprocess_res["image"]
            preprocess_metadata = preprocess_res["metadata"]
        except Exception as e:
            st.error(f"Preprocessing Sandbox error: {e}")
            st.stop()
            
    # Side-by-Side Visual Comparison
    st.markdown("### 🖼️ Real-Time Preprocessing Sandbox Output")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Original Prescription</h4>", unsafe_allow_html=True)
        st.image(original_rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Binary Preprocessed (B&W)</h4>", unsafe_allow_html=True)
        st.image(preprocessed_binary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Trigger Full OCR Pipeline Button
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Confirm that the text is clean and legible in the preprocessed output above. Then trigger the text recognition engine.")
    
    # Store results in session state to maintain state across interactions
    if "ocr_results" not in st.session_state or st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state["ocr_results"] = None
        st.session_state["last_uploaded_name"] = uploaded_file.name
        
    run_ocr = st.button("⚡ Extract & Verify Prescription Text", type="primary")
    
    if run_ocr or st.session_state["ocr_results"] is not None:
        if run_ocr:
            with st.spinner("Recognizing characters & extracting medical data..."):
                try:
                    # 1. OCR text recognition
                    ocr_res = OCRPipeline(pipeline_config).recognize(preprocessed_binary, dpi=dpi)
                    
                    # 2. Extract structured fields
                    extracted_fields = extract_prescription_fields(ocr_res["raw_text"], pipeline_config)
                    
                    # 3. Medicine Validation
                    medicine_names = [med["name"] for med in extracted_fields["medicines"] if med["name"]]
                    validation_res = validate_medicines(medicine_names, client=None)
                    
                    # Save results in session state
                    st.session_state["ocr_results"] = {
                        "ocr": ocr_res,
                        "fields": extracted_fields,
                        "validation": validation_res,
                        "metadata": preprocess_metadata
                    }
                except Exception as e:
                    st.error(f"OCR Pipeline failed: {e}")
                    st.session_state["ocr_results"] = None
                    st.stop()
                    
        # Load results from session state
        res = st.session_state["ocr_results"]
        ocr_res = res["ocr"]
        extracted_fields = res["fields"]
        validation_res = res["validation"]
        preprocess_meta = res["metadata"]
        
        # Display Metrics Row
        st.markdown("### 📊 Metrics Summary")
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-value">{ocr_res['confidence']:.1f}%</div>
                    <div class="metric-label">OCR Character Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{preprocess_meta['confidence'] * 100:.0f}%</div>
                    <div class="metric-label">Image Readiness Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{len(extracted_fields['medicines'])}</div>
                    <div class="metric-label">Medicines Detected</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{ocr_res['engine_used'].upper()}</div>
                    <div class="metric-label">Recognition Engine</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display Columns
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("<div class='glass-card'><h3 style='margin-top:0; color:#00B4DB;'>📝 Structured Prescription Details</h3>", unsafe_allow_html=True)
            
            # Sub-columns for metadata
            meta1, meta2 = st.columns(2)
            with meta1:
                st.write(f"**Patient Name:** {extracted_fields.get('patient_name') or 'N/A'}")
                st.write(f"**Age:** {extracted_fields.get('patient_age') or 'N/A'}")
                st.write(f"**Date:** {extracted_fields.get('date') or 'N/A'}")
            with meta2:
                st.write(f"**Doctor:** {extracted_fields.get('doctor_name') or 'N/A'}")
                st.write(f"**Diagnosis:** {extracted_fields.get('diagnosis') or 'N/A'}")
                st.write(f"**Notes / Advice:** {extracted_fields.get('notes') or 'N/A'}")
            
            # Formatted Text Summary construction
            summary_lines = [
                "============================================================",
                "PRESCRIPTION OCR EXTRACTION RESULTS",
                "============================================================",
                f"Date: {extracted_fields.get('date') or 'N/A'}",
                f"Patient Name: {extracted_fields.get('patient_name') or 'N/A'}",
                f"Patient Age: {extracted_fields.get('patient_age') or 'N/A'}",
                f"Doctor Name: {extracted_fields.get('doctor_name') or 'N/A'}",
                f"Diagnosis: {extracted_fields.get('diagnosis') or 'N/A'}",
                f"Notes: {extracted_fields.get('notes') or 'N/A'}",
                "",
                "Detected Medicines:",
            ]
            for med in extracted_fields.get("medicines", []):
                m_name = med.get("name") or "N/A"
                m_dose = med.get("dosage") or "N/A"
                m_freq = med.get("frequency") or "N/A"
                m_dur = med.get("duration") or "N/A"
                summary_lines.append(f"- {m_name} (Dosage: {m_dose}, Freq: {m_freq}, Duration: {m_dur})")
            summary_lines.append("============================================================")
            summary_text = "\n".join(summary_lines)

            st.markdown("<hr style='margin: 15px 0; opacity: 0.15;'>", unsafe_allow_html=True)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download JSON Results",
                    data=json.dumps(extracted_fields, indent=2),
                    file_name=f"prescription_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )
            with col_dl2:
                st.download_button(
                    label="📥 Download Text Summary",
                    data=summary_text,
                    file_name=f"prescription_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Medicines Table
            st.markdown("<div class='glass-card'><h3 style='margin-top:0; color:#00B4DB;'>💊 Detected Medicines & Verification</h3>", unsafe_allow_html=True)
            
            if not extracted_fields.get("medicines"):
                st.info("No medicines identified in the prescription text.")
            else:
                # Build custom HTML table for rich formatting
                table_html = """
                <table class="med-table">
                    <thead>
                        <tr>
                            <th>Medicine Name</th>
                            <th>Dosage</th>
                            <th>Frequency</th>
                            <th>Duration</th>
                            <th>Verification Status</th>
                            <th>Fuzzy Matches / Suggestions</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for med in extracted_fields["medicines"]:
                    name = med.get("name") or "N/A"
                    dosage = med.get("dosage") or "N/A"
                    frequency = med.get("frequency") or "N/A"
                    duration = med.get("duration") or "N/A"
                    
                    # Determine validation status
                    is_flagged = name in validation_res.get("flagged", [])
                    if is_flagged:
                        status_badge = '<span class="badge badge-warning">Low Confidence</span>'
                        # Get closest matches for this specific drug
                        matches = validation_res.get("closest_matches", {}).get(name, [])
                        suggestions_html = ", ".join(f"<code>{m}</code>" for m in matches) if matches else "No suggestion"
                    else:
                        status_badge = '<span class="badge badge-success">FDA Validated</span>'
                        suggestions_html = "<span style='color: #48BB78;'>✓ Verified Match</span>"
                        
                    table_html += f"""
                        <tr>
                            <td><strong>{name}</strong></td>
                            <td>{dosage}</td>
                            <td>{frequency}</td>
                            <td>{duration}</td>
                            <td>{status_badge}</td>
                            <td>{suggestions_html}</td>
                        </tr>
                    """
                    
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with right_col:
            st.markdown("<div class='glass-card'><h3 style='margin-top:0; color:#00B4DB;'>🔍 Raw Recognition Output</h3>", unsafe_allow_html=True)
            st.text_area("Extracted OCR Text", ocr_res["raw_text"], height=300)
            
            # Show additional pipeline info
            st.markdown("<h4>Pipeline Run Details</h4>", unsafe_allow_html=True)
            st.write(f"- **Rotation Correction:** {preprocess_meta.get('skew_angle_degrees', 0.0):+.2f}°")
            st.write(f"- **Resolution Shape:** {preprocess_meta.get('processed_shape', (0,0))[1]}x{preprocess_meta.get('processed_shape', (0,0))[0]} px")
            st.write(f"- **CLAHE Filter:** Applied")
            st.write(f"- **Adaptive Threshold:** Block={adaptive_block_size}, C={adaptive_c}")
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Please upload a prescription image or PDF to start analysis.")
