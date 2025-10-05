import os
import sys
import json
from pathlib import Path
import pytesseract
import streamlit as st
import pandas as pd


# OCR path
pytesseract.pytesseract.tesseract_cmd = "Path of tesseract.exe" #like r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ensure `src` is importable when running from project root
ROOT = os.path.abspath(os.getcwd())
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)


# imports
from extractors import document_extractor, ocr_extractor
from cleansing import engine as cleansing_engine
from analysis import llm_analysis


# dirs
RAW_DIR = os.path.join(ROOT, "data", "raw")
EXTRACTED_DIR = os.path.join(ROOT, "data", "extracted")
CLEANSED_DIR = os.path.join(ROOT, "data", "cleansed")
OUTPUTS_DIR = os.path.join(ROOT, "data", "outputs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(CLEANSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# allowed formats
ALLOWED = [".pptx", ".pdf", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"]


# ---------------- UI CONFIG ---------------- #
st.set_page_config(
    page_title="File Cleansing & Analysis",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="collapsed"
)


# custom CSS
st.markdown("""
    <style>
        /* Main page background */
        .stApp {
            background-color: #f7f9fb;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        /* Title styling */
        h1 {
            text-align: center;
            color: #1f4e79;
            font-family: 'Segoe UI', sans-serif;
            padding-bottom: 0.3em;
            margin-bottom: 0.5em;
            font-weight: 700;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        }
        /* Upload box */
        .uploadedFile {
            border-radius: 12px !important;
            border: 2px dashed #1f77b4 !important;
            background: rgba(255,255,255,0.9) !important;
            backdrop-filter: blur(10px);
        }
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #1f77b4 0%, #155a8a 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8em 2em;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
        }
        /* Info and success messages */
        .stAlert {
            border-radius: 12px !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        /* Tables */
        .dataframe {
            border: 1px solid #ddd !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background: white;
        }
        /* Expander text box */
        .stCodeBlock {
            background-color: #fafafa !important;
            border-radius: 12px;
            border: 1px solid #e6e6e6;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }
        /* File uploader label */
        .stFileUploader label {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #1f4e79 !important;
        }
        /* Success message enhancement */
        .stSuccess {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        /* Error message enhancement */
        .stError {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        /* Info message enhancement */
        .stInfo {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            max-width: 95%;
        }
        /* Hide sidebar */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)


# ---------------- HEADER ---------------- #
st.markdown("<h1>📄 File Cleansing & Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#2c3e50; font-size:18px; margin-bottom:2rem;'>Upload your document or image and process it through extraction → cleansing → analysis</p>", unsafe_allow_html=True)


# ---------------- FUNCTIONS ---------------- #
def save_uploaded_file(uploaded_file, dest_dir=RAW_DIR):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(dest_dir, uploaded_file.name)
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path


def normalize_and_save_extracted(file_name, file_type, raw_text, description, metadata, target_dir=EXTRACTED_DIR):
    base = Path(file_name).stem
    out_name = f"{base}_extracted.txt"
    result = {
        "file_name": file_name,
        "file_type": file_type,
        "raw_text": raw_text,
        "file_description": description,
        "metadata": metadata or {}
    }
    out_path = os.path.join(target_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))
    return out_path, result


def cleanse_and_save(extracted_result, target_dir=CLEANSED_DIR):
    base = Path(extracted_result["file_name"]).stem
    cleansed = extracted_result.copy()
    cleansed["raw_text"] = cleansing_engine.cleanse_text(extracted_result.get("raw_text",""))
    cleansed["file_description"] = cleansing_engine.cleanse_text(extracted_result.get("file_description",""))
    out_path = os.path.join(target_dir, f"{base}_cleansed.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cleansed, indent=2, ensure_ascii=False))
    return out_path, cleansed


def analyze_and_save(cleansed_dict, api_key: str, target_dir=OUTPUTS_DIR):
    configured = llm_analysis.configure_api_with_key(api_key)
    if not configured:
        raise RuntimeError("Could not configure Gemini API with provided key.")
    file_description = cleansed_dict.get("file_description", "")
    raw_text = cleansed_dict.get("raw_text", "")
    raw_out = llm_analysis.analyze_cleansed_text(file_description, raw_text)
    parsed = llm_analysis.parse_analysis_output(raw_out)
    final = {
        "file_name": cleansed_dict.get("file_name"),
        "file_type": cleansed_dict.get("file_type"),
        "file_description": parsed.get("description", ""),
        "key_findings": parsed.get("findings", "")
    }
    base = Path(cleansed_dict.get("file_name")).stem
    out_path = os.path.join(target_dir, f"{base}_analyzed.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final, indent=2, ensure_ascii=False))
    return out_path, final


def run_pipeline_for_file(saved_path: str, api_key: str=None):
    ext = Path(saved_path).suffix.lower()
    fname = Path(saved_path).name


    if ext in [".pptx", ".pdf", ".xlsx", ".xls"]:
        res = document_extractor.process_file(saved_path, output_dir=EXTRACTED_DIR, use_ocr=True)
        out_path = res.get("output_path")
        if out_path and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_text = data.get("text", "")
            description = data.get("metadata", {}).get("description", "") or ""
            metadata = data.get("metadata", {})
            extracted_path, extracted_result = normalize_and_save_extracted(fname, ext, raw_text, description, metadata)
        else:
            raw_text = res.get("text","")
            extracted_path, extracted_result = normalize_and_save_extracted(fname, ext, raw_text, "", {})
    elif ext in [".png", ".jpg", ".jpeg"]:
        result = ocr_extractor.process_single_image(saved_path)
        text = result.get("text", "")
        desc = result.get("description", "")
        metadata = result.get("metadata", {})
        extracted_path, extracted_result = normalize_and_save_extracted(fname, ext, text, desc, metadata)

    else:
        raise ValueError("Unsupported file type: " + ext)
    
    cleansed_path, cleansed = cleanse_and_save(extracted_result)
    
    # Use environment variable for API key only
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        analyzed_path, analyzed = analyze_and_save(cleansed, api_key)
    else:
        analyzed = {
            "file_name": cleansed.get("file_name"),
            "file_type": cleansed.get("file_type"),
            "file_description": "[LLM analysis requires GOOGLE_API_KEY environment variable]",
            "key_findings": "[LLM analysis requires GOOGLE_API_KEY environment variable]"
        }
        analyzed_path = None


    return {
        "extracted_path": extracted_path,
        "cleansed_path": cleansed_path,
        "analyzed_path": analyzed_path,
        "final_result": analyzed
    }


# ---------------- MAIN UI ---------------- #
st.markdown("### 📤 File Upload")
uploaded = st.file_uploader(
    "**Upload your file (PPTX, PDF, XLSX, PNG, JPG, JPEG)**",
    type=["pptx","pdf","xlsx","xls","png","jpg","jpeg"],
    help="Supported formats: Presentations (PPTX), Documents (PDF), Spreadsheets (XLSX/XLS), Images (PNG/JPG/JPEG)"
)


if uploaded:
    # File info card
    file_size = len(uploaded.getvalue()) / 1024  # KB
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded.name)
    with col2:
        st.metric("File Type", Path(uploaded.name).suffix.upper())
    with col3:
        st.metric("File Size", f"{file_size:.1f} KB")
    
    st.success(f"📁 **File `{uploaded.name}` uploaded successfully!** Ready for processing.")
    
    # Check if API key is available
    api_key_available = bool(os.getenv("GOOGLE_API_KEY", ""))
    if api_key_available:
        st.info("🔑 **LLM Analysis**: Enabled (using environment variable)")
    else:
        st.warning("⚠️ **LLM Analysis**: Disabled - Set GOOGLE_API_KEY environment variable to enable AI-powered analysis")
    
    # Centered run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Run Analysis Pipeline", use_container_width=True)
    
    if run_button:
        with st.spinner("⏳ Processing file through extraction → cleansing → analysis ..."):
            try:
                out = run_pipeline_for_file(save_uploaded_file(uploaded))
                final = out["final_result"]


                st.success("✅ Processing complete!")
                
                st.markdown("### 📊 Analysis Summary")
                
                # Simple table with 4 columns
                analysis_data = {
                    "File Name": [final["file_name"]],
                    "File Type": [final["file_type"]],
                    "File Description": [final["file_description"]],
                    "Key Findings": [final["key_findings"]]
                }
                
                df = pd.DataFrame(analysis_data)
                st.table(df)


                # Detailed view in expander
                with st.expander("🔍 View Detailed Intermediate Files (JSON Outputs)", expanded=False):
                    tab1, tab2, tab3 = st.tabs(["📄 Extracted", "✨ Cleansed", "🤖 Analyzed"])
                    
                    with tab1:
                        st.write("**Extracted File Content:**")
                        if out["extracted_path"] and os.path.exists(out["extracted_path"]):
                            st.code(open(out["extracted_path"],"r",encoding="utf-8").read(), language="json")
                        else:
                            st.info("No extracted file content available")
                    
                    with tab2:
                        st.write("**Cleansed File Content:**")
                        if out["cleansed_path"] and os.path.exists(out["cleansed_path"]):
                            st.code(open(out["cleansed_path"],"r",encoding="utf-8").read(), language="json")
                        else:
                            st.info("No cleansed file content available")
                    
                    with tab3:
                        st.write("**Analyzed (LLM) Output:**")
                        if out["analyzed_path"] and os.path.exists(out["analyzed_path"]):
                            st.code(open(out["analyzed_path"],"r",encoding="utf-8").read(), language="json")
                        else:
                            st.info("LLM analysis was not enabled or no analyzed content available")


            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style='text-align: center; padding: 3rem; color: #666;'>
        <h3 style='color: #1f4e79; margin-bottom: 1rem;'>🎯 How to Use</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
            1. <strong>Upload</strong> your document or image file<br>
            2. <strong>Run</strong> the analysis pipeline<br>
            3. <strong>View</strong> extracted, cleansed, and analyzed results
        </p>
        <p style='margin-top: 2rem; color: #999;'>
            Supported formats: PPTX, PDF, XLSX, XLS, PNG, JPG, JPEG
        </p>
        <div style='margin-top: 2rem; padding: 1rem; background: rgba(31, 119, 180, 0.1); border-radius: 12px; display: inline-block;'>
            <p style='margin: 0; color: #1f4e79; font-weight: 600;'>
                💡 Tip: Set GOOGLE_API_KEY environment variable to enable AI-powered analysis
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
