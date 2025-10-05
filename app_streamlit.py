import os
import sys
import json
from pathlib import Path
import pytesseract
import streamlit as st
import pandas as pd

# OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
)

# custom CSS
st.markdown("""
    <style>
        /* Main page background */
        .stApp {
            background-color: #f7f9fb;
        }
        /* Title styling */
        h1 {
            text-align: center;
            color: #1f4e79;
            font-family: 'Segoe UI', sans-serif;
            padding-bottom: 0.3em;
        }
        /* Upload box */
        .uploadedFile {
            border-radius: 8px !important;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 16px;
            font-weight: 500;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #155a8a;
        }
        /* Info and success messages */
        .stAlert {
            border-radius: 10px !important;
        }
        /* Tables */
        .dataframe {
            border: 1px solid #ddd !important;
            border-radius: 10px;
            overflow: hidden;
        }
        /* Expander text box */
        .stCodeBlock {
            background-color: #fafafa !important;
            border-radius: 8px;
            border: 1px solid #e6e6e6;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<h1>📄 File Cleansing & Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload your document or image and process it through extraction → cleansing → analysis.</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/584/584796.png", width=80)
st.sidebar.markdown("### ⚙️ Configuration")

env_key = os.getenv("GOOGLE_API_KEY", "")
api_key_input = st.sidebar.text_input("🔑 Google API Key (optional)", value=env_key, type="password")
use_api = st.sidebar.checkbox("Enable LLM Analysis", value=bool(api_key_input or env_key))

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Upload PDF, PPTX, XLSX, or image files to start the pipeline.")

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
    cleansed_text = (cleansed_dict.get("raw_text","") or "") + "\n" + (cleansed_dict.get("file_description","") or "")
    raw_out = llm_analysis.analyze_cleansed_text(cleansed_text)
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
        text, w, h = ocr_extractor.ocr_image(saved_path)
        desc = ocr_extractor.get_scene_description(saved_path)
        metadata = {"resolution": f"{w}x{h}"}
        extracted_path, extracted_result = normalize_and_save_extracted(fname, ext, text, desc, metadata)
    else:
        raise ValueError("Unsupported file type: " + ext)
    
    cleansed_path, cleansed = cleanse_and_save(extracted_result)
    
    if api_key:
        analyzed_path, analyzed = analyze_and_save(cleansed, api_key)
    else:
        analyzed = {
            "file_name": cleansed.get("file_name"),
            "file_type": cleansed.get("file_type"),
            "file_description": "[LLM disabled]",
            "key_findings": "[LLM disabled]"
        }
        analyzed_path = None

    return {
        "extracted_path": extracted_path,
        "cleansed_path": cleansed_path,
        "analyzed_path": analyzed_path,
        "final_result": analyzed
    }

# ---------------- MAIN UI ---------------- #
uploaded = st.file_uploader(
    "📤 Upload a file (PPTX, PDF, XLSX, PNG, JPG)",
    type=["pptx","pdf","xlsx","xls","png","jpg","jpeg"]
)

if uploaded:
    st.info(f"📁 File `{uploaded.name}` uploaded successfully!")
    saved_path = save_uploaded_file(uploaded)
    run_button = st.button("🚀 Run Analysis Pipeline")
    if run_button:
        with st.spinner("⏳ Processing file through extraction → cleansing → analysis ..."):
            try:
                api_key = api_key_input or os.getenv("GOOGLE_API_KEY", "")
                out = run_pipeline_for_file(saved_path, api_key if use_api else None)
                final = out["final_result"]

                st.success("✅ Processing complete!")
                st.markdown("### 📊 Analysis Summary")
                df = pd.DataFrame([{
                    "File Name": final["file_name"],
                    "File Type": final["file_type"],
                    "Description": final["file_description"],
                    "Key Findings": final["key_findings"]
                }])
                st.table(df)

                with st.expander("🔍 View Intermediate Files (JSON Outputs)"):
                    st.write("**Extracted File:**")
                    st.code(open(out["extracted_path"],"r",encoding="utf-8").read() if out["extracted_path"] else "No extracted file")

                    st.write("**Cleansed File:**")
                    st.code(open(out["cleansed_path"],"r",encoding="utf-8").read())

                    if out["analyzed_path"]:
                        st.write("**Analyzed (LLM) Output:**")
                        st.code(open(out["analyzed_path"],"r",encoding="utf-8").read())

            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
else:
    st.markdown("<p style='text-align:center; color:#999;'>Upload a file above to begin analysis.</p>", unsafe_allow_html=True)
