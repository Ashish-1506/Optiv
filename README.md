Optiv Document Analysis Dashboard
Project Overview

This project provides an automated pipeline to analyze documents of various formats (PDF, PPTX, XLSX, images) and extract meaningful content and insights. The pipeline includes OCR for scanned documents/images, PII cleansing, and AI-based security analysis. Users can interact with the project through a Streamlit dashboard to upload files and view the results in a structured table.

Key Features:

Extract text from PDFs, PPTX, XLSX, and images.

Automatic OCR for scanned PDFs and images.

PII detection and cleansing.

AI-based file description and key findings generation.

Streamlit dashboard with file upload and results table.

File Structure
Optiv/
│
├── app_streamlit.py          # Streamlit dashboard UI
├── ocr_extractor.py          # OCR functions (Parvesh)
├── document_extractor.py     # File extraction (Aditi)
├── cleanse_text.py           # PII cleansing (Sanyukta)
├── ai_analysis.py            # AI-based analysis (Ashwin)
├── requirements.txt          # All Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   ├── raw/                  # Place input files here (PDF, PPTX, XLSX, images)
│   ├── cleansed/             # Files after PII cleansing
│   └── outputs/              # Final JSON/Text analysis outputs

Dependencies

Python Packages:

pip install streamlit pandas numpy opencv-python pytesseract pillow pdfplumber pdf2image python-pptx transformers torch spacy openpyxl google-generativeai
python -m spacy download en_core_web_sm


System Dependencies:

Tesseract OCR: Download here

Make sure Tesseract executable is installed (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe) and added to your system PATH.

Poppler (Windows only, for PDF conversion to images):

Download here

Add the bin folder to your PATH (e.g., C:/poppler/bin)

Setup Instructions

Clone the repository or copy project files to your local machine.

Create a Python virtual environment (optional but recommended):

python -m venv .venv


Activate the virtual environment:

Windows PowerShell:

& .\.venv\Scripts\Activate.ps1


Windows CMD:

.venv\Scripts\activate.bat


Linux/Mac:

source .venv/bin/activate


Install Python dependencies:

pip install -r requirements.txt


Ensure Tesseract is accessible:

Verify:

tesseract --version


If not recognized, add the Tesseract folder to PATH or hardcode path in ocr_extractor.py:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


Ensure spaCy model is downloaded:

python -m spacy download en_core_web_sm

Running the Project
1. Streamlit Dashboard
streamlit run app_streamlit.py


Open the displayed localhost URL in your browser.

Upload files (PDF, PPTX, XLSX, PNG, JPG).

View the results table with the following columns:

File Name

File Type

File Description

Key Findings

2. CLI Testing (Optional)

Extract text and run analysis manually:

python document_extractor.py
python cleanse_text.py
python ai_analysis.py


Outputs are saved in data/outputs as JSON or text files.

Output Format

Each processed file produces a JSON/text file with the following structure:

{
  "file_name": "PS_01_EV1.png",
  "file_type": ".png",
  "file_description": "A cleaned document containing text and image description related to access control.",
  "key_findings": "- Describes use of smart card access systems.\n- No sensitive data detected.\n- Indicates secure facility entry control."
}

Notes

OCR may take a few seconds per page/image depending on file size.

Large PDFs or PPTX files with many images may increase processing time.

Ensure your Google Gemini API key is set correctly in ai_analysis.py for AI analysis.
