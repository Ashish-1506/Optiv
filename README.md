# Document Intelligence Pipeline

This project is a comprehensive pipeline for extracting, cleansing, and analyzing content from various document types, including presentations, spreadsheets, PDFs, and images. It features a user-friendly web interface built with Streamlit for easy interaction and a series of Jupyter notebooks for detailed, step-by-step demonstrations of the underlying processes.

## 🌟 Features

-   **Multi-Format Support**: Ingests and processes `.pptx`, `.pdf`, `.xlsx`, `.png`, and `.jpg` files.
-   **Advanced Text Extraction**: Extracts text from standard documents (PPTX, XLSX) and digital PDFs.
-   **Optical Character Recognition (OCR)**: Automatically detects and extracts text from images and scanned PDFs using Tesseract.
-   **AI-Powered Image Analysis**: Generates descriptive captions for images using the BLIP Vision-Language model.
-   **Text Cleansing Engine**: Normalizes and cleans extracted text to improve quality and consistency for analysis.
-   **LLM-Powered Analysis**: Leverages Google's Gemini API to perform deep analysis, generating concise descriptions and key findings from the cleansed content.
-   **Interactive Web UI**: A Streamlit application allows users to upload files and run the entire pipeline with the click of a button.
-   **Modular Notebooks**: Includes a set of Jupyter notebooks that break down each stage of the pipeline, from extraction to analysis.

## 📂 Project Structure

```
.
├── app_streamlit.py         # Main Streamlit web application
├── requirements.txt         # Python package dependencies
├── README.md                # Project documentation
├── data/                    # Data files for each stage
│   ├── raw/                 # Original uploaded files
│   ├── extracted/           # Raw text extracted from files
│   ├── cleansed/            # Cleansed and normalized text
│   └── outputs/             # Final analysis results from the LLM
├── notebooks/               # Jupyter notebooks for pipeline demonstration
│   ├── 00_setup.ipynb
│   ├── 01_extraction.ipynb
│   ├── 02_ocr.ipynb
│   ├── 03_cleansing.ipynb
│   ├── 04_analysis.ipynb
│   └── 05_pipeline_demo.ipynb
└── src/                     # Source code for the pipeline modules
    ├── extractors/          # Modules for file and text extraction
    ├── cleansing/           # Text cleansing engine
    ├── analysis/            # LLM analysis module
    └── integrator/          # Pipeline integration logic
```

## 🛠️ Setup & Installation

### Prerequisites

1.  **Python 3.9+**: Make sure you have a modern version of Python installed.
2.  **Tesseract-OCR**: This is required for the OCR functionality.
    -   **Windows**: Download and install from the [official Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki). During installation, make sure to select the option to add Tesseract to your system's PATH. The code currently expects the executable at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3.  **Poppler**: This is required for converting PDF pages to images for OCR.
    -   **Windows**: Download the latest binary from [this page](https://github.com/oschwartz10612/poppler-windows/releases/). Unzip the folder (e.g., to `C:\poppler`) and add the `bin` subdirectory to your system's PATH. The code currently expects it at `C:\poppler\Library\bin`.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

### 1. Running the Streamlit Web App

The easiest way to use the pipeline is through the interactive web interface.

1.  **Set Your API Key (Optional):**
    For the final analysis step, you need a Google Gemini API key. You can set this as an environment variable:
    ```bash
    # For Windows (PowerShell)
    $env:GOOGLE_API_KEY="your_api_key_here"

    # For macOS/Linux
    export GOOGLE_API_KEY="your_api_key_here"
    ```
    Alternatively, you can enter the key directly in the Streamlit app's sidebar.

2.  **Launch the application:**
    Make sure you are in the project's root directory, then run:
    ```bash
    streamlit run app_streamlit.py
    ```
    Your web browser will open with the application running locally at `http://localhost:8501`.

3.  **Use the App:**
    -   Upload a supported file.
    -   Enable LLM analysis if you have an API key.
    -   Click "Run analysis pipeline" to see the results.

### 2. Using the Jupyter Notebooks

For a more detailed, step-by-step guide through the pipeline's components, you can run the Jupyter notebooks located in the `notebooks/` directory. Start with `00_setup.ipynb` and proceed in order.

1.  **Start Jupyter Lab or Jupyter Notebook:**
    ```bash
    jupyter lab
    ```

2.  **Navigate to the `notebooks/` directory** and open the notebooks to explore each part of the process.

