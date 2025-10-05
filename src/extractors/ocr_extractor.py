import os
import sys
import json
import cv2
import pytesseract
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# Configuration
# ---------------------------
# IMPORTANT: Ensure the path to your Tesseract installation is correct.
pytesseract.pytesseract.tesseract_cmd = "Path of tesseract.exe" #like r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# IMPORTANT: Ensure the path to your Poppler installation is correct for PDF processing.

POPPLER_PATH = "Give Poppler path" #like r'C:\poppler\Library\bin'
INPUT_DIR = '../data/raw'
OUTPUT_DIR = '../data/extracted'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load UPGRADED Offline Image Captioning Model
# ---------------------------
# This model runs 100% locally to generate scene descriptions.
print("Loading offline BLIP captioning model (first run may download files)...")
try:
    # UPGRADE: Using the larger, more accurate model for better descriptions.
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("✅ BLIP-large model loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load the BLIP model. This is a one-time setup step.")
    print(f"   Please ensure you have a stable internet connection and try running the script again.")
    print(f"   Error details: {e}")
    caption_model = None

# ---------------------------
# Advanced Image Preprocessing for OCR
# ---------------------------
def advanced_ocr_pipeline(image: np.ndarray) -> str:
    """
    Implements a sophisticated, multi-stage pipeline to find and extract text from challenging images.
    """
    # 1. Standard Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # 2. Find Contours (potential text regions)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_texts = []

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 3. If a 4-point contour is found (likely a rectangle), process it
        if len(approx) == 4:
            if cv2.contourArea(contour) < 500:
                continue

            # 4. Apply Perspective Warp
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            if maxWidth == 0 or maxHeight == 0: continue

            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            
            # 5. OCR the warped, flattened image region
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(warped_thresh, config='--psm 6').strip()
            if text:
                extracted_texts.append(text)
    
    # --- SMARTER FALLBACK LOGIC ---
    # 6. Only if the advanced method finds NO text, run OCR on the whole image.
    if not extracted_texts:
        full_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        full_image_thresh = cv2.threshold(full_image_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        full_text = pytesseract.image_to_string(full_image_thresh, config='--psm 6').strip()
        if full_text:
            extracted_texts.append(full_text)
        
    return "\n".join(list(set(extracted_texts))) # Use set to remove duplicate text blocks

# ---------------------------
# Core Extraction Functions
# ---------------------------
def get_image_caption(image_pil: Image.Image) -> str:
    """Generates a detailed caption for an image using the local BLIP model."""
    if caption_model is None:
        return "Captioning model not available."
    try:
        inputs = caption_processor(images=image_pil, return_tensors="pt")
        out = caption_model.generate(**inputs, max_length=75, num_beams=5, early_stopping=True)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Caption generation failed: {e}"

def process_single_image(file_path: str) -> dict:
    """Processes a single image file for both OCR and captioning."""
    try:
        img_cv = cv2.imread(file_path)
        img_pil = Image.open(file_path).convert('RGB')
        
        # 1. Generate rich description with the upgraded model
        description = get_image_caption(img_pil)
        
        # 2. Perform advanced OCR using the new pipeline
        text = advanced_ocr_pipeline(img_cv)
        
        return {
            "description": description,
            "text": text,
            "metadata": {"resolution": f"{img_cv.shape[1]}x{img_cv.shape[0]}"}
        }
    except Exception as e:
        print(f"    [ERROR] Processing failed for {os.path.basename(file_path)}: {e}")
        return {"description": "", "text": "", "metadata": {"error": str(e)}}

def process_pdf(file_path: str) -> dict:
    """Processes each page of a PDF for both OCR and captioning."""
    try:
        pages = convert_from_path(file_path, dpi=200, poppler_path=POPPLER_PATH)
        all_text, all_desc = [], []
        for i, page_img in enumerate(pages):
            print(f"    -> Processing page {i+1}/{len(pages)}...")
            description = get_image_caption(page_img)
            all_desc.append(f"Page {i+1}: {description}")

            img_cv = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB_BGR)
            # Use a simpler preprocessing for PDF pages as they are usually flat
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(thresh, config='--psm 3')
            all_text.append(text)
        
        return {
            "description": "\n".join(all_desc),
            "text": "\n\n--- Page Break ---\n\n".join(all_text),
            "metadata": {"pages": len(pages)}
        }
    except Exception as e:
        print(f"    [ERROR] PDF processing failed for {os.path.basename(file_path)}: {e}")
        return {"description": "", "text": "", "metadata": {"error": str(e)}}

# ---------------------------
# Main Processing Loop
# ---------------------------
def extract_data_from_files(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """Processes all image and PDF files in a directory."""
    # --- PRE-FLIGHT CHECK ---
    if caption_model is None:
        print("\nHalting execution because the offline captioning model is not loaded.")
        print("Please ensure you have an internet connection for the first run to download the model.")
        sys.exit(1) # Exit the script

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

    for fname in sorted(files):
        fpath = os.path.join(input_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        print(f"[INFO] Processing {fname}...")

        result_data = {}
        if ext in ['.png', '.jpg', '.jpeg']:
            result_data = process_single_image(fpath)
        elif ext == '.pdf':
            result_data = process_pdf(fpath)
        
        output_json = {
            "file_name": fname,
            "file_type": ext,
            "raw_text": result_data.get("text", "").strip(),
            "file_description": result_data.get("description", "").strip(),
            "metadata": result_data.get("metadata", {})
        }

        json_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "_extracted.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {json_path}")

    print("\nAll files processed successfully!\n")

# ---------------------------
# Run Directly
# ---------------------------
if __name__ == "__main__":
    extract_data_from_files()
