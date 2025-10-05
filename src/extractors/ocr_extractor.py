import os
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "Path of tesseract.exe"

import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# ---------------------------
# Configuration
# ---------------------------
POPPLER_PATH = "Give Poppler path" #like r'C:\poppler\Library\bin'
INPUT_DIR = '../data/raw'
OUTPUT_DIR = '../data/extracted'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load BLIP Captioning Model
# ---------------------------
print("Loading BLIP captioning model (first run may take a few minutes)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# ---------------------------
# Helper Functions
# ---------------------------
def preprocess(img):
    """Convert to grayscale and threshold for better OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_image(file_path):
    """Perform OCR on a single image."""
    img = cv2.imread(file_path)
    processed = preprocess(img)
    text = pytesseract.image_to_string(processed)
    return text, img.shape[1], img.shape[0]  # width, height


def get_scene_description(file_path):
    """Generate natural-language caption for an image."""
    image = Image.open(file_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def ocr_pdf(file_path):
    """Convert PDF pages to images and perform OCR + captioning."""
    pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
    all_text, all_desc = [], []
    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        processed = preprocess(img)
        text = pytesseract.image_to_string(processed)
        all_text.append(text)
        # temp save for captioning
        temp_path = os.path.join(OUTPUT_DIR, f"temp_page{i}.jpg")
        cv2.imwrite(temp_path, img)
        desc = get_scene_description(temp_path)
        all_desc.append(desc)
        os.remove(temp_path)
    return "\n\n".join(all_text), " | ".join(all_desc), len(pages)


# ---------------------------
# Main Function
# ---------------------------
def extract_text_from_image_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """Process all .png/.jpg/.jpeg/.pdf files and save text file per output."""
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]

    for fname in sorted(files):
        ext = os.path.splitext(fname)[1].lower()
        fpath = os.path.join(input_dir, fname)
        print(f"[INFO] Processing {fname}...")

        if ext in ['.png', '.jpg', '.jpeg']:
            text, width, height = ocr_image(fpath)
            description = get_scene_description(fpath)
            metadata = {"resolution": f"{width}x{height}"}

        elif ext == '.pdf':
            text, description, numpages = ocr_pdf(fpath)
            metadata = {"pages": numpages}

        else:
            print(f"[WARN] Skipping unsupported file type: {fname}")
            continue

        # Create dictionary for output
        result = {
            "file_name": fname,
            "file_type": ext,
            "raw_text": text.strip(),
            "file_description": description.strip(),
            "metadata": metadata
        }

        # Convert dictionary to formatted JSON string (for readability)
        output_text = json.dumps(result, indent=2, ensure_ascii=False)

        # Save as .txt file
        txt_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "_extracted.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"✅ Saved: {txt_path}")

    print("\nAll files processed successfully!\n")


# ---------------------------
# Run Directly
# ---------------------------
if __name__ == "__main__":
    extract_text_from_image_folder()
