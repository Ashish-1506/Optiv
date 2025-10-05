import os
import re
import json
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Directories
INPUT_DIR = "../data/extracted"
OUTPUT_DIR = "../data/cleansed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def redact_pii_with_spacy(text):
    """
    Redacts common PII entities using spaCy's Named Entity Recognition.
    """
    doc = nlp(text)
    redacted_text = text
    pii_labels = ["PERSON", "GPE", "ORG"]

    # Sort entities in reverse order to avoid messing up indexes
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in pii_labels:
            start, end = ent.start_char, ent.end_char
            redacted_text = redacted_text[:start] + f"[{ent.label_} REDACTED]" + redacted_text[end:]
    return redacted_text


def redact_custom_patterns(text):
    """
    Redacts regex-based patterns like SSN, phone, email, or client names/logos.
    """
    patterns = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "CLIENT_NAME": r"\bOptiv Security Inc\.?\b|\bVIT\s*CAMPUS\s*CONNECT\b",
    }
    cleaned = text
    for tag, pattern in patterns.items():
        cleaned = re.sub(pattern, f"[{tag} REDACTED]", cleaned, flags=re.IGNORECASE)
    return cleaned


def cleanse_text(text):
    """Apply spaCy + regex cleaning pipeline."""
    text_after_spacy = redact_pii_with_spacy(text)
    return redact_custom_patterns(text_after_spacy)


def cleanse_extracted_files(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Reads extracted .txt files (JSON-style), cleanses sensitive fields,
    and writes sanitized versions to data/cleansed/.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith("_extracted.txt")]
    for fname in sorted(files):
        print(f"[INFO] Cleansing {fname}...")
        fpath = os.path.join(input_dir, fname)

        # Read input file (it’s formatted JSON in text)
        with open(fpath, "r", encoding="utf-8") as infile:
            data = json.loads(infile.read())

        # Clean the relevant fields
        data["raw_text"] = cleanse_text(data.get("raw_text", ""))
        data["file_description"] = cleanse_text(data.get("file_description", ""))

        # Save the cleansed text as a .txt file
        out_path = os.path.join(output_dir, fname.replace("_extracted.txt", "_cleansed.txt"))
        with open(out_path, "w", encoding="utf-8") as outfile:
            outfile.write(json.dumps(data, indent=2, ensure_ascii=False))

        print(f"✅ Saved cleansed file: {out_path}")

    print("\nAll extracted files cleansed successfully!\n")


# Example standalone test
if __name__ == "__main__":
    cleanse_extracted_files()
