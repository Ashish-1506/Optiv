import os
import json
import google.generativeai as genai

# -------------------------------
# Configuration
# -------------------------------
MY_API_KEY = "Your API key"

MODEL_NAME = 'models/gemini-2.5-flash-preview-05-20'  # gemini model used
INPUT_DIR = "../data/cleansed"  # folder containing cleansed .txt files
OUTPUT_DIR = "../data/outputs"  # folder to save analyzed results
os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_api_with_key(api_key: str) -> bool:
    """Configures the Google AI client using the provided API key."""
    if not api_key or "PASTE" in api_key:
        print("ERROR: API Key is missing. Please paste your key into the MY_API_KEY variable at the top of the script.")
        return False
    try:
        genai.configure(api_key=api_key)
        print("Google API Key configured successfully!")
        return True
    except Exception as e:
        print(f"ERROR: Could not configure the API Key. It might be invalid. Details: {e}")
        return False


def create_security_analysis_prompt(cleansed_text: str) -> str:
    """Creates the specific prompt for the Gemini model to get the desired output format."""
    prompt = f"""
    You are a senior cybersecurity analyst at Optiv. Your task is to analyze a cleansed document and provide a summary for a presentation, exactly matching the company's required style.

    Analyze the following cleansed document text. Based ONLY on the provided text, generate two specific outputs in the exact format shown below:

    **1. File Description:**
    Provide a single, literal, and objective sentence describing what the document *is*. AVOID summarizing its contents or interpreting its meaning. For example, instead of "This summarizes meetings," say "A text document containing meeting notes and PII."

    **2. Key Findings:**
    Provide 2-4 concise bullet points. Each bullet point must be a distinct security observation, risk, or implication that a consultant would find valuable.

    ---
    CLEANSED DOCUMENT TEXT:
    {cleansed_text}
    ---

    **REQUIRED OUTPUT FORMAT (Use '###' as a separator):**
    File Description: [Your single, literal sentence description here]
    ###
    Key Findings:
    - [Finding 1]
    - [Finding 2]
    - [Finding 3]
    """
    return prompt


def analyze_cleansed_text(cleansed_text: str) -> str:
    """Sends the cleansed text to the Google Gemini model and returns the analysis."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt_to_send = create_security_analysis_prompt(cleansed_text)
        response = model.generate_content(prompt_to_send)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred during AI analysis: {e}"


def parse_analysis_output(analysis_text: str) -> dict:
    """Parses the raw AI output into a structured dictionary for the final table."""
    if '###' not in analysis_text:
        return {'description': 'Parsing Error: Invalid format received from AI.', 'findings': analysis_text}
    try:
        parts = analysis_text.split('###')
        description = parts[0].replace('File Description:', '').strip()
        findings = parts[1].replace('Key Findings:', '').strip()
        return {'description': description, 'findings': findings}
    except IndexError:
        return {'description': 'Parsing Error: Could not split the AI response.', 'findings': analysis_text}


def main():
    """Main function to run the file analysis pipeline for all cleansed files."""
    print("--- Starting Optiv File Analysis Engine ---")

    if not configure_api_with_key(MY_API_KEY):
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_cleansed.txt")]
    if not files:
        print("No cleansed files found in the input directory.")
        return

    for fname in files:
        input_path = os.path.join(INPUT_DIR, fname)
        print(f"\n[INFO] Processing {fname}...")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())  # cleansed JSON structure
        except Exception as e:
            print(f"ERROR reading file {fname}: {e}")
            continue

        # Prepare content for analysis
        file_name = data.get("file_name", fname)
        file_type = data.get("file_type", "unknown")
        cleansed_text = data.get("raw_text", "") + "\n" + data.get("file_description", "")

        # Send to model
        print("[INFO] Sending to Gemini for analysis...")
        raw_output = analyze_cleansed_text(cleansed_text)
        parsed_output = parse_analysis_output(raw_output)

        # Build final structured result
        final_result = {
            "file_name": file_name,
            "file_type": file_type,
            "file_description": parsed_output["description"],
            "key_findings": parsed_output["findings"]
        }

        # Save to output folder
        output_path = os.path.join(OUTPUT_DIR, fname.replace("_cleansed.txt", "_analyzed.txt"))
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(final_result, indent=2, ensure_ascii=False))

        print(f"✅ Analysis complete. Saved: {output_path}")

    print("\n--- All files analyzed successfully! ---")


if __name__ == "__main__":
    main()
