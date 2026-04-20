import json
import re
import requests


OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

DOCUMENT_SCHEMAS = {
    "visa": [
        "surname_and_given_name", "passport_no", "visa_type",
        "no_of_entries","date_of_issue","date_of_expiry(dd/mm/yyyy)",
        "special_endorsment"
    ],
    "pan": [
        "name", "father_name", "date_of_birth", "pan_number"
    ],
    "aadhaar": [
        "name", "date_of_birth", "gender", "address",
        "pincode", "aadhaar_number"
    ],
    "passport": [
        "surname", "given_name", "nationality", "date_of_birth",
        "place_of_birth", "date_of_issue", "date_of_expiry",
        "passport_number", "place_of_issue", "gender", "file_number"
    ]
}

VALUE_PATTERNS = {
    "date"           : r"\d{2}[/\-]\d{2}[/\-]\d{4}",
    "pan_number"     : r"[A-Z]{5}\d{4}[A-Z]",
    "passport_number": r"[A-Z]\d{8}",
    "visa_number"    : r"[A-Z]{2}\d{7}",
    "aadhaar_number" : r"\d{4}\s?\d{4}\s?\d{4}"
}


def build_prompt(ocr_output: dict, metadata: dict) -> str:
    english_text = ocr_output.get("english", {}).get("text", "")
    hindi_text   = ocr_output.get("hindi",   {}).get("text", "")

    schemas_json = json.dumps(DOCUMENT_SCHEMAS, indent=2)

    return f"""You are a document validation agent specialized in Indian identity documents.
You will receive OCR output from a scanned document in English and Hindi, along with metadata.
Your task is to:
    1. Identify the document type (visa, pan, aadhaar, passport).
    2. Extract all fields defined in the schema for that document type.
    3. Cross-validate the English and Hindi versions — flag any inconsistency.
    4. Return a single JSON object. No explanation, no markdown, only raw JSON.

Document schemas:
{schemas_json}

English OCR text:
{english_text}

Hindi OCR text:
{hindi_text}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Return this exact JSON structure:
{{
    "document_type": "<detected type>",
    "fields": {{
        "<field_name>": "<extracted value or null if not found>"
    }},
    "inconsistencies": [
        {{
            "field"      : "<field name>",
            "english"    : "<value in English>",
            "hindi"      : "<value in Hindi>",
            "description": "<what is inconsistent>"
        }}
    ],
    "confidence": "<high | medium | low>",
    "notes": "<any relevant observation about the document>"
}}"""


def call_gemma(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)


def validate_fields(fields: dict) -> list[dict]:
    issues = []
    for field, value in fields.items():
        if not value:
            continue
        for pattern_name, pattern in VALUE_PATTERNS.items():
            if pattern_name.replace("_number", "") in field.lower():
                if not re.fullmatch(pattern, str(value).upper().strip()):
                    issues.append({
                        "field"  : field,
                        "value"  : value,
                        "pattern": pattern_name,
                        "issue"  : "value does not match expected format"
                    })
    return issues


def run_agent(ocr_output: dict, metadata: dict, debug: bool = False) -> dict:
    prompt = build_prompt(ocr_output, metadata)

    if debug:
        print("\n── PROMPT ───────────────────────────────────────────────")
        print(prompt)

    print("Calling Gemma 3...")
    raw      = call_gemma(prompt)

    if debug:
        print("\n── RAW RESPONSE ─────────────────────────────────────────")
        print(raw)

    result           = parse_response(raw)
    format_issues    = validate_fields(result.get("fields", {}))

    if format_issues:
        result["format_issues"] = format_issues

    return result