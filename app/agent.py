import json
import re
import requests
from models import PipelineOutput, MRZResult, Config


def _load_fields(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolved_from_mrz(mrz: MRZResult) -> dict:
    return {
        "surname"    : mrz.surname,
        "given_names": mrz.given_names,
        "country"    : mrz.country,
        "birth_date" : mrz.birth_date,
        "expiry_date": mrz.expiry_date,
        "number"     : mrz.number,
        "sex"        : mrz.sex
    }


def _build_prompt(output: PipelineOutput, all_fields: list[str]) -> str:
    resolved = _resolved_from_mrz(output.mrz) if output.mrz else {}
    missing  = [f for f in all_fields if f not in resolved]

    resolved_str = json.dumps(resolved, ensure_ascii=False, indent=2) if resolved else "none"
    missing_str  = ", ".join(missing) if missing else "none"

    return f"""You are a document validation agent for official identity documents.

Already resolved via MRZ (do not re-extract):
{resolved_str}

Extract only these remaining fields: {missing_str}

Instructions:
- Use only the Latin/English text provided
- If a field is not found, set it to null
- Detect the document type from the content
- Flag any suspicious inconsistencies
- Return only raw JSON, no explanation, no markdown

Text:
{output.english_text}

Return this exact structure:
{{
    "document_type": "<detected type>",
    "fields": {{
        "<field_name>": "<value or null>"
    }},
    "inconsistencies": [
        {{
            "field": "<field>",
            "description": "<what is inconsistent>"
        }}
    ],
    "confidence": "<high | medium | low>",
    "verdict": "<genuine | suspicious>",
    "notes": "<optional observation>"
}}"""


def _call_gemma(prompt: str, config: Config) -> str:
    response = requests.post(config.ollama_url, json={
        "model" : config.ollama_model,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def _parse_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)


def analyze(output: PipelineOutput, config: Config) -> dict:
    fields_data = _load_fields(config.document_fields_path)
    all_fields  = fields_data.get("default", [])

    prompt = _build_prompt(output, all_fields)
    raw    = _call_gemma(prompt, config)
    result = _parse_response(raw)

    if output.mrz:
        result["mrz"] = {
            "valid"      : output.mrz.valid,
            "surname"    : output.mrz.surname,
            "given_names": output.mrz.given_names,
            "country"    : output.mrz.country,
            "birth_date" : output.mrz.birth_date,
            "expiry_date": output.mrz.expiry_date,
            "number"     : output.mrz.number,
            "sex"        : output.mrz.sex
        }

    result["source"]         = output.source
    result["confidence_avg"] = output.confidence_avg
    return result