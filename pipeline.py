import numpy as np
from app.backends.agent import analyze
from models import Config, PipelineOutput
from app.language      import filter_latin
from app.mrz           import detect

from app.backends.base import LLMBackend


def _avg_confidence(lines: list) -> float:
    if not lines:
        return 0.0
    return sum(l.confidence for l in lines) / len(lines)


def run(image: np.ndarray,
        ocr_engine,
        config: Config,
        backend: LLMBackend,
        debug: bool = False) -> tuple[PipelineOutput | None, dict]:

    lines = ocr_engine.extract(image)

    if not lines:
        return None, {"error": "no text extracted", "source": None}

    confidence_avg = _avg_confidence(lines)

    if debug:
        print(f"Lines extracted  : {len(lines)}")
        print(f"Confidence avg   : {confidence_avg:.2f}")

    if confidence_avg < config.confidence_threshold:
        return None, {
            "error"         : "low confidence — document quality insufficient",
            "confidence_avg": round(confidence_avg, 4),
            "source"        : None
        }

    english_lines = filter_latin(lines)

    if debug:
        print(f"English lines    : {len(english_lines)}")

    english_text = "\n".join(l.text for l in english_lines)
    english_lines = filter_latin(lines)

    mrz = detect(english_lines)

    print(f"detect() returned: {mrz}")

    mrz_verified   = mrz if (mrz and mrz.valid)  else None
    mrz_unverified = mrz if (mrz and not mrz.valid) else None

    if mrz_verified:
        source = "mrz+gemma"
    elif mrz_unverified:
        source = "mrz_partial+gemma"
    else:
        source = "gemma"

    if debug:
        print(f"MRZ detected     : {mrz is not None}")
        print(f"MRZ valid        : {mrz.valid if mrz else False}")
        print(f"Source           : {source}")

    output = PipelineOutput(
        mrz_verified   = mrz_verified,
        mrz_unverified = mrz_unverified,
        english_lines  = english_lines,
        english_text   = english_text,
        source         = source,
        confidence_avg = round(confidence_avg, 4),
        raw_lines      = lines
    )

    return output, analyze(output, config, backend)