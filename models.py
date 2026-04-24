from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Config:
    confidence_threshold: float = 0.60
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "gemma3:4b"
    document_fields_path: str = "data/document_fields.json"
    max_image_side: int = 1_400


@dataclass
class TextLine:
    text: str
    confidence: float
    bbox: np.ndarray = field(repr=False, compare=False)
    lang: str = "unknown"
    original_text: Optional[str] = None
    was_mixed: bool = False


@dataclass
class MRZResult:
    valid: bool
    surname: str
    given_names: str
    country: str
    birth_date: str
    expiry_date: str
    number: str
    sex: str


@dataclass
class PipelineOutput:
    mrz: Optional[MRZResult]
    english_lines: list[TextLine]
    english_text: str
    source: str
    confidence_avg: float
    raw_lines: list[TextLine]  # ← agregar esto