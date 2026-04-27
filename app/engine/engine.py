import os
from pathlib import Path

from app.engine.base import OCREngine
from app.engine.image import load_image
from models import Config, TextLine
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"

import numpy as np



class PaddleOCRAdapter(OCREngine):
    """PaddleOCR v3.x implementation of OCREngine."""

    def __init__(self, config: Config, **paddle_kwargs) -> None:
        from paddleocr import PaddleOCR
        self._threshold = config.confidence_threshold
        self._engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **paddle_kwargs,
        )

    def extract(self,
                image: np.ndarray | str | Path,
                max_side: int = 1_600) -> list[TextLine]:

        if isinstance(image, (str, Path)):
            image = load_image(image, max_side)

        lines: list[TextLine] = []
        for result in self._engine.predict(image):
            res    = result.get("res", result)
            texts  = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            bboxes = res.get("dt_polys", res.get("rec_polys", res.get("det_polys", [])))
            for text, score, bbox in zip(texts, scores, bboxes):
                if score >= self._threshold and text.strip():
                    lines.append(TextLine(
                        text=text.strip(),
                        confidence=round(float(score), 4),
                        bbox=np.array(bbox, dtype=np.int32),
                    ))
        return lines
