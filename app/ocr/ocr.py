import sys
import os
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Missing OpenCV.  pip install opencv-python")

try:
    import easyocr
except ImportError:
    sys.exit("Missing EasyOCR. pip install easyocr")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

from .classification import classify_words
from .reconstruction import build_agent_output


def load_image(path: str) -> list[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if not PDF2IMAGE_OK:
            sys.exit("For PDFs: pip install pdf2image (and poppler)")
        pages = convert_from_path(path, dpi=300)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not load image: {path}")
    return [img]


def run_easyocr(image_bgr: np.ndarray,
                reader: easyocr.Reader,
                page: int = 1,
                debug: bool = False) -> list[dict]:
    h, w = image_bgr.shape[:2]
    raw = reader.readtext(
        image_bgr,
        paragraph=False,
        detail=1,
        width_ths=0.7,      # más alto = agrupa más caracteres cercanos
        contrast_ths=0.1,
        adjust_contrast=0.5,
        link_threshold=0.3  # controla cómo CRAFT une caracteres en palabras
    )

    words = []
    for (bbox, text, confidence) in raw:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, x2   = min(x_coords), max(x_coords)
        y1, y2   = min(y_coords), max(y_coords)

        words.append({
            "text"      : text,
            "confidence": round(float(confidence), 3),
            "page"      : page,
            "geometry"  : (
                (round(x1 / w, 4), round(y1 / h, 4)),
                (round(x2 / w, 4), round(y2 / h, 4))
            )
        })

    if debug:
        print(f"Page {page}: EasyOCR detected {len(words)} words")

    return words


def process(image_path: str, debug: bool = False) -> tuple[dict, np.ndarray]:
    print("Loading EasyOCR models (downloads ~200MB on first run)...")
    reader = easyocr.Reader(
        ['en', 'hi'],
        gpu=False,
        recog_network='standard'  # prueba cambiar a 'latin_g2'
    )

    images = load_image(image_path)
    print(f"Pages found: {len(images)}")

    all_english        = []
    all_hindi          = []
    all_low_confidence = []

    for page_num, image_bgr in enumerate(images, start=1):
        if debug:
            print(f"\n── Page {page_num} ───────────────────────────")

        words = run_easyocr(image_bgr, reader, page=page_num, debug=debug)
        english, hindi, low_confidence = classify_words(words, debug)

        all_english        .extend(english)
        all_hindi          .extend(hindi)
        all_low_confidence .extend(low_confidence)

    output = build_agent_output(all_english, all_hindi, all_low_confidence)
    return output, images[0]