"""
Bilingual OCR processor for English/Hindi documents using EasyOCR.

What this script does:
    Runs EasyOCR on a scanned Indian document (visa, PAN card, Aadhaar, etc.)
    and classifies each detected word by script — English (Latin) or Hindi
    (Devanagari). Both are kept and returned as separate structured outputs
    so the analysis agent (Gemma 3) can use both for cross-validation.

Why we classify instead of discard:
    Since EasyOCR correctly recognizes both scripts, discarding Hindi would
    lose useful information. In Indian documents, key fields like name and
    document number often appear in both scripts. The agent can cross-check
    the English and Hindi versions to detect inconsistencies that indicate
    a manipulated document.

Output structure passed to the agent:
    {
        "english": {
            "text":  "reconstructed English text ordered spatially",
            "words": [ { text, confidence, geometry }, ... ]
        },
        "hindi": {
            "text":  "reconstructed Hindi text ordered spatially",
            "words": [ { text, confidence, geometry }, ... ]
        },
        "low_confidence": [
            { text, confidence, geometry }, ...
        ]
    }

Installation:
    pip install easyocr opencv-python pdf2image

    EasyOCR downloads pretrained models for 'en' and 'hi' on first run
    (~200MB). Internet connection required only the first time.

    For PDFs also install poppler:
        Windows : https://github.com/oschwartz10612/poppler-windows/releases
        Linux   : sudo apt install poppler-utils
        macOS   : brew install poppler

Usage:
    python detector.py --image visa.png
    python detector.py --image PANcard.png --debug
    python detector.py --image visa.pdf --output results/visa_result.jpg
"""

import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

font_manager.fontManager.addfont("assets/fonts/NotoSansDevanagari.ttf")
plt.rcParams["font.family"] = "Noto Sans Devanagari"

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


# ── parameters ────────────────────────────────────────────────────────────────
#
# CONFIDENCE_THRESHOLD:
#   Words below this score are considered unreliable regardless of script.
#   They are kept in a separate 'low_confidence' bucket so the agent
#   knows they exist but can treat them with lower trust.
#   0.35 is intentionally permissive for low-quality scans.
#
# LATIN_THRESHOLD:
#   Minimum fraction of characters below U+0900 (Latin Unicode range)
#   for a word to be classified as English.
#   Words above this threshold → English.
#   Words below this threshold → Hindi.
#   Mixed words (e.g. "New विभाग") are classified by their majority script.
#
# LINE_THRESHOLD:
#   Maximum vertical distance (normalized 0.0-1.0) between two words
#   to be considered part of the same line when reconstructing text.
#   0.02 works for most documents (~12px in a 600px tall image).

CONFIDENCE_THRESHOLD = 0.20
LATIN_THRESHOLD      = 0.50
LINE_THRESHOLD       = 0.02


# ── image loading ─────────────────────────────────────────────────────────────

def load_image(path: str) -> list[np.ndarray]:
    """Returns a list of BGR images, one per page if the input is a PDF."""
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


# ── script classification ─────────────────────────────────────────────────────

def classify_script(text: str) -> str:
    """
    Classifies a word as 'english', 'hindi', or 'mixed' based on the
    Unicode range of its characters.

    The Devanagari Unicode block occupies U+0900 to U+097F.
    Everything before U+0900 is Latin letters, Arabic digits, punctuation,
    and spaces.

    We count the fraction of characters that fall in each range:
        >= LATIN_THRESHOLD  → 'english'
        <  LATIN_THRESHOLD  and some Latin chars present → 'mixed'
        0% Latin            → 'hindi'

    Mixed words (e.g. "विभाग DEPT") are classified as 'mixed' so the
    caller can decide how to handle them — in practice we split them
    by character and assign each portion to its respective script.

    Examples:
        "INDIA"        → 'english'
        "विभाग"        → 'hindi'
        "New विभाग"    → 'mixed'
        "123"          → 'english'  (digits are Latin range)
    """
    if not text.strip():
        return "english"  # empty/punctuation → treat as English

    latin_chars = sum(1 for c in text if ord(c) < 0x0900)
    fraction    = latin_chars / len(text)

    if fraction >= LATIN_THRESHOLD:
        return "english"
    elif fraction == 0.0:
        return "hindi"
    else:
        return "mixed"


def split_mixed_word(text: str) -> tuple[str, str]:
    """
    Splits a mixed-script word into its Latin and Devanagari portions.

    Characters below U+0900 go to the English portion.
    Characters at or above U+0900 go to the Hindi portion.
    Spaces are assigned to whichever portion has adjacent characters.

    Returns (english_portion, hindi_portion).

    Example:
        "विभाग DEPT" → ("DEPT", "विभाग")
        "New भारत"   → ("New", "भारत")
    """
    english_chars = []
    hindi_chars   = []

    for c in text:
        if ord(c) < 0x0900:
            english_chars.append(c)
        else:
            hindi_chars.append(c)

    english_portion = ''.join(english_chars).strip()
    hindi_portion   = ''.join(hindi_chars).strip()

    return english_portion, hindi_portion


# ── EasyOCR pipeline ──────────────────────────────────────────────────────────

def run_easyocr(image_bgr: np.ndarray,
                reader: easyocr.Reader,
                debug: bool = False) -> list[dict]:
    """
    Runs EasyOCR on a single image and returns a structured list of words.

    EasyOCR returns results as:
        (bounding_box, text, confidence)

    Where bounding_box is four corner points:
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    We normalize coordinates to 0.0-1.0 range for consistency.
    """
    h, w = image_bgr.shape[:2]
    raw  = reader.readtext(image_bgr)

    words = []
    for (bbox, text, confidence) in raw:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, x2   = min(x_coords), max(x_coords)
        y1, y2   = min(y_coords), max(y_coords)

        geometry = (
            (round(x1 / w, 4), round(y1 / h, 4)),
            (round(x2 / w, 4), round(y2 / h, 4))
        )

        words.append({
            "text"      : text,
            "confidence": round(float(confidence), 3),
            "geometry"  : geometry
        })

    if debug:
        print(f"EasyOCR detected {len(words)} words total\n")

    return words


# ── classification pipeline ───────────────────────────────────────────────────

def classify_words(words: list,
                   debug: bool = False) -> tuple[list, list, list]:
    """
    Classifies each word into one of three buckets:
        english        : Latin text, reliable confidence → main validation
        hindi          : Devanagari text, reliable confidence → cross-validation
        low_confidence : any script, confidence too low to trust

    For mixed words (Latin + Devanagari in the same detected block),
    we split the text by character and add each portion to its
    respective bucket, preserving the same geometry for both since
    they came from the same region.

    Returns (english_words, hindi_words, low_confidence_words).
    """
    english        = []
    hindi          = []
    low_confidence = []

    for w in words:
        # low confidence → separate bucket regardless of script
        if w["confidence"] < CONFIDENCE_THRESHOLD:
            low_confidence.append(w)
            if debug:
                print(f"  LOW CONF  — conf={w['confidence']:.2f}: '{w['text']}'")
            continue

        script = classify_script(w["text"])

        if script == "english":
            english.append(w)
            if debug:
                print(f"  ENGLISH   — conf={w['confidence']:.2f}: '{w['text']}'")

        elif script == "hindi":
            hindi.append(w)
            if debug:
                print(f"  HINDI     — conf={w['confidence']:.2f}: '{w['text']}'")

        else:
            # mixed word — split by character into both buckets
            eng_text, hin_text = split_mixed_word(w["text"])

            if eng_text:
                english.append({**w, "text": eng_text, "original": w["text"]})
                if debug:
                    print(f"  MIXED→ENG — conf={w['confidence']:.2f}: "
                          f"'{w['text']}' → '{eng_text}'")
            if hin_text:
                hindi.append({**w, "text": hin_text, "original": w["text"]})
                if debug:
                    print(f"  MIXED→HIN — conf={w['confidence']:.2f}: "
                          f"'{w['text']}' → '{hin_text}'")

    return english, hindi, low_confidence


# ── text reconstruction ───────────────────────────────────────────────────────

def reconstruct_text(words: list,
                     line_threshold: float = LINE_THRESHOLD) -> str:
    """
    Reconstructs coherent text from a list of words by sorting them
    spatially and grouping them into lines.

    Steps:
        1. Sort all words by their vertical center (top to bottom).
        2. Group words whose vertical centers are within line_threshold
           of each other into the same line.
        3. Within each line, sort words left to right by horizontal start.
        4. Join words within a line with spaces, lines with newlines.

    line_threshold is in normalized coordinates (0.0-1.0).
    0.02 ≈ 12px in a 600px tall image — enough to absorb small alignment
    variations within a real text line.
    """
    if not words:
        return ""

    def y_center(w):
        (_, y1), (_, y2) = w["geometry"]
        return (y1 + y2) / 2

    def x_start(w):
        (x1, _), _ = w["geometry"]
        return x1

    sorted_words = sorted(words, key=y_center)

    lines   = []
    current = [sorted_words[0]]

    for word in sorted_words[1:]:
        if abs(y_center(word) - y_center(current[-1])) <= line_threshold:
            current.append(word)
        else:
            lines.append(sorted(current, key=x_start))
            current = [word]

    lines.append(sorted(current, key=x_start))

    return "\n".join(" ".join(w["text"] for w in line) for line in lines)


# ── agent output builder ──────────────────────────────────────────────────────

def build_agent_output(english: list,
                       hindi: list,
                       low_confidence: list) -> dict:
    """
    Builds the structured output that gets passed to the Gemma 3 agent.

    Structure:
        english.text        → spatially reconstructed English text
        english.words       → individual words with confidence and position
        hindi.text          → spatially reconstructed Hindi text
        hindi.words         → individual words with confidence and position
        low_confidence      → words below confidence threshold (any script)

    The agent uses english.text as the primary source for field extraction
    and hindi.text for cross-validation of key fields like name and
    document number.
    """
    return {
        "english": {
            "text" : reconstruct_text(english),
            "words": [
                {
                    "text"      : w["text"],
                    "confidence": w["confidence"],
                    "geometry"  : w["geometry"]
                }
                for w in english
            ]
        },
        "hindi": {
            "text" : reconstruct_text(hindi),
            "words": [
                {
                    "text"      : w["text"],
                    "confidence": w["confidence"],
                    "geometry"  : w["geometry"]
                }
                for w in hindi
            ]
        },
        "low_confidence": [
            {
                "text"      : w["text"],
                "confidence": w["confidence"],
                "geometry"  : w["geometry"]
            }
            for w in low_confidence
        ]
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def process(image_path: str, debug: bool = False) -> tuple[dict, np.ndarray]:
    """
    Full pipeline:
        1. EasyOCR loads pretrained models for English and Hindi.
        2. Image is loaded and passed to EasyOCR for detection
           and recognition of both scripts in one pass.
        3. Each detected word is classified by script.
        4. Text is reconstructed spatially for each script.
        5. Structured output is built for the Gemma 3 agent.
    """
    print("Loading EasyOCR models (downloads ~200MB on first run)...")
    reader = easyocr.Reader(
        ['en', 'hi'],
        gpu=False  # set to True if you have a CUDA-compatible GPU
    )

    images    = load_image(image_path)
    image_bgr = images[0]

    print("Running EasyOCR...")
    words = run_easyocr(image_bgr, reader, debug)

    english, hindi, low_confidence = classify_words(words, debug)
    output = build_agent_output(english, hindi, low_confidence)

    return output, image_bgr


# ── visualization ─────────────────────────────────────────────────────────────

def visualize(image_bgr: np.ndarray,
              output: dict,
              output_path: str = "result.jpg"):
    """
    Draws colored bounding boxes on the original image:
        Green  = English text
        Orange = Hindi / Devanagari text
        Red    = low confidence (any script)
    """
    vis  = image_bgr.copy()
    h, w = vis.shape[:2]
    GREEN  = ( 34, 197,  94)
    ORANGE = (  0, 165, 255)
    RED    = ( 59,  52, 220)

    for word in output["english"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, pt1, pt2, GREEN, 2)
        cv2.putText(vis, word["text"], (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)

    for word in output["hindi"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, pt1, pt2, ORANGE, 2)

    for word in output["low_confidence"]:
        (x1, y1), (x2, y2) = word["geometry"]
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, pt1, pt2, RED, 2)

    cv2.imwrite(output_path, vis)
    print(f"\nOutput image saved to: {output_path}")
    print("  Green  = English text")
    print("  Orange = Hindi text")
    print("  Red    = low confidence")


# ── report ────────────────────────────────────────────────────────────────────

def print_report(output: dict):
    eng = output["english"]
    hin = output["hindi"]
    low = output["low_confidence"]

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"English words : {len(eng['words'])}")
    print(f"Hindi words   : {len(hin['words'])}")
    print(f"Low confidence: {len(low)}")

    print("\n── ENGLISH TEXT ─────────────────────────────────────────")
    print(eng["text"])

    print("\n── HINDI TEXT ───────────────────────────────────────────")
    print(hin["text"])

    if low:
        print("\n── LOW CONFIDENCE (any script) ──────────────────────────")
        for w in low:
            print(f"  '{w['text']}'  conf={w['confidence']:.2f}")

    print("\n── AGENT OUTPUT (JSON) ──────────────────────────────────")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("=" * 60)



def visualize_matplotlib(image_bgr, output, output_path="result.png"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_bgr.shape[:2]

    for word in output["english"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        rect = patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#22c55e", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1 * w, y1 * h - 4, word["text"],
                fontsize=6, color="#22c55e", va="bottom")

    for word in output["hindi"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        rect = patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#f97316", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1 * w, y1 * h - 4, word["text"],
                fontsize=6, color="#f97316", va="bottom")

    for word in output["low_confidence"]:
        (x1, y1), (x2, y2) = word["geometry"]
        rect = patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#ef4444", facecolor="none"
        )
        ax.add_patch(rect)

    # legend
    legend = [
        patches.Patch(edgecolor="#22c55e", facecolor="none", label="English"),
        patches.Patch(edgecolor="#f97316", facecolor="none", label="Hindi"),
        patches.Patch(edgecolor="#ef4444", facecolor="none", label="Low confidence"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Matplotlib output saved to: {output_path}")

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bilingual OCR processor for English/Hindi documents. "
                    "Classifies text by script for Gemma 3 cross-validation."
    )
    parser.add_argument("--image",  required=True,
                        help="Path to the image or PDF file")
    parser.add_argument("--debug",  action="store_true",
                        help="Print classification result for every word")
    parser.add_argument("--output", default="result.jpg",
                        help="Path for the output image with colored bounding boxes")
    args = parser.parse_args()

    output, image_bgr = process(args.image, args.debug)
    print_report(output)
    visualize(image_bgr, output, args.output)
    visualize_matplotlib(image_bgr, output, args.output)


if __name__ == "__main__":
    main()