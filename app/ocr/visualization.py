import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "..", "..", "assets", "fonts", "NotoSansDevanagari.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans Devanagari"
else:
    print(f"Warning: font not found at {FONT_PATH}. Hindi text may not render correctly.")


def visualize(image_bgr: np.ndarray,
              output: dict,
              output_path: str = "result.png"):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image_rgb)
    ax.axis("off")

    h, w = image_bgr.shape[:2]

    for word in output["english"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        ax.add_patch(patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#22c55e", facecolor="none"
        ))
        ax.text(x1 * w, y1 * h - 4, word["text"],
                fontsize=6, color="#22c55e", va="bottom")

    for word in output["hindi"]["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        ax.add_patch(patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#f97316", facecolor="none"
        ))
        ax.text(x1 * w, y1 * h - 4, word["text"],
                fontsize=6, color="#f97316", va="bottom")

    for word in output["low_confidence"]:
        (x1, y1), (x2, y2) = word["geometry"]
        ax.add_patch(patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=1.5, edgecolor="#ef4444", facecolor="none"
        ))

    ax.legend(handles=[
        patches.Patch(edgecolor="#22c55e", facecolor="none", label="English"),
        patches.Patch(edgecolor="#f97316", facecolor="none", label="Hindi"),
        patches.Patch(edgecolor="#ef4444", facecolor="none", label="Low confidence"),
    ], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Output saved to: {output_path}")


def print_report(output: dict):
    eng = output["english"]
    hin = output["hindi"]
    low = output["low_confidence"]

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"English words  : {len(eng['words'])}")
    print(f"Hindi words    : {len(hin['words'])}")
    print(f"Low confidence : {len(low)}")

    print("\n── ENGLISH TEXT ─────────────────────────────────────────")
    print(eng["text"])

    print("\n── ENGLISH FIELDS ───────────────────────────────────────")
    for i, field in enumerate(eng["fields"], 1):
        print(f"  [{i}] {' | '.join(field['lines'])}")

    print("\n── HINDI TEXT ───────────────────────────────────────────")
    print(hin["text"])

    print("\n── HINDI FIELDS ─────────────────────────────────────────")
    for i, field in enumerate(hin["fields"], 1):
        print(f"  [{i}] {' | '.join(field['lines'])}")

    if low:
        print("\n── LOW CONFIDENCE ───────────────────────────────────────")
        for w in low:
            print(f"  '{w['text']}'  conf={w['confidence']:.2f}")