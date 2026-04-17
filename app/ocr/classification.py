CONFIDENCE_THRESHOLD = 0.20
LATIN_THRESHOLD      = 0.85
DEVANAGARI_THRESHOLD = 0.85


def classify_script(text: str) -> str:
    clean = [c for c in text if c.strip()]
    if not clean:
        return "english"

    latin_chars    = sum(1 for c in clean if ord(c) < 0x0900)
    fraction_latin = latin_chars / len(clean)

    if fraction_latin >= LATIN_THRESHOLD:
        return "english"
    elif fraction_latin <= (1 - DEVANAGARI_THRESHOLD):
        return "hindi"
    else:
        return "mixed"


def split_mixed_word(text: str) -> tuple[str, str]:
    english_chars = []
    hindi_chars   = []

    for c in text:
        if not c.strip():
            continue
        if ord(c) < 0x0900:
            english_chars.append(c)
        else:
            hindi_chars.append(c)

    return ''.join(english_chars).strip(), ''.join(hindi_chars).strip()


def classify_words(words: list, debug: bool = False) -> tuple[list, list, list]:
    english        = []
    hindi          = []
    low_confidence = []

    for w in words:
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
            eng_text, hin_text = split_mixed_word(w["text"])

            if eng_text:
                english.append({**w, "text": eng_text, "original": w["text"]})
                if debug:
                    print(f"  MIXED→ENG — '{w['text']}' → '{eng_text}'")
            if hin_text:
                hindi.append({**w, "text": hin_text, "original": w["text"]})
                if debug:
                    print(f"  MIXED→HIN — '{w['text']}' → '{hin_text}'")

    return english, hindi, low_confidence