LINE_THRESHOLD  = 0.02
FIELD_THRESHOLD = 0.06


def y_center(word: dict) -> float:
    (_, y1), (_, y2) = word["geometry"]
    return (y1 + y2) / 2


def x_start(word: dict) -> float:
    (x1, _), _ = word["geometry"]
    return x1


def line_y_avg(line: list) -> float:
    return sum(y_center(w) for w in line) / len(line)


def group_into_lines(words: list,
                     line_threshold: float = LINE_THRESHOLD) -> list[list]:
    if not words:
        return []

    sorted_words = sorted(words, key=y_center)
    lines        = []
    current      = [sorted_words[0]]

    for word in sorted_words[1:]:
        if abs(y_center(word) - line_y_avg(current)) <= line_threshold:
            current.append(word)
        else:
            lines.append(sorted(current, key=x_start))
            current = [word]

    lines.append(sorted(current, key=x_start))
    return lines


def group_into_fields(lines: list,
                      field_threshold: float = FIELD_THRESHOLD) -> list[dict]:
    if not lines:
        return []

    fields  = []
    current = [lines[0]]

    for line in lines[1:]:
        if abs(line_y_avg(line) - line_y_avg(current[-1])) <= field_threshold:
            current.append(line)
        else:
            fields.append(current)
            current = [line]

    fields.append(current)

    result = []
    for field in fields:
        result.append({
            "lines"     : [" ".join(w["text"] for w in line) for line in field],
            "full_text" : " ".join(w["text"] for line in field for w in line),
            "geometry"  : {
                "top_left"    : field[0][0]["geometry"][0],
                "bottom_right": field[-1][-1]["geometry"][1]
            }
        })

    return result


def reconstruct_text(words: list,
                     line_threshold: float = LINE_THRESHOLD) -> str:
    lines = group_into_lines(words, line_threshold)
    return "\n".join(" ".join(w["text"] for w in line) for line in lines)


def build_agent_output(english: list,
                       hindi: list,
                       low_confidence: list) -> dict:
    eng_lines  = group_into_lines(english)
    hin_lines  = group_into_lines(hindi)
    eng_fields = group_into_fields(eng_lines)
    hin_fields = group_into_fields(hin_lines)

    return {
        "english": {
            "text"  : reconstruct_text(english),
            "fields": eng_fields,
            "words" : [
                {
                    "text"      : w["text"],
                    "confidence": w["confidence"],
                    "geometry"  : w["geometry"]
                }
                for w in english
            ]
        },
        "hindi": {
            "text"  : reconstruct_text(hindi),
            "fields": hin_fields,
            "words" : [
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