import re
from models import TextLine, MRZResult

MRZ_PATTERN = re.compile(r'^[A-Z0-9<]{30,44}$')


def _normalize_mrz_text(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "")
        .replace("«", "<")
        .replace("‹", "<")
        .replace("＜", "<")
        .upper()
    )


def _find_lines(lines: list[TextLine]) -> list[str]:
    candidates = []

    for line in lines:
        clean = _normalize_mrz_text(line.text)

        if MRZ_PATTERN.match(clean) and "<" in clean:
            candidates.append(clean)

    td3_lines = [l for l in candidates if len(l) == 44]
    td2_lines = [l for l in candidates if len(l) == 36]
    td1_lines = [l for l in candidates if len(l) == 30]

    if len(td3_lines) >= 2:
        return td3_lines[:2]

    if len(td1_lines) >= 3:
        return td1_lines[:3]

    if len(td2_lines) >= 2:
        return td2_lines[:2]

    return []


def _reconstruct(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    target = len(lines[0])

    return [
        line.ljust(target, "<")[:target]
        for line in lines
    ]


def _build_checker(lines: list[str]):
    line_count = len(lines)
    line_length = len(lines[0]) if lines else 0
    mrz_string = "\n".join(lines)

    if line_count == 2 and line_length == 44:
        from mrz.checker.td3 import TD3CodeChecker
        return TD3CodeChecker(mrz_string)

    if line_count == 2 and line_length == 36:
        from mrz.checker.td2 import TD2CodeChecker
        return TD2CodeChecker(mrz_string)

    if line_count == 3 and line_length == 30:
        from mrz.checker.td1 import TD1CodeChecker
        return TD1CodeChecker(mrz_string)

    return None


def _parse(lines: list[str]) -> MRZResult | None:
    try:
        print("MRZ lines:")
        for i, line in enumerate(lines):
            print(f"{i + 1}: {repr(line)} len={len(line)}")

        mrz_string = "\n".join(lines)
        print(f"Total length: {len(mrz_string)}")

        checker = _build_checker(lines)

        if checker is None:
            print("Unsupported MRZ format")
            return None

        is_valid = bool(checker)
        print(f"Checker valid: {is_valid}")

        try:
            f = checker.fields()
        except Exception as field_error:
            print(f"MRZ fields error: {type(field_error).__name__}: {field_error}")

            return MRZResult(
                valid=False,
                surname=None,
                given_names=None,
                country=None,
                birth_date=None,
                expiry_date=None,
                number=None,
                sex=None
            )

        return MRZResult(
            valid=is_valid,
            surname=getattr(f, "surname", None),
            given_names=getattr(f, "name", getattr(f, "given_names", None)),
            country=getattr(f, "country", None),
            birth_date=getattr(f, "birth_date", None),
            expiry_date=getattr(f, "expiry_date", None),
            number=getattr(f, "document_number", getattr(f, "number", None)),
            sex=getattr(f, "sex", None)
        )

    except Exception as e:
        print(f"MRZ parse error: {type(e).__name__}: {e}")
        return None


def detect(lines: list[TextLine]) -> MRZResult | None:
    mrz_lines = _find_lines(lines)

    print(f"MRZ candidate lines found: {len(mrz_lines)}")

    if len(mrz_lines) < 2:
        return None

    mrz_lines = _reconstruct(mrz_lines)
    result = _parse(mrz_lines)

    print(f"MRZ parse result: {result.valid if result else None}")

    return result