"""Пути IACPaaS (см. «JSON-представление ИР», раздел «Пути»):
  ИР:      "user@mail.ru / Мой Фонд / папка / Имя ИР$;"
  вершина: "user@mail.ru / Мой Фонд / папка / Имя ИР$ / вершина / подвершина;"
  метка с "/" обрамляется тройным штрихом: ‴г/л‴ (U+2034).
REST API ждёт путь ИР ВНУТРИ «Мой Фонд» без префикса "<e-mail>/Мой Фонд/".
Функции ниже принимают любую из этих форм и приводят к каноническому виду "папка/Имя ИР"."""
from __future__ import annotations

BORDER = "‴"   # ‴
MY_FUND = "Мой Фонд"


class PathError(ValueError):
    pass


def split_labels(path: str) -> list[str]:
    """Разбивает путь на метки по '/', не трогая '/' внутри ‴…‴. Пробелы вокруг меток убираются."""
    labels, buf, quoted = [], [], False
    for ch in path:
        if ch == BORDER:
            quoted = not quoted
            continue
        if ch == "/" and not quoted:
            labels.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if quoted:
        raise PathError(f"Незакрытый тройной штрих {BORDER} в пути: {path!r}")
    labels.append("".join(buf).strip())
    return [label for label in labels if label != ""]


def join_labels(labels: list[str]) -> str:
    return "/".join(f"{BORDER}{label}{BORDER}" if "/" in label else label for label in labels)


def _strip_tail(path: str) -> str:
    s = path.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def split_full(path: str) -> tuple[list[str], list[str]]:
    """Полный путь -> (метки пути ИР, метки пути вершины). '$' отделяет ИР от пути вершины."""
    s = _strip_tail(path)
    if "$" in s:
        ir_part, concept_part = s.split("$", 1)
    else:
        ir_part, concept_part = s, ""
    return split_labels(ir_part), split_labels(concept_part)


def normalize_ir_path(path: str) -> str:
    """Любая форма пути ИР -> "папка/Имя ИР" внутри «Мой Фонд»."""
    if not path or not path.strip():
        raise PathError("Пустой путь ИР")
    ir_labels, _ = split_full(path)
    if MY_FUND in ir_labels:
        ir_labels = ir_labels[ir_labels.index(MY_FUND) + 1:]
    if not ir_labels:
        raise PathError(f"В пути нет имени ИР: {path!r}")
    return join_labels(ir_labels)


def normalize_concept_path(path: str) -> str:
    """Путь вершины внутри ИР: "a/b", "/a/b", "a / b" или полный путь с '$' -> "a/b". Пусто = корень."""
    if not path or not path.strip():
        return ""
    s = _strip_tail(path)
    if "$" in s:
        _, labels = split_full(s)
    else:
        labels = split_labels(s)
    return join_labels(labels)


def full_ir_path(path: str, user_email: str | None) -> str:
    """Канонический путь -> полный путь платформы (для сообщений и поля path в JSON)."""
    rel = normalize_ir_path(path)
    prefix = f"{user_email}/{MY_FUND}/" if user_email else f"{MY_FUND}/"
    return f"{prefix}{rel}$;"
