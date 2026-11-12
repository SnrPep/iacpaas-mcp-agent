"""Газ базы строкой таблицы и его значения: состав, чистота (доля основного компонента), сумма примесей, остаток до
100 %, вероятная ошибка ввода. Числа – из базы как есть (Decimal), сравнение точное. Операции над таблицей – query.py."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from .graph import find_node, label, walk
from .knowledge import _parts_of, composition, dec_text, impurity_fields
from .paths import join_labels, split_labels

MAIN, SUM = "\0main", "\0sum"                     # «по чистоте» (доля основного компонента), «по сумме примесей»
class AnswerError(ValueError):
    """Непонятный параметр (нет такого компонента, порог не число) – текст для модели."""


def _gases(ir: dict, meta: str, root: str) -> list[dict[str, Any]]:
    base = find_node(ir, root) if root else ir
    prefix = split_labels(root) if root else []
    out = []
    for path, node in walk(base or {}):
        if "link" in node or node.get("meta") != meta or len(path) < 2:
            continue
        parts = composition(node)
        main = parts[0]["value"] if parts and parts[0]["bound"] == "не менее" else None
        imp = impurity_fields(parts, float(main) if main is not None else None) if parts else {}
        s = imp.get("impurities_max_sum", "")
        out.append({"name": label(node), "path": join_labels(prefix + list(path[1:])),
                    "group": path[-2] if len(path) > 2 else "", "parts": {p["name"]: p for p in parts},
                    "main_name": parts[0]["name"] if parts else "", "main_text": parts[0]["text"] if parts else "",
                    "main": main, "sum": Decimal(s) if s[:1].isdigit() else None,
                    "rest": Decimal(imp["rest_to_100"]) if "rest_to_100" in imp else None,
                    "odd": imp.get("exceeds_rest") or [], "filled": bool(parts)})
    return out


def value(g: dict, k: str) -> Decimal | None:
    """Число газа по ключу: примесь – «не более … %» отдельной строкой (не основной компонент), чистота, сумма."""
    if k == MAIN:
        return g["main"]
    if k == SUM:
        return g["sum"]
    p = g["parts"].get(k)
    if p is None or k == g["main_name"] or p["bound"] != "не более" or p["unit"] != "%":
        return None
    return p["value"]


def shown(g: dict, k: str) -> str:
    if k == MAIN:
        return f"{g['main_name']} {g['main_text']}" if g["main_name"] else "доля основного компонента не указана"
    if k == SUM:
        return f"сумма примесей {dec_text(g['sum'])} %" if g["sum"] is not None else "сумма примесей не указана"
    if k == g["main_name"]:
        return f"{k} – основной компонент ({g['main_text']})"
    if (p := g["parts"].get(k)) is not None:
        return f"{k} {p['text']}"
    if sums := [f"{n} {p['text']}" for n, p in g["parts"].items() if k in _parts_of(n)]:
        return f"{k} отдельно не указан (только в сумме: {sums[0]})"
    return f"{k} нет"


def _about(k: str) -> str:
    return "чистоте" if k == MAIN else "сумме примесей" if k == SUM else k


def label_of(k: str) -> str:
    return "чистота" if k == MAIN else "сумма примесей" if k == SUM else k


def _absent(g: dict, k: str) -> bool:
    """Компонента нет у газа вовсе – ни отдельно, ни в сумме; основной компонент – не «без»."""
    return k not in g["parts"] and not any(k in _parts_of(n) for n in g["parts"]) and k != g["main_name"]


PICK_LABEL = {(MAIN, "max"): "самая высокая доля основного компонента (самые чистые)",
              (MAIN, "min"): "самая низкая доля основного компонента (наименее чистые)",
              (SUM, "max"): "больше всего сумма примесей (грязнее всего)",
              (SUM, "min"): "меньше всего сумма примесей"}


def _fmt(d: Decimal) -> str:
    """Отношение: до двух знаков, без лишних нулей (33.33, 5, 0.5)."""
    q = d.quantize(Decimal("0.01")) if d >= Decimal("0.01") else d.quantize(Decimal("0.0001"))
    return dec_text(q)
