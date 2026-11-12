"""Знания поверх графа ИР (universal JSON), чистые функции:
- карточка элемента читаемым текстом: значения по «Онтологии описания характеристик и их значений» («Не менее»,
  «Не более», «Числовой интервал», единица) сворачиваются в «не менее 99.9999 %»;
- таблица элементов группы с фильтром по доле основного компонента;
- проверка ИР по ограничениям онтологии (спецификаторы: сколько потомков; варианты альтернативы: ровно один);
- подборка: новый ИР на той же онтологии из выбранных элементов (для импорта в свою папку);
- черновик смеси: новая газовая смесь по онтологии – компоненты-ссылки на газы и концентрации."""
from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .graph import find_node, label, walk
from .paths import join_labels, split_full, split_labels

TERMINALS = ("ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ")
BOUNDS = {"Не менее": "не менее", "Не более": "не более"}
SIGNS = ("≥", "≤")
ALTERNATIVE = "описать вариант альтернативы"


# ------------------------------------------------------------------ карточка
def fmt_num(v: Any) -> str:
    """1.5e-05 → «0.000015»: та же величина без степени – модель не пересчитывает и не пишет e-нотацию."""
    s = str(v)
    if "e" in s.lower():
        try:
            return format(Decimal(s), "f")
        except InvalidOperation:
            return s
    return s


def _num(n: dict) -> bool:
    return n.get("type") == "ТЕРМИНАЛ-ЗНАЧЕНИЕ" and n.get("valtype") in ("REAL", "INTEGER") and "value" in n


def _unit(n: dict) -> bool:
    """Терминал-«копия» вроде «%» или «ppm»: значение совпадает с метапонятием."""
    return (n.get("type") == "ТЕРМИНАЛ-ЗНАЧЕНИЕ" and n.get("valtype") == "STRING"
            and str(n.get("value")) == str(n.get("meta")) and str(n.get("value")) not in SIGNS)


def _first_num(n: dict | None) -> str:
    for _, x in walk(n or {}):
        if _num(x):
            return fmt_num(x["value"])
    return "(не указано)"


def _valuey(n: dict) -> bool:
    return any(_unit(k) or _num(k) or label(k) in BOUNDS or label(k) == "Числовой интервал"
               for k in n.get("successors") or [] if "link" not in k)


def value_text(n: dict) -> str | None:
    """Значение характеристики одной фразой («не более 0.000015 %», «от 1 до 2 %», «95.0 %») или
    None, если вершина – не значение."""
    if "link" in n or not _valuey(n):
        return None
    kids = [k for k in n.get("successors") or [] if "link" not in k]
    units = list(dict.fromkeys(str(k["value"]) for k in kids if _unit(k)))
    parts: list[str] = []
    for k in kids:
        lab = label(k)
        if lab in BOUNDS:
            parts.append(f"{BOUNDS[lab]} {_first_num(k)}")
        elif lab == "Числовой интервал":
            subs = {label(c): c for c in k.get("successors") or []}
            parts.append(f"от {_first_num(subs.get('Нижняя граница'))} до {_first_num(subs.get('Верхняя граница'))}")
        elif _num(k):
            parts.append(fmt_num(k["value"]))
        elif k.get("type") == "НЕТЕРМИНАЛ" and _valuey(k):          # обёртки вроде «Содержание»
            inner = value_text(k)
            if inner:
                parts.append(inner)
    unit = " ".join(units)
    if not parts:
        return f"значение не указано (есть только единица «{unit}»)" if unit else None
    return " ".join(parts) + (f" {unit}" if unit and not parts[-1].endswith(unit) else "")


def _link_target(n: dict) -> str:
    _, concept = split_full(str(n.get("link", "")))
    return join_labels(concept)


def _term_value(n: dict) -> str:
    v = n.get("value", label(n))
    return fmt_num(v) if _num(n) else str(v)


def _brief(n: dict | None) -> str:
    """Основной компонент и сумма примесей элемента для ссылки: «Ar – не менее 99.987 %; сумма примесей
    0.014 %» – в смеси видно чистоту каждого газа без лишних вызовов."""
    if n is None:
        return ""
    comps = _components(n)
    if not comps:
        return "в базе не заполнен"
    imp = impurity_fields(composition(n), _main_value(n))["impurities_max_sum"]
    return comps[0].split(". ", 1)[-1] + f"; сумма примесей {imp}" + (" %" if imp[:1].isdigit() else "")


def card_lines(n: dict, depth: int = 0, ir: dict | None = None) -> list[str]:
    """Вершина и её подграф – строками с отступами; значения свёрнуты (value_text). ir – чтобы раскрыть ссылки."""
    pad = "  " * depth
    lab = label(n)
    if "link" in n:
        target = _link_target(n)
        brief = _brief(find_node(ir, target)) if ir else ""
        return [f"{pad}{n.get('meta') or 'ссылка'}: {lab}" + (f" ({brief})" if brief else "") + f" – `{target}`"]
    if n.get("type") in TERMINALS:
        if _unit(n) or str(n.get("value")) in SIGNS:
            return []
        return [f"{pad}{n.get('meta') or lab}: {_term_value(n)}"]
    vt = value_text(n)
    if vt is not None:
        return [f"{pad}{lab} – {vt}"]
    kids = n.get("successors") or []
    terms = [k for k in kids if k.get("type") in TERMINALS and not _unit(k)]
    vals = [k for k in kids if k.get("type") == "НЕТЕРМИНАЛ" and value_text(k) is not None]
    if len(terms) > 1 and len(vals) == 1 and len(terms) + 1 == len(kids)             and len({t.get("meta") for t in terms}) == 1 and "единиц" not in str(terms[0].get("meta")):
        # «Совокупность химических веществ»: несколько веществ и одно общее значение
        return [f"{pad}{' + '.join(label(t) for t in terms)} (суммарно) – {value_text(vals[0])}"]
    if lab.isdigit():                                   # элемент последовательности «1», «2», … – одной строкой
        inner = [line.strip() for k in kids for line in card_lines(k, 0, ir)]
        return [f"{pad}{lab}. " + ("; ".join(inner) if inner else "(пусто)")]
    meta = n.get("meta")
    head = f"{pad}{lab}" + (f" [{meta}]" if meta and meta != lab else "") + (":" if kids else "")
    out = [head]
    if n.get("comment"):
        out.append(f"{pad}  (комментарий: {' / '.join(str(n['comment']).split(chr(10)))})")
    for k in kids:
        out += card_lines(k, depth + 1, ir)
    return out


def card(n: dict, ir: dict | None = None) -> str:
    lines = card_lines(n, 0, ir)
    if not n.get("successors") and "link" not in n and n.get("type") not in TERMINALS:
        lines.append("  (в базе только название – характеристики не заполнены)")
    if "смес" in str(n.get("meta")).lower() and any("link" in x for _, x in walk(n)):
        lines.append("  (общей чистоты и суммы примесей смеси в базе нет – их не вычислять и не складывать)")
    elif (parts := composition(n)) and len(parts) > 1:
        # готовыми числами то, что модель иначе считала сама и с ошибками: сумма примесей, перевод в ppm
        f = impurity_fields(parts, _main_value(n))
        if f["impurities_max_sum"][:1].isdigit():
            lines.append(f"  Сумма допустимых примесей (все «не более», %): {f['impurities_max_sum']}"
                         + (" – вероятна ошибка ввода в базе: " + "; ".join(f["exceeds_rest"]) if f.get("exceeds_rest")
                            else ""))
        if ppm := [f"{p['name']} – {v}" for p in parts[1:] if (v := ppm_text(p))]:
            lines.append("  Те же примеси в ppm (1 % = 10 000 ppm): " + "; ".join(ppm))
    return "\n".join(lines)


def ppm_text(p: dict[str, Any]) -> str:
    """Примесь «не более 0.000065 %» → «0.65 ppm» (точно, Decimal): GigaChat переводил сам и писал «65 ppm»."""
    if p.get("bound") == "не более" and p.get("unit") == "%" and p.get("value") is not None and p["value"] < 1:
        return f"{dec_text(p['value'] * 10000)} ppm"
    return ""


def all_components(ir: dict, meta: str = "Газ") -> set[str]:
    """Все компоненты, которые встречаются в составах (Ar, CH₄, «O + Ar (суммарно)» …)."""
    return {p["name"] for _, node in walk(ir) if node.get("meta") == meta and "link" not in node
            for p in composition(node)}


def is_group(n: dict) -> bool:
    """Группа однотипных элементов (например, класс газов): потомки – нетерминалы одного метапонятия, не «1, 2, …»."""
    kids = [k for k in n.get("successors") or [] if "link" not in k]
    if not kids or any(k.get("type") != "НЕТЕРМИНАЛ" for k in kids):
        return False
    return len({k.get("meta") for k in kids}) == 1 and not all(label(k).isdigit() for k in kids)


def group_line(n: dict, limit: int = 12) -> str:
    kids = [k for k in n.get("successors") or [] if "link" not in k]
    names = ", ".join(label(k) for k in kids[:limit]) + (" …" if len(kids) > limit else "")
    return f"{label(n)} [{n.get('meta')}] – {len(kids)} элем.: {names}"


# ------------------------------------------------------------------ подсчёт по группам
def group_counts(ir: dict) -> list[dict[str, Any]]:
    """Группы второго уровня (например, классы газов в «Моногазах»): сколько в каждой элементов, по убыванию."""
    out = []
    for sec in ir.get("successors") or []:
        if "link" in sec:
            continue
        for g in sec.get("successors") or []:
            if "link" in g or g.get("type") != "НЕТЕРМИНАЛ":
                continue
            kids = [k for k in g.get("successors") or [] if "link" not in k]
            metas = Counter(k.get("meta") for k in kids)
            out.append({"section": label(sec), "group": label(g), "elements": len(kids),
                        "element_meta": metas.most_common(1)[0][0] if metas else ""})
    return sorted(out, key=lambda x: -x["elements"])


# ------------------------------------------------------------------ таблица элементов
def _components(n: dict, ir: dict | None = None) -> list[str]:
    """Строки первой вложенной последовательности «1, 2, …» (компоненты состава)."""
    for _, x in walk(n):
        kids = [k for k in x.get("successors") or [] if "link" not in k]
        if kids and all(label(k).isdigit() for k in kids):
            return [line.strip() for k in kids for line in card_lines(k, 0, ir)]
    return []


_PART = re.compile(r"^\d+\.\s+(?P<name>.+?) – (?P<bound>не менее|не более) (?P<num>\d+(?:\.\d+)?) (?P<unit>\S+)$")


def dec_text(d: Decimal) -> str:
    """Decimal без степени и лишних нулей: 0.000300 → «0.0003»."""
    return format(d.normalize(), "f")


def composition(n: dict) -> list[dict[str, Any]]:
    """Компоненты состава по строкам карточки: вещество, «не менее / не более», число (Decimal – точная
    десятичная арифметика без ошибок float), единица; строка без числа – value None."""
    out = []
    for line in _components(n):
        if m := _PART.match(line):
            out.append({"name": m["name"], "text": f"{m['bound']} {m['num']} {m['unit']}", "bound": m["bound"],
                        "value": Decimal(m["num"]), "unit": m["unit"]})
        else:
            name, _, rest = line.split(". ", 1)[-1].partition(" – ")
            out.append({"name": name, "text": rest or "(не указано)", "bound": "", "value": None, "unit": ""})
    return out


def impurity_fields(parts: list[dict[str, Any]], main: float | None) -> dict[str, Any]:
    """Сумма допустимых примесей – сложены все «не более» в % у компонентов, кроме основного (первого), и
    сколько остаётся до 100 % по основному компоненту. Что не сложилось (нет числа, другая единица) – списком."""
    out: dict[str, Any] = {}
    imp = parts[1:]
    summed = [p for p in imp if p["bound"] == "не более" and p["unit"] == "%" and p["value"] is not None]
    # примесей в базе нет – так и пишем: «0» было бы неправдой
    out["impurities_max_sum"] = dec_text(sum((p["value"] for p in summed), Decimal(0))) if summed else "не указаны"
    if skipped := [f"{p['name']} ({p['text']})" for p in imp if p not in summed]:
        out["not_summed"] = skipped
    if main is not None:
        rest = Decimal(100) - Decimal(str(main))
        out["rest_to_100"] = dec_text(rest)
        # предел примеси в сотни раз больше всего остатка до 100 % – почти наверняка ошибка ввода (не та единица).
        # В 2–20 раз бывает часто (водяные пары: чистоту основного газа считают без влаги) – это не помечаем
        if over := [f"{p['name']} – {p['text']}" for p in summed if p["value"] > rest * 100]:
            out["exceeds_rest"] = over
    return out


def _main_value(n: dict) -> float | None:
    """Доля основного компонента: первое «Не менее» в подграфе элемента."""
    for _, x in walk(n):
        if label(x) == "Не менее":
            for c in x.get("successors") or []:
                if _num(c):
                    try:
                        return float(c["value"])
                    except (TypeError, ValueError):
                        return None
    return None


def element_rows(ir: dict, meta: str, root: str = "", text: str = "", min_main: float | None = None,
                 with_composition: bool = False) -> list[dict[str, Any]]:
    base = find_node(ir, root) if root else ir
    if base is None:
        return []
    prefix = split_labels(root) if root else []
    rows = []
    for path, node in walk(base):
        if "link" in node or node.get("meta") != meta or len(path) < 2:
            continue
        name = label(node)
        if text and text.lower() not in name.lower():
            continue
        main = _main_value(node)
        if min_main is not None and (main is None or main < min_main):
            continue
        comps = _components(node)
        linked = any("link" in x for _, x in walk(node))
        fields = {str(k.get("meta")): _term_value(k) for k in node.get("successors") or []
                  if k.get("type") in TERMINALS and not _unit(k)}
        rows.append({"path": join_labels(prefix + list(path[1:])), "name": name,
                     "group": path[-2] if len(path) > 2 else "", "fields": fields,
                     "main": comps[0] if comps else "", "main_value": main,
                     "other_components": max(len(comps) - 1, 0), "filled": bool(node.get("successors"))})
        if linked:                                     # смесь: компоненты-ссылки с чистотой газов
            rows[-1]["components"] = _components(node, ir)
            # модель складывала суммы примесей компонентов в «общую» – такой величины в базе нет
            rows[-1]["mixture_total"] = "в базе нет: чистоту и примеси смеси целиком не вычислять и не складывать"
        elif comps:                                    # газ: сумма примесей – готовым числом, модель не считает
            parts = composition(node)
            rows[-1].update(impurity_fields(parts, main))
            if with_composition:
                rows[-1]["composition"] = {p["name"]: p["text"] for p in parts}
                if ppm := {p["name"]: v for p in parts[1:] if (v := ppm_text(p))}:
                    rows[-1]["composition_ppm"] = ppm
    return rows


# ------------------------------------------------------------------ одна примесь у всех газов, сравнение марок
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def comp_key(s: str) -> str:
    """«CH4» и «CH₄», «Ar» и «ar» – одно написание для сравнения."""
    return re.sub(r"[\s.,()]+", "", s.translate(_SUB).lower())


def same_words(a: str, b: str) -> bool:
    """«аргона» ~ «аргон», «водяных паров» ~ «водяные пары»: слова попарно с общим началом (без окончаний)."""
    wa, wb = a.lower().split(), b.lower().split()
    if len(wa) != len(wb):
        return False
    for x, y in zip(wa, wb):
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        if n < max(3, min(len(x), len(y)) - 2):
            return False
    return True


def component_names(ir: dict, meta: str = "Газ") -> dict[str, str]:
    """Название вещества → обозначение из самой базы: газ класса «Газы на основе Аргона» с полем «Химическое
    обозначение» Ar – значит «аргон» = Ar. Вещество – из класса, а не из первого слова названия («Газообразный азот…»
    дал бы «газообразный» = N). Новое вещество в базе распознаётся само, без правки кода."""
    out: dict[str, str] = {}
    for path, node in walk(ir):
        if node.get("meta") == meta and "link" not in node and len(path) > 2:
            sym = next((_term_value(k) for k in node.get("successors") or []
                        if k.get("meta") == "Химическое обозначение" and k.get("type") in TERMINALS), "")
            if sym and (m := re.search(r"основе\s+(\w+)", str(path[-2]))) and m.group(1).lower() not in out:
                out[m.group(1).lower()] = sym
    return out


def resolve_component(asked: str, known: set[str], names: dict[str, str] | None = None) -> str | None:
    """Компонент из вопроса → имя в составе базы: по обозначению («CH4» → «CH₄»), по названию в составе
    («водяных паров»), по названию вещества из базы и синонимам из config/components.toml («метан» → CH₄)."""
    k = comp_key(asked)
    for n in known:
        if comp_key(n) == k or same_words(asked, n):
            return n
    # «H₂», «N2», «O₂» – молекула; в составах записан элемент: H, N, O
    if k in ("h2", "n2", "o2"):
        return next((n for n in known if comp_key(n) == k[0]), None)
    for word, sym in (names or {}).items():
        if same_words(asked, word):
            return next((n for n in known if comp_key(n) == comp_key(sym)), None)
    return None


def _parts_of(name: str) -> list[str]:
    """«N + Ar + Ne (суммарно)» → ['N', 'Ar', 'Ne']."""
    return [p.strip() for p in re.sub(r"\(.*?\)", "", name).split("+") if p.strip()] if "+" in name else []


def _dec(p: dict[str, Any]) -> Decimal | None:
    return p["value"] if p["bound"] == "не более" and p["unit"] == "%" else None


def rank_component(ir: dict, meta: str, component: str, root: str = "", order: str = "desc",
                   sort_by: str = "", show: list[str] | None = None,
                   synonyms: dict[str, str] | None = None) -> dict[str, Any]:
    """Все элементы, у которых компонент указан отдельной примесью («не более», %), – по убыванию или
    возрастанию; отдельно – где он только в сумме с другими («O + Ar (суммарно)») и где он основной компонент.
    Числа – из базы как есть (Decimal), сравнение точное."""
    base = find_node(ir, root) if root else ir
    prefix = split_labels(root) if root else []
    items = []
    for path, node in walk(base or {}):
        if "link" in node or node.get("meta") != meta or len(path) < 2:
            continue
        items.append((join_labels(prefix + list(path[1:])), label(node), path[-2] if len(path) > 2 else "",
                      composition(node), node))
    # компоненты: отдельные и части сумм (Xe в базе есть только в «N + Ar + Ne + Xe + Kr (суммарно)»)
    known = {p["name"] for *_, parts, _ in items for p in parts}
    known |= {x for c in known for x in _parts_of(c)}
    names = {**component_names(ir, meta), **(synonyms or {})}     # из базы + синонимы из настроек
    name = resolve_component(component, known, names)
    if name is None:
        singles = sorted(n for n in known if "+" not in n)
        return {"error": f"Компонента «{component}» в составах нет. Есть: {', '.join(singles)}."}
    extra = [resolve_component(x, known, names) or x for x in (show or []) if x.strip()]
    key2 = resolve_component(sort_by, known, names) if sort_by else None
    if sort_by and key2 is None:
        return {"error": f"Компонента «{sort_by}» (sort_by) в составах нет. Есть: "
                         f"{', '.join(sorted(n for n in known if '+' not in n))}."}
    rows, in_sums, as_main = [], [], []
    filled = 0
    for path, title, group, parts, node in items:
        if not parts:
            continue
        filled += 1
        by = {p["name"]: p for p in parts}
        if parts[0]["name"] == name:
            as_main.append(f"{title} ({name} – {parts[0]['text']})")
            continue
        if name in by and _dec(by[name]) is not None:
            row = {"name": title, "path": path, "group": group, "value": dec_text(by[name]["value"]),
                   "text": by[name]["text"], "ppm": ppm_text(by[name]), "main": f"{parts[0]['name']} – {parts[0]['text']}"}
            if extra or key2:
                row["also"] = {x: (by[x]["text"] if x in by else "не указан") for x in dict.fromkeys(
                    extra + ([key2] if key2 else []))}
            if over := impurity_fields(parts, _main_value(node)).get("exceeds_rest"):
                row["exceeds_rest"] = over
            rows.append(row)
        for p in parts[1:]:
            if name in _parts_of(p["name"]):
                in_sums.append({"name": title, "path": path, "component": p["name"], "text": p["text"]})
    rev = order != "asc"
    if key2:                                           # «по увеличению метана»: у кого метана нет – в конце
        def v2(r):
            m = re.match(r"не более (\d+(?:\.\d+)?) %$", r["also"][key2])
            return Decimal(m.group(1)) if m else None
        have = sorted((r for r in rows if v2(r) is not None), key=v2, reverse=rev)
        rows = have + [r for r in rows if v2(r) is None]
    else:
        rows.sort(key=lambda r: Decimal(r["value"]), reverse=rev)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    seen = {r["path"] for r in rows} | {x["path"] for x in in_sums}
    # газы самого вещества по классу («Газы на основе Ксенона»): незаполненные иначе не видны вовсе
    own_empty = [title for _, title, group, parts, _ in items if not parts
                 and (m := re.search(r"основе\s+(\w+)", group))
                 and resolve_component(m.group(1), {name}, names) == name]
    # «аргоны без примеси метана»: заполненные газы, где компонента нет ни отдельно, ни в сумме
    absent = [title for path, title, _, parts, _ in items if parts and path not in seen and parts[0]["name"] != name]
    return {"component": name, "sort_by": key2 or name, "elements": len(items), "filled": filled, "rows": rows,
            "in_sums": in_sums, "as_main": as_main, "own_empty": own_empty,
            "absent": len(absent), "absent_names": absent}


def compare_rows(ir: dict, meta: str, paths: list[str]) -> dict[str, Any]:
    """Сравнение элементов: строки – компоненты, столбцы – элементы, в ячейке ограничение из базы как есть;
    внизу – сумма допустимых примесей. Готовая таблица Markdown: модели остаётся переписать её."""
    cols, missing = [], []
    for p in paths:
        node = find_node(ir, p)
        if node is None or node.get("meta") != meta:
            missing.append(p)
            continue
        parts = composition(node)
        cols.append({"name": label(node), "path": p, "by": {x["name"]: x["text"] for x in parts},
                     "order": [x["name"] for x in parts], **impurity_fields(parts, _main_value(node))})
    order: list[str] = []
    for c in cols:
        order += [n for n in c["order"] if n not in order]
    matrix = [{"component": n, **{c["name"]: c["by"].get(n, "—") for c in cols}} for n in order]
    sums = {c["name"]: c.get("impurities_max_sum", "не указаны") for c in cols}
    head = "| Компонент | " + " | ".join(c["name"] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + r["component"] + " | " + " | ".join(r[c["name"]] for c in cols) + " |" for r in matrix]
    lines.append("| Сумма примесей (все «не более»), % | " + " | ".join(sums[c["name"]] for c in cols) + " |")
    return {"columns": [{"name": c["name"], "path": c["path"]} for c in cols], "matrix": matrix,
            "impurities_max_sum": sums, "table": "\n".join(lines), "missing": missing,
            "odd": {c["name"]: c["exceeds_rest"] for c in cols if c.get("exceeds_rest")}}


def _tokens(name: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?|\w+", name.lower())


def closest_element(path: str, known: list[str]) -> str | None:
    """Настоящий путь элемента вместо сочинённого моделью («Гелий газообразный марка 5.0» → «Гелий газообразный
    (сжатый) высокой чистоты, марка 5.0»). Отличительное должно совпасть: класс (если такой есть), числа марки,
    буква марки, сорт «высший / первый / второй»; дальше – больше общих основ слов. Нет надёжного – None."""
    *cls, name = split_labels(path)
    pool = [k for k in known if split_labels(k)[:-1] == cls] or known
    toks = _tokens(name)
    must = {t for t in toks if t[0].isdigit() or len(t) == 1} | {t[:4] for t in toks if t[:4] in ("высш", "перв", "втор")}
    stems = {t[:5] for t in toks if len(t) > 2}
    subst = toks[0][:5] if toks else ""                 # вещество – первое слово названия («Гелий», «Аргон»)
    best, score = None, 1                               # хотя бы две общие основы слов
    for k in pool:
        kt = _tokens(split_labels(k)[-1])
        have = set(kt) | {t[:4] for t in kt}
        if not must <= have or subst not in {t[:5] for t in _tokens(k)}:
            continue
        s = len(stems & {t[:5] for t in kt if len(t) > 2})
        if s > score:
            best, score = k, s
    return best


# ------------------------------------------------------------------ газы по названиям: «аргон 4.8; аргон 6.0»
# GigaChat в списке путей (elements) передавал только первый путь, а строкой «газ 1; газ 2» – все (проба:
# 0 из 10 и 4 из 4). Поэтому газы приходят строкой, а настоящие пути по названиям находит код.
SKIP_WORDS = {"газ", "газа", "газы", "марка", "марки", "марку", "марок", "сорт", "сорта", "и", "на", "по", "из",
              "основе", "й", "го", "ой"}                     # «в», «с» – не предлоги: бывают марки «В»
LOOKALIKE = {"a": "а", "b": "в"}                             # латинская «A» в марке вместо русской


def split_items(value: Any) -> list[str]:
    """Строка «газ 1; газ 2» (или список) → ['газ 1', 'газ 2']. Запятая – не разделитель: она есть в названиях."""
    items = value if isinstance(value, list) else re.split(r"[;\n]", str(value or ""))
    return list(dict.fromkeys(x.strip().strip("«»\"'").strip() for x in items if str(x).strip()))


def _word_tokens(text: str) -> list[str]:
    """Слова названия для поиска: числа – Decimal-строкой («6,0» = «6.0» = «6»), доля – с «%» («He-5%» – не марка
    5.0), «первый / 1-й» – «1»."""
    out = []
    for t in re.findall(r"[a-z]+\d[a-z\d]*|\d+(?:[.,]\d+)?%?|[^\W\d_]+", text.translate(_SUB).lower().replace("ё", "е")):
        if t[0].isascii() and t[0].isalpha() and any(ch.isdigit() for ch in t):   # обозначение «CO2», «CH₄»
            out.append(t)
        elif t[0].isdigit():
            pct = t.endswith("%")
            out.append(format(Decimal(t.rstrip("%").replace(",", ".")).normalize(), "f") + "%" * pct)
        elif t.startswith("перв"):
            out.append("1")
        elif t.startswith("втор"):
            out.append("2")
        elif t not in SKIP_WORDS:
            out.append(LOOKALIKE.get(t, t) if len(t) == 1 else t)
    return out


def _word_match(q: str, have: list[str]) -> bool:
    """Число и одна буква – точно; слово – по общему началу («гелия» ~ «гелий», «высшего» ~ «высший»)."""
    if q in have or q[0].isdigit() or len(q) <= 3:          # «co2», «Е941» и короткие – только точно
        return q in have
    for h in have:
        if h[0].isdigit() or len(h) <= 3:
            continue
        n = next((i for i, (x, y) in enumerate(zip(q, h)) if x != y), min(len(q), len(h)))
        if n >= max(4, min(len(q), len(h)) - 2):
            return True
    return False


def _with_symbols(text: str, synonyms: dict[str, str]) -> str:
    """«углекислота высший сорт» → «CO₂ высший сорт»: синоним из config/components.toml – обозначением."""
    words = text.split()
    for word, sym in synonyms.items():
        n = len(word.split())
        for i in range(len(words) - n + 1):
            if same_words(" ".join(words[i:i + n]), word):
                words[i:i + n] = [sym] + [""] * (n - 1)
    return " ".join(w for w in words if w)


def same_grade(ir: dict, path: str) -> list[str]:
    """Газы того же класса с той же маркой («Гелий газообразный марка А» и «… (сжатый, марка А)»): на «гелий
    марки А» GigaChat описывал один, а в базе их два."""
    parts = split_labels(path)
    parent = find_node(ir, join_labels(parts[:-1])) if len(parts) > 1 else None
    mark = {t for t in _word_tokens(parts[-1]) if t[0].isdigit() and not t.endswith("%") or len(t) == 1}
    if parent is None or not mark:
        return []
    return [join_labels(parts[:-1] + [label(k)]) for k in parent.get("successors") or []
            if "link" not in k and label(k) != parts[-1] and k.get("meta") == find_node(ir, path).get("meta")
            and {t for t in _word_tokens(label(k)) if t[0].isdigit() and not t.endswith("%") or len(t) == 1} == mark]


def find_elements(ir: dict, items: list[str], metas: tuple[str, ...] = ("Газ",), groups: bool = False,
                  synonyms: dict[str, str] | None = None) -> dict[str, Any]:
    """Газы по названиям или путям (как в вопросе, как написала модель): «аргон 4.8», «Гелий газообразный марка Б»,
    настоящий путь. Совпасть должны все слова (без окончаний), числа марки и буквы – точно. Код не угадывает:
    подошло несколько газов – ambiguous (все варианты), ни одного – missing (с похожими). groups – можно указать и
    целый класс (путь или название, «Газы на основе Гелия»). Похоже, но не всеми словами – fuzzy (показывается)."""
    pool = []                                            # (путь, название, слова пути, слова вещества)
    group_paths = {}
    names = component_names(ir)
    for path, node in walk(ir):
        if "link" in node or len(path) < 2:
            continue
        p = join_labels(list(path[1:]))
        if node.get("meta") in metas:
            subst = re.search(r"основе\s+(.+)$", str(path[-2]) if len(path) > 2 else "")
            sw = _word_tokens(subst.group(1)) if subst else []
            # обозначение вещества класса: из базы, а у незаполненных газов («Метан…») – из синонимов
            sym = _word_tokens(names.get(subst.group(1).split()[0].lower(), "") or next(
                (s for w, s in (synonyms or {}).items() if same_words(w, subst.group(1))), "")) if subst else []
            pool.append((p, label(node), _word_tokens(" ".join(map(str, path[1:]))) + sym, sw + sym))
        elif groups and is_group(node):
            group_paths[p] = label(node)
    known = {p for p, *_ in pool}
    out: dict[str, Any] = {"paths": [], "picked": [], "ambiguous": {}, "missing": {}}

    def add(asked: str, found: list[str], how: str) -> None:
        if len(found) == 1:
            out["picked"].append({"asked": asked, "path": found[0], "how": how})
            out["paths"] += [x for x in found if x not in out["paths"]]
        else:
            out["ambiguous"][asked] = found

    for asked in items:
        path = join_labels(split_labels(asked))
        if path in known or path in group_paths:
            add(asked, [path], "path")
            continue
        last = split_labels(asked)[-1] if "/" in asked else asked
        key = " ".join(last.lower().replace("ё", "е").split())
        same = [p for p, n, *_ in pool if " ".join(n.lower().replace("ё", "е").split()) == key]
        same = same or [p for p, n in group_paths.items() if n.lower() == key]
        if same:
            add(asked, same, "name")
            continue
        want = _word_tokens(last)
        if synonyms and not all(any(_word_match(w, t) for *_, t, _ in pool) for w in want):
            want = _word_tokens(_with_symbols(last, synonyms))   # «углекислота» в названиях нет – CO₂
        if not want:
            out["missing"][asked] = []
            continue
        full = [p for p, _, toks, _ in pool if all(_word_match(w, toks) for w in want)]
        if full:
            add(asked, full, "words")
            continue
        # не все слова: «Азот газообразный … марка 6.0», а в базе «Азот высокой чистоты (марка 6.0)». Тогда числа,
        # буквы и сорт – все, вещество – совпало, слов – больше всех; иначе не угадываем
        strict = [w for w in want if w[0].isdigit() or len(w) <= 2]
        scored = []
        for p, _, toks, sw in pool:
            if all(w in toks for w in strict) and any(_word_match(w, sw) for w in want):
                scored.append((sum(_word_match(w, toks) for w in want), p))
        best = max((s for s, _ in scored), default=0)
        top = [p for s, p in scored if s == best]
        # без марки (числа или буквы) не угадываем: «гелий высокой чистоты жидкий» ≠ «Гелий газообразный … 6.0»
        if best >= 2 and len(top) == 1 and strict:
            add(asked, top, "fuzzy")
        elif best >= 2 and strict:
            out["ambiguous"][asked] = top
        else:
            # похожие – только того же вещества: «азот 6.0» не должен подсказать «гелий 6.0»
            near = sorted(((sum(_word_match(w, toks) for w in want), p) for p, _, toks, sw in pool
                           if any(_word_match(w, sw) for w in want)), reverse=True)
            out["missing"][asked] = [p for s, p in near[:3] if s]
    return out


# ------------------------------------------------------------------ проверка по онтологии
@dataclass
class _Item:
    label: str
    kind: str                 # one | max1 | min1 | any
    alt: bool
    concepts: list[dict]      # кандидаты: в онтологии бывают одноимённые вершины («Характеристика» ×3)
    onto: str
    valtype: str | None
    spec: str
    group: str = ""           # варианты альтернативы из разных «пропусков» – разные выборы


def _spec_kind(spec: str | None) -> str:
    s = (spec or "").strip().lower()
    if s in ("пропуск", "proxy"):
        return "proxy"
    if s in ("в точности один", "ровно один", "one", "копия", "copy"):
        return "one"
    if s in ("ноль или один", "onemm", "возможное отсутствие", "copymm", "копия, может отсутствовать"):
        return "max1"
    if s.startswith("непуст") or s in ("list", "set", "seq"):
        return "min1"
    return "any"


class OntologyIndex:
    """Онтологии по названию; разворачивает ссылки «→ /…» (в т. ч. на другую онтологию) и «пропуск»."""

    def __init__(self, ontologies: dict[str, dict]):
        self.ontos = ontologies

    def resolve(self, link: str) -> tuple[str, list[dict]]:
        ir_labels, concept = split_full(link)
        title = ir_labels[-1] if ir_labels else ""
        onto = self.ontos.get(title)
        if onto is None:
            return title, []
        found = [onto]
        for lab in concept:                          # все вершины с такими метками – одноимённых может быть несколько
            found = [c for n in found for c in n.get("successors") or [] if label(c) == lab and "link" not in c]
        return title, found

    def items(self, concept: dict, onto: str, _depth: int = 0, _group: str = "") -> list[_Item]:
        out: list[_Item] = []
        for k in concept.get("successors") or []:
            targets, where = [k], onto
            if "link" in k:
                where, targets = self.resolve(str(k["link"]))
            kind = _spec_kind(k.get("specifier"))
            if kind == "proxy":
                # «пропуск»: в ИР этой вершины нет – её потомки стоят прямо у родителя
                if targets and _depth < 8:
                    out += self.items(targets[0], where, _depth + 1, f"{_group}/{label(k)}")
                continue
            valtype = k.get("valtype") or (targets[0].get("valtype") if targets else None)
            out.append(_Item(label(k), kind, k.get("meta") == ALTERNATIVE, targets, where, valtype,
                             str(k.get("specifier") or ""), _group))
        return out

    def best(self, it: "_Item", node: dict) -> dict | None:
        """Из одноимённых понятий – то, чьи потомки лучше совпадают с потомками вершины ИР."""
        if len(it.concepts) <= 1:
            return it.concepts[0] if it.concepts else None
        metas = {k.get("meta") for k in node.get("successors") or []}
        return max(it.concepts, key=lambda c: len(metas & {x.label for x in self.items(c, it.onto)}))



def check(ir: dict, onto_title: str, index: OntologyIndex, max_problems: int = 2000) -> list[dict[str, str]]:
    """Нарушения ограничений онтологии: нет обязательного потомка («в точности один», «копия», «непустой…»),
    лишний («ноль или один»), не выбран вариант альтернативы, неизвестное понятие, не тот тип значения."""
    problems: list[dict[str, str]] = []
    onto = index.ontos.get(onto_title)
    if onto is None:
        return [{"path": "", "problem": f"онтология «{onto_title}» недоступна"}]

    def add(path: list[str], text: str, kind: str) -> None:
        if len(problems) < max_problems:
            problems.append({"path": join_labels(path), "problem": text, "kind": kind})

    def visit(node: dict, concept: dict, where: str, path: list[str], depth: int) -> None:
        if depth > 40:
            return
        items = index.items(concept, where)
        kids = node.get("successors") or []
        have = Counter(k.get("meta") for k in kids)
        known = {it.label: it for it in items}
        for k in kids:
            if k.get("meta") not in known:
                add(path, f"«{label(k)}»: понятия «{k.get('meta')}» здесь нет в онтологии", "неизвестное понятие")
        for it in items:
            if it.alt:
                continue
            n = have[it.label]
            if it.kind == "one" and n == 0:
                add(path, f"нет обязательного «{it.label}» ({it.spec})", "нет обязательного")
            elif it.kind in ("one", "max1") and n > 1:
                add(path, f"«{it.label}» – {n}, а допустимо не больше одного ({it.spec})", "лишний")
            elif it.kind == "min1" and n == 0:
                add(path, f"нет ни одного «{it.label}» ({it.spec})", "нет обязательного")
        for group in dict.fromkeys(it.group for it in items if it.alt):
            alts = [it for it in items if it.alt and it.group == group]
            chosen = [it.label for it in alts if have[it.label]]
            if not chosen:
                add(path, "не задан ни один вариант: " + " / ".join(it.label for it in alts), "не выбран вариант")
            elif len(chosen) > 1:
                add(path, "заданы сразу несколько вариантов: " + ", ".join(chosen), "несколько вариантов")
        for k in kids:
            it = known.get(k.get("meta"))
            if it is None or "link" in k:
                continue
            if k.get("type") in TERMINALS and it.valtype and k.get("valtype") and k["valtype"] != it.valtype:
                add(path + [label(k)], f"тип значения {k['valtype']}, а по онтологии {it.valtype}", "тип значения")
            target = index.best(it, k)
            if target is not None and k.get("type") == "НЕТЕРМИНАЛ":
                visit(k, target, it.onto, path + [label(k)], depth + 1)

    visit(ir, onto, onto_title, [], 0)
    return problems


def linked_ontologies(onto: dict) -> list[str]:
    """Пути других ИР, на которые ссылается онтология (например, «Онтология описания характеристик…»)."""
    own = str(onto.get("title") or "")
    out: list[str] = []
    for _, n in walk(onto):
        if "link" in n:
            ir_labels, _ = split_full(str(n["link"]))
            if ir_labels and ir_labels[-1] != own:
                full = str(n["link"]).split("$", 1)[0] + "$;"
                if full not in out:
                    out.append(full)
    return out


# ------------------------------------------------------------------ подборка
def full_ir_path(ir: dict, folder: str, title: str) -> str:
    """Полный путь нового ИР для ссылок: «владелец / Мой Фонд / папка / ИР». Владелец – из пути исходного ИР;
    без него – только имя (песочница в памяти)."""
    owner = str(ir.get("path") or "").split(" / Мой Фонд")[0].strip()
    if not owner or not folder:
        return title
    return " / ".join([owner, "Мой Фонд", *folder.strip("/").split("/"), title])


def build_selection(ir: dict, paths: list[str], title: str, comment: str = "",
                    notes: dict[str, str] | None = None, link_ir: str = "") -> tuple[dict, list[str], list[str]]:
    """Новый ИР на той же онтологии: выбранные элементы (или целые группы) вместе с вершинами-предками.
    Элементы, на которые ссылаются выбранные (например, газы в составе смеси), добавляются сами – иначе ссылки
    повиснут. Служебные id/link_id убираются (link_id при импорте важнее пути и вёл бы в исходный ИР), ссылки
    внутри ИР переписываются на новое имя. notes – пометки {путь: текст}, пишутся в комментарий вершины.
    Возвращает (JSON, добавленные пути, ненайденные пути)."""
    src_title = str(ir.get("title") or label(ir))
    wanted: list[list[str]] = []
    missing: list[str] = []
    for p in paths:
        labels = split_labels(p)
        if not labels or find_node(ir, p) is None:
            missing.append(p)
        elif labels not in wanted:
            wanted.append(labels)

    def covered(labels: list[str]) -> bool:
        return any(labels[:len(w)] == w for w in wanted)

    added: list[str] = []
    changed = True
    while changed:                                     # зависимости по ссылкам – пока появляются новые
        changed = False
        for w in list(wanted):
            for _, n in walk(find_node(ir, join_labels(w)) or {}):
                if "link" not in n:
                    continue
                ir_labels, concept = split_full(str(n["link"]))
                if ir_labels and ir_labels[-1] == src_title and concept and not covered(concept) \
                        and find_node(ir, join_labels(concept)) is not None:
                    wanted.append(concept)
                    added.append(join_labels(concept))
                    changed = True

    notes = {join_labels(split_labels(p)): v for p, v in (notes or {}).items()}

    def clean(n: dict, here: list[str]) -> dict:
        out = {k: v for k, v in n.items() if k not in ("id", "link_id", "successors")}
        if note := notes.get(join_labels(here)):
            out["comment"] = f"{out['comment']}\n{note}" if out.get("comment") else note
        if "link" in n:
            ir_labels, concept = split_full(str(n["link"]))
            if ir_labels and ir_labels[-1] == src_title:
                out["link"] = f"{link_ir or title}$/{join_labels(concept)};"
        return out

    def build(n: dict, here: list[str]) -> dict | None:
        if here and covered(here):
            res = clean(n, here)
            if n.get("successors") and "link" not in n:
                res["successors"] = [build(c, here + [label(c)]) for c in n["successors"]]
            return res
        on_way = any(w[:len(here)] == here for w in wanted)
        if not on_way:
            # разделы верхнего уровня оставляем пустыми: в онтологии они «копия» – обязательны
            return clean(n, here) if len(here) == 1 else None
        res = clean(n, here)
        kids = [b for c in n.get("successors") or [] if (b := build(c, here + [label(c)])) is not None]
        if kids:
            res["successors"] = kids
        return res

    body = build(ir, []) or {}
    root = {"title": title, "json_type": "universal", "ontology": ir.get("ontology"), "type": "КОРЕНЬ",
            "name": title, "meta": ir.get("meta")}
    if comment:
        root["comment"] = comment
    if body.get("successors"):
        root["successors"] = body["successors"]
    return copy.deepcopy(root), added, missing


def build_mixture(ir: dict, title: str, name: str, group: str, parts: list[tuple[str, float]],
                  standard: str = "", comment: str = "", symbol: str = "",
                  link_ir: str = "") -> tuple[dict, list[str]]:
    """Новый ИР: газы-компоненты (копии из базы) и смесь из них в «Многокомпонентных газовых смесях».
    Устройство смеси – как в онтологии газов: Газовая смесь → Соотношение компонентов → Компонент смеси
    (ссылка на газ + Концентрация, %). Возвращает (JSON, ненайденные газы)."""
    gases = [p for p, _ in parts]
    missing = [p for p in gases if (find_node(ir, p) or {}).get("meta") != "Газ"]
    if missing:
        return {}, missing
    sel, _, _ = build_selection(ir, gases, title, comment, link_ir=link_ir)
    comps = []
    for i, (gas, pct) in enumerate(parts, 1):
        comps.append({"type": "НЕТЕРМИНАЛ", "name": str(i), "meta": "Компонент смеси", "successors": [
            {"type": "НЕТЕРМИНАЛ", "name": split_labels(gas)[-1], "meta": "Газ",
             "link": f"{link_ir or title}$/{join_labels(split_labels(gas))};"},
            {"type": "НЕТЕРМИНАЛ", "name": "Концентрация", "meta": "Концентрация", "successors": [
                {"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": float(pct), "valtype": "REAL", "meta": "Числовое значение"},
                {"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": "%", "valtype": "STRING", "meta": "%"}]}]})
    mix: dict[str, Any] = {"type": "НЕТЕРМИНАЛ", "name": name, "meta": "Газовая смесь", "successors": []}
    if symbol:
        mix["successors"].append({"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": symbol, "valtype": "STRING",
                                  "meta": "Химическое обозначение"})
    mix["successors"].append(
        {"type": "НЕТЕРМИНАЛ", "name": "Соотношение компонентов газовой смеси",
         "meta": "Соотношение компонентов газовой смеси", "successors": comps})
    if standard:
        mix["successors"].append({"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": standard, "valtype": "STRING",
                                  "meta": "Стандарт (норматив)"})
    if comment:
        mix["comment"] = comment
    section = next((s for s in sel.get("successors") or [] if s.get("meta") == "Многокомпонентные газовые смеси"),
                   None)
    if section is None:
        section = {"type": "НЕТЕРМИНАЛ", "name": "Многокомпонентные газовые смеси",
                   "meta": "Многокомпонентные газовые смеси"}
        sel.setdefault("successors", []).append(section)
    section.setdefault("successors", []).append(
        {"type": "НЕТЕРМИНАЛ", "name": group, "meta": "Класс газовых смесей", "successors": [mix]})
    return sel, []
