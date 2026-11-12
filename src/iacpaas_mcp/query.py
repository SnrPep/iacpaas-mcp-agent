"""Запрос к таблице газов: модель присылает план (операция и фильтры по схеме), код считает по данным базы и
пишет готовый кусок ответа и строку «Понял вопрос так: …». Чисел и названий газов модель сама не пишет.

Таблица строится из ИР целиком при каждом запросе (база маленькая): газ – путь, класс, раздел, марка / сорт,
стандарт, состояние, состав (Decimal, как записано в базе), сумма примесей, остаток до 100 %. Операции – закрытый
набор (OPS): нет нужной – ошибка «не умею», а не догадка."""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from . import answers
from .answers import MAIN, SUM, AnswerError, label_of, shown, value
from .graph import label, walk
from .knowledge import _components, comp_key, component_names, dec_text, find_elements, resolve_component, \
    same_words, split_items
from .paths import join_labels, split_labels

OPS = ("list", "count", "max", "min", "sort", "card", "compare", "impurities", "errors", "unfilled", "classes",
       "mixtures")
GAS, MIX = "Газ", "Газовая смесь"
PPM = Decimal(10000)
OP_TEXT = {"list": "список", "count": "сколько", "max": "больше всего", "min": "меньше всего", "sort": "по порядку",
           "card": "состав и данные газа", "compare": "сравнение таблицей", "impurities": "примеси групп",
           "errors": "ошибки в составе", "unfilled": "газы только с названием", "classes": "классы газов",
           "mixtures": "газовые смеси"}


def block_id(tool: str, args: dict[str, Any]) -> str:
    """Метка куска: один и тот же запрос – одна и та же метка."""
    raw = json.dumps({"tool": tool, **args}, ensure_ascii=False, sort_keys=True)
    return "{{#" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:4] + "}}"


def data_version(ir: dict) -> str:
    """Версия данных: дата изменения ИР на платформе (нет – хеш содержимого)."""
    if ir.get("date"):
        return str(ir["date"])
    return hashlib.sha1(json.dumps(ir, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ таблица
def _state(name: str) -> str:
    return "жидкий" if re.search(r"жидк|сжиж", name.lower()) else "газообразный"


def gas_table(ir: dict) -> list[dict[str, Any]]:
    """Все газы базы строками таблицы (поля состава – как в answers: parts, main, sum, rest, odd)."""
    nodes = {join_labels(list(p[1:])): n for p, n in walk(ir) if n.get("meta") == GAS and "link" not in n}
    rows = []
    for g in answers._gases(ir, GAS, ""):
        node = nodes.get(g["path"], {})
        fields = {str(k.get("meta")): str(k.get("value") if k.get("value") is not None else label(k)).strip()
                  for k in node.get("successors") or [] if k.get("type") != "НЕТЕРМИНАЛ" and "link" not in k}
        parts = split_labels(g["path"])
        rows.append({**g, "section": parts[0] if parts else "", "grade": fields.get("Марка", ""),
                     "sort": fields.get("Сорт", ""), "standard": fields.get("Стандарт (норматив)", ""),
                     "symbol": fields.get("Химическое обозначение", ""), "state": _state(g["name"])})
    return rows


_MIXPART = re.compile(r"^\d+\.\s+Газ:\s+(?P<name>.+?)(?:\s+\((?P<pure>[^)]*)\))?\s+–\s+`(?P<path>[^`]+)`"
                      r"(?:;\s*Концентрация\s+–\s+(?P<conc>[\d.,]+)\s*%)?")


def mixture_table(ir: dict) -> list[dict[str, Any]]:
    """Смеси: название, путь, компоненты (газ, его путь, концентрация в смеси – как в базе)."""
    out = []
    for p, n in walk(ir):
        if n.get("meta") != MIX or "link" in n:
            continue
        comps = []
        for line in _components(n, ir):
            if m := _MIXPART.match(line):
                comps.append({"gas": m["name"], "path": m["path"], "purity": m["pure"] or "",
                              "share": m["conc"].replace(",", ".") if m["conc"] else ""})
        out.append({"name": label(n), "path": join_labels(list(p[1:])), "group": p[-2] if len(p) > 2 else "",
                    "components": comps})
    return out


# ------------------------------------------------------------------ какие газы
_CLASS = re.compile(r"Газы на основе (.+)")


def _class_of(word: str, rows: list[dict]) -> list[str]:
    """«аргон», «аргоны», «газы на основе аргона», путь «Моногазы/Газы на основе Аргона» → класс(ы) «Газы на основе
    Аргона»."""
    word = word.strip().rsplit("/", 1)[-1] if "/" in word else word
    w = re.sub(r"^(?:все\s+)?(?:газы\s+на\s+основе\s+)?", "", word.strip().lower()).strip()
    if not w:
        return []
    out = []
    for g in {r["group"] for r in rows}:
        m = _CLASS.fullmatch(g)
        subst = m[1].lower() if m else g.lower()
        first = subst.split()[0]
        if g.lower() == word.strip().lower() or same_words(w, subst) or (" " not in w and same_words(w, first)) \
                or same_words(re.sub(r"(ы|и|ов|ев)$", "", w), first):
            out.append(g)
    return sorted(out)


class Pick:
    """Газы запроса: одно множество на каждый пункт «gases» (класс или названия газов)."""

    def __init__(self, ir: dict, rows: list[dict], gases: str, synonyms: dict[str, str] | None):
        self.sets: list[tuple[str, list[dict]]] = []      # (подпись, газы)
        self.notes: list[str] = []
        self.missing: list[str] = []
        by_path = {r["path"]: r for r in rows}
        items = split_items(gases)
        if not items or [x.lower() for x in items] in (["все"], ["все газы"], ["все газы базы"]):
            self.sets.append(("все газы базы", rows))
            return
        for item in items:
            if classes := _class_of(item, rows):
                self.sets.append((" и ".join(f"«{c}»" for c in classes), [r for r in rows if r["group"] in classes]))
                continue
            found = find_elements(ir, [item], (GAS,), synonyms=synonyms)
            if found["missing"]:
                near = found["missing"].get(item) or []
                self.missing.append(f"газа «{item}» в базе нет"
                                    + (f" (похожие: {', '.join('«' + split_labels(x)[-1] + '»' for x in near)})"
                                       if near else ""))
                continue
            paths = list(found["paths"])
            for many in found["ambiguous"].values():
                paths += [x for x in many if x not in paths]
                self.notes.append(f"«{item}» – подходят {len(many)}: "
                                  + ", ".join(f"«{split_labels(x)[-1]}»" for x in many) + " – взяты все.")
            got = [by_path[p] for p in paths if p in by_path]
            if got:
                self.sets.append((", ".join(f"«{g['name']}»" for g in got), got))

    @property
    def gases(self) -> list[dict]:
        seen, out = set(), []
        for _, gs in self.sets:
            for g in gs:
                if g["path"] not in seen:
                    seen.add(g["path"])
                    out.append(g)
        return out

    @property
    def title(self) -> str:
        return "; ".join(t for t, _ in self.sets) or "ничего"


class Keys:
    """Слово из запроса → ключ таблицы: компонент как в составах базы, «чистота» или «сумма примесей»."""

    def __init__(self, ir: dict, rows: list[dict], synonyms: dict[str, str] | None):
        known = {n for g in rows for n in g["parts"]}
        self.known = known | {x for c in known for x in answers._parts_of(c)}
        self.names = {**component_names(ir, GAS), **(synonyms or {})}

    def find(self, word: str) -> str | None:
        w = re.sub(r"^\s*(?:примес\w*|содержани\w*|доля|количеств\w*)\s+", "", word.strip().lower()).strip()
        if re.match(r"чист|основн|дол[яи] основ", w):
            return MAIN
        if re.match(r"сумм", w):
            return SUM
        return resolve_component(word.strip(), self.known, self.names) or resolve_component(w, self.known, self.names)

    def key(self, word: str) -> str:
        if (k := self.find(word)) is None:
            raise AnswerError(f"Компонента «{word.strip()}» в составах нет. Есть: {', '.join(self.singles())}; ещё "
                              "«чистота» (доля основного компонента) и «сумма примесей».")
        return k

    def singles(self) -> list[str]:
        return sorted(n for n in self.known if "+" not in n)


# ------------------------------------------------------------------ условия
OPS_SIGN = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "≤": "≤", "≥": "≥", "=": "="}
WORDS = (("не больше", "≤"), ("не более", "≤"), ("не меньше", "≥"), ("не менее", "≥"), ("не ниже", "≥"),
         ("не выше", "≤"), ("больше", ">"), ("меньше", "<"), ("выше", ">"), ("ниже", "<"), ("чище", ">"))


def _number(text: str) -> tuple[Decimal, bool] | None:
    """«0,01 %» → (0.01, False); «1 ppm» → (1, True)."""
    m = re.fullmatch(r"\s*([\d]+(?:[.,]\d+)?)\s*(%|ppm)?\s*", text, re.I)
    if not m:
        return None
    try:
        return Decimal(m[1].replace(",", ".")), (m[2] or "").lower() == "ppm"
    except InvalidOperation:
        return None


def parse_where(where: str, keys: Keys, ir: dict, rows: list[dict], synonyms) -> list[tuple[str, str, Decimal, str]]:
    """«N > 0.01; O < 1 ppm; чистота > Азот 6.0» → [(ключ, знак, порог в %, как показать порог)]."""
    out = []
    for item in [x for x in re.split(r"[;\n]|\s+и\s+", where) if x.strip()]:
        m = re.fullmatch(r"\s*(.+?)\s*(<=|>=|<|>|≤|≥|=)\s*(.+?)\s*", item)
        if m:
            left, op, right = m[1], OPS_SIGN[m[2]], m[3]
        else:
            w = next(((w, s) for w, s in WORDS if re.search(rf"\s{w}\s", f" {item} ")), None)
            if not w:
                raise AnswerError(f"Условие «{item.strip()}» не понятно. Пример: 'N > 0.01; O <= 1 ppm; чистота > "
                                  "Азот 6.0'.")
            left, _, right = item.partition(f" {w[0]} ")
            op = w[1]
            left = left or "чистота"                          # «чище Азот 6.0»
        if re.match(r"\s*(?:state|состояни|жидк|газообр|марк|сорт|grade)", left, re.I) or "!" in item:
            raise AnswerError(f"Условие «{item.strip()}» – не о составе: жидкие / газообразные – state, марка или сорт "
                              "– grade, «кроме …» – name_not; в where – только компоненты, чистота и сумма примесей.")
        k = keys.key(left)
        if (num := _number(right)) is not None:
            x, ppm = num
            x = x / PPM if ppm else x
            if op == "=" and x == 0:                          # «метан = 0» – без метана
                out.append((k, "нет", x, "нет"))
                continue
            out.append((k, op, x, f"{dec_text(num[0])} {'ppm' if ppm else '%'}"))
            continue
        found = find_elements(ir, [right.strip()], (GAS,), synonyms=synonyms)
        if len(found["paths"]) != 1 or found["ambiguous"]:
            # «сумма примесей > чистота / остаток» – не порог, а ошибка состава
            if re.search(r"^\s*(?:чистот\w*|остат\w*|100\s*-.*|основн\w*.*)\s*$", right, re.I):
                raise AnswerError("Сумма примесей больше остатка до 100 % (примесей больше, чем самого газа) – это "
                                  "ошибка состава: op=\"errors\".")
            raise AnswerError(f"Порог «{right.strip()}» – не число и не один газ базы. Пример: 'O < 0.001' или "
                              "'чистота > Азот высокой чистоты (марка 6.0)'.")
        ref = next(r for r in rows if r["path"] == found["paths"][0])
        if (x := value(ref, k)) is None:
            raise AnswerError(f"У «{ref['name']}» {label_of(k)} отдельным числом не указан – сравнить нельзя.")
        out.append((k, op, x, f"чем у «{ref['name']}» ({dec_text(x)} %)"))
    return out


def _test(v: Decimal, op: str, x: Decimal) -> bool:
    return {"<": v < x, "≤": v <= x, ">": v > x, "≥": v >= x, "=": v == x}[op]


def _grade_ok(g: dict, grade: str) -> bool:
    """Марка или сорт: поле «Марка» / «Сорт» или в названии («первый» = «1 сорт» = «первого сорта»)."""
    w = grade.strip().lower().replace(",", ".")
    name = g["name"].lower().replace(",", ".")
    if w in (g["grade"].lower().replace(",", "."), g["sort"].lower()):
        return True
    for word, alts in (("перв", ("перв", "1 сорт")), ("втор", ("втор", "2 сорт")), ("высш", ("высш",))):
        if w.startswith(word) or w in (a for a in alts):
            return any(a in name for a in alts) or g["sort"].lower().startswith(word)
    if re.fullmatch(r"\d+(?:\.\d+)?|[a-zа-я]", w):
        return bool(re.search(rf"(?:марк\w*|сорт)\s*{re.escape(w)}(?![\d.]*\d)", name)) or \
            bool(re.search(rf"(?<![\d.]){re.escape(w)}(?![\d.]*\d)\s*(?:сорт)", name))
    return False


def _std_key(s: str) -> str:
    return re.sub(r"[\s\-–]", "", s.lower())


# ------------------------------------------------------------------ оформление
def _count(n: int) -> str:
    one = n % 10 == 1 and n % 100 != 11
    word = "газ" if one else "газа" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "газов"
    return f"{n} {word}"


def _names(gs: list[dict], limit: int = 12) -> str:
    return ", ".join(f"«{g['name']}»" for g in gs[:limit]) + (f" и ещё {len(gs) - limit}" if len(gs) > limit else "")


def _few(gs: list[dict]) -> str:
    """Для пояснений: до 4 названий, иначе число и первые 3."""
    return _names(gs) if len(gs) <= 4 else f"{_count(len(gs))} ({_names(gs, 3)})"


def _show(g: dict, k: str, ppm: bool) -> str:
    s = shown(g, k)
    v = value(g, k)
    if ppm and v is not None and k not in (MAIN,):
        s += f" ({dec_text(v * PPM)} ppm)"
    return s


def _line(g: dict, keys: list[str], ppm: bool) -> str:
    bits = [_show(g, k, ppm) for k in keys]
    return f"- **{g['name']}**" + (" – " + "; ".join(bits) if bits else "")


def _card(g: dict, ppm: bool, only: list[str] | None = None) -> str:
    """Карточка газа: класс, марка / сорт, стандарт, состав; only – только спрошенное («сколько неона в гелии 5.5»)."""
    head = [f"**{g['name']}**", f"класс «{g['group']}»"]
    if g["grade"]:
        head.append(f"марка {g['grade']}")
    if g["sort"]:
        head.append(f"сорт {g['sort']}")
    if g["standard"]:
        head.append(f"стандарт {g['standard']}")
    lines = [" – ".join(head[:1]) + " (" + "; ".join(head[1:]) + ")"]
    if not g["filled"]:
        return lines[0] + "\n- в базе только название, состав не заполнен"
    if only:
        return "\n".join(lines + [f"- {_show(g, k, ppm)}" for k in only])
    for n, p in g["parts"].items():
        extra = ""
        if ppm and p["bound"] == "не более" and p["unit"] == "%" and p["value"] is not None:
            extra = f" ({dec_text(p['value'] * PPM)} ppm)"
        lines.append(f"- {n} – {p['text']}{extra}" + (" – основной компонент" if n == g["main_name"] else ""))
    if g["sum"] is not None:
        lines.append(f"- сумма примесей (все «не более» в %) – {dec_text(g['sum'])} %"
                     + (f"; остаток до 100 % – {dec_text(g['rest'])} %" if g["rest"] is not None else ""))
    if g["sum"] is not None and g["rest"] is not None and g["sum"] > g["rest"]:
        lines.append("- _ошибка в составе: сумма примесей больше остатка до 100 %_")
    return "\n".join(lines)


# ------------------------------------------------------------------ запрос
def run(ir: dict, *, op: str = "list", gases: str = "", where: str = "", has: str = "", without: str = "",
        name_has: str = "", name_not: str = "", state: str = "", grade: str = "", standard: str = "", by: str = "",
        order: str = "desc", top: int = 0, show: str = "", unit: str = "%",
        synonyms: dict[str, str] | None = None) -> dict[str, Any]:
    """План → кусок ответа. Порядок отбора: газы (класс / названия) → название, состояние, марка, стандарт → «без» →
    «есть» → условия → операция (крайние со всеми равными, топ-N, сортировка, таблица…)."""
    if op not in OPS:
        raise AnswerError(f"Операции «{op}» нет. Умею: {', '.join(OPS)}.")
    rows = gas_table(ir)
    ppm = unit.strip().lower() == "ppm"
    if op == "classes":
        return _classes(rows)
    if op == "mixtures":
        return _mixtures(ir, gases)
    pick = Pick(ir, rows, gases, synonyms)
    keys = Keys(ir, rows, synonyms)
    said = [f"газы – {pick.title}"]
    notes = list(pick.notes)
    if pick.missing:
        notes.extend(m[0].upper() + m[1:] + "." for m in pick.missing)
        if not pick.sets:
            text = "**" + "; ".join(m[0].upper() + m[1:] for m in pick.missing) + ".**"
            return _done(op, said, text, [], [])
    gs = pick.gases
    if name_has.strip():
        gs = [g for g in gs if name_has.strip().lower() in g["name"].lower()]
        said.append(f"в названии «{name_has.strip()}»")
    for w in [x.strip() for x in re.split(r"[;,]", name_not) if x.strip()]:
        stems = [s[:max(3, len(s) - 2)] for s in re.findall(r"\w+", w.lower())]
        gs = [g for g in gs if not all(s in g["name"].lower() for s in stems)]
        said.append(f"кроме «{w}»")
    if state.strip():
        st = "жидкий" if state.strip().lower().startswith(("жид", "сжиж")) else "газообразный"
        gs = [g for g in gs if g["state"] == st]
        said.append("только жидкие" if st == "жидкий" else "только газообразные")
    if grade.strip():
        if _class_of(grade, rows):
            raise AnswerError(f"«{grade.strip()}» – не марка и не сорт, а класс газов: укажи gases=\"{grade.strip()}\", "
                              "grade – только марка или сорт («5.5», «А», «первый»).")
        gs = [g for g in gs if _grade_ok(g, grade)]
        said.append(f"марка / сорт «{grade.strip()}»")
    if standard.strip():
        # «не для шаров», «жидкий» в standard – не стандарт: без этой подсказки ответом было «таких газов нет»
        if not re.search(r"\d|гост|\bту\b|iso|ост\b", standard, re.I):
            raise AnswerError(f"«{standard.strip()}» – не стандарт: standard – часть номера стандарта («ГОСТ 10157», "
                              "«ТУ 2114»); «кроме …» – name_not, «жидкий / газообразный» – state.")
        gs = [g for g in gs if _std_key(standard) in _std_key(g["standard"])]
        said.append(f"стандарт «{standard.strip()}»")
    if op == "card":
        only = [keys.key(x) for x in re.split(r"[;,]", show) if x.strip()]
        if only:
            said.append("только " + ", ".join(label_of(k) for k in only))
        if odd := [g for g in gs if g["odd"] and not only]:
            notes.append(f"У {_names(odd)} предел примеси в сотни раз больше остатка до 100 % – в базе вероятна "
                         "ошибка ввода (не та единица).")
        return _done(op, said, "\n\n".join(_card(g, ppm, only) for g in gs) or "**Таких газов в базе нет.**", gs,
                     notes)
    if op == "unfilled":
        empty = [g for g in gs if not g["filled"]]
        text = (f"**Только название, состав не заполнен – {_count(len(empty))}:**\n" + "\n".join(
            f"- {g['name']} ({g['group']})" for g in empty)) if empty else "**Газов только с названием нет.**"
        return _done(op, said, text, empty, notes)
    if op == "impurities":
        return _impurities(pick, said, notes)
    if op == "errors":
        return _errors(gs, said, notes)
    if op == "count":
        filled = [g for g in gs if g["filled"]]
        if where.strip() or has.strip() or without.strip():
            gs, keys_used, notes2 = _filter(filled, keys, ir, rows, synonyms, where, has, without, said)
            notes += notes2
            text = f"**{_count(len(gs))}**" + (f": {_names(gs, 50)}." if gs else ".")
            return _done(op, said, text, gs, notes)
        text = f"**{_count(len(gs))}**" + (f": {_names(gs, 50)}." if gs else ".")
        if len(filled) != len(gs):
            notes.append(f"Из них {_count(len(gs) - len(filled))} только с названием (состав не заполнен).")
        return _done(op, said, text, gs, notes)
    selected = len(gs)
    if empty := [g for g in gs if not g["filled"]]:
        # простой список: газы только с названием – тоже газы отбора, и сколько их всего – прямо в сноске
        if op == "list" and not (where.strip() or has.strip() or without.strip()):
            notes.append(f"Ещё {_count(len(empty))} только с названием (состав не заполнен) – всего в отборе "
                         f"{_count(selected)}: {_few(empty)}.")
        else:
            notes.append(f"Только с названием (состав не заполнен), сравнить не с чем: {_few(empty)}.")
        if len(empty) == len(gs):
            text = (f"**У {'всех названных газов' if len(gs) > 1 else 'этого газа'} в базе только название – состав "
                    f"не заполнен, ответить по составу нельзя:** {_names(gs)}.")
            return _done(op, said, text, [], notes[:-1])
    gs = [g for g in gs if g["filled"]]
    gs, used, notes2 = _filter(gs, keys, ir, rows, synonyms, where, has, without, said)
    notes += notes2
    extra = [keys.key(x) for x in re.split(r"[;,]", show) if x.strip()]
    if op == "compare":
        return _compare(gs, extra, said, notes, ppm, keys, by)
    if op in ("max", "min", "sort"):
        if not by.strip():
            raise AnswerError(f"Для op={op} нужен by: компонент, «чистота» или «сумма примесей».")
        k = keys.key(by)
        have = [g for g in gs if value(g, k) is not None and not (k == SUM and g["odd"])]
        if k == SUM and (odd := [g for g in gs if g["odd"] and g["sum"] is not None]):
            notes.append(f"Без газов с вероятной ошибкой ввода (предел примеси в сотни раз больше остатка до 100 %): "
                         f"{_names(odd)}.")
        if lost := [g for g in gs if value(g, k) is None and not (k == SUM and g["odd"])]:
            notes.append(f"{label_of(k).capitalize()} отдельно не указан(а) – в отбор не вошли: {_few(lost)}.")
        rev = op == "max" or (op == "sort" and order != "asc")
        have.sort(key=lambda g: value(g, k), reverse=rev)
        if op == "sort":
            gs = have[:top] if top > 0 else have
            said.append(f"по {answers._about(k)}, " + ("сначала наибольшие" if rev else "сначала наименьшие")
                        + (f", первые {top}" if top > 0 else ""))
        elif have:
            if top > 0:
                cut = value(have[min(top, len(have)) - 1], k)
                gs = [g for g in have if (value(g, k) >= cut if rev else value(g, k) <= cut)]
                said.append(f"{top} с {'наибольшим' if rev else 'наименьшим'} значением по {answers._about(k)}"
                            + (" (с равными на границе)" if len(gs) > top else ""))
            else:
                edge = value(have[0], k)
                gs = [g for g in have if value(g, k) == edge]
                said.append(answers.PICK_LABEL.get((k, op)) or
                            (f"больше всего {label_of(k)} (хуже всего)" if op == "max"
                             else f"меньше всего {label_of(k)} (лучше всего)")
                            + (f" – поровну у {len(gs)}" if len(gs) > 1 else ""))
        else:
            gs = []
        used.insert(0, k)
    # «водяные пары не более 10 %» у особо чистого водорода – в базе вероятна ошибка ввода: говорим об этом рядом
    if odd := [g for g in gs if g["odd"]]:
        notes.append(f"У {_names(odd)} предел примеси в сотни раз больше остатка до 100 % – в базе вероятна ошибка "
                     "ввода (не та единица).")
    used = list(dict.fromkeys(used + extra)) or [MAIN]
    head = f"**{_count(len(gs))}:**" if op in ("list", "sort") else f"**{said[-1][:1].upper() + said[-1][1:]}:**"
    lines = [head] + [_line(g, used, ppm) for g in gs] if gs else ["**Таких газов в базе нет – не подходит ни один.**"]
    return _done(op, said, "\n".join(lines), gs, notes, used)


def _filter(gs, keys, ir, rows, synonyms, where, has, without, said):
    used, notes = [], []
    for w in [x for x in re.split(r"[;,]", without) if x.strip()]:
        k = keys.key(w)
        gs = [g for g in gs if answers._absent(g, k)]
        said.append(f"без {k}")
    for w in [x for x in re.split(r"[;,]", has) if x.strip()]:
        k = keys.find(w)
        if k is None:
            said.append(f"есть «{w.strip()}»")
            notes.append(f"Компонента «{w.strip()}» в составах базы нет (есть: {', '.join(keys.singles())}).")
            return [], used, notes
        if lost := [g for g in gs if value(g, k) is None and not answers._absent(g, k) and k != g["main_name"]]:
            notes.append(f"{k} только в сумме с другими (отдельного значения нет): {_few(lost)}.")
        gs = [g for g in gs if value(g, k) is not None]
        said.append(f"есть {label_of(k)}")
        used.append(k)
    for k, op, x, shown_x in (parse_where(where, keys, ir, rows, synonyms) if where.strip() else []):
        if op == "нет":                                    # «X = 0» – как without
            gs = [g for g in gs if answers._absent(g, k)]
            said.append(f"без {k}")
            continue
        if lost := [g for g in gs if value(g, k) is None]:
            notes.append(f"{label_of(k).capitalize()} отдельно не указан(а) у {_few(lost)} – по условию не "
                         "проверить, в отбор не вошли.")
        gs = [g for g in gs if (v := value(g, k)) is not None and _test(v, op, x)]
        said.append(f"{label_of(k)} {op} {shown_x}")
        if shown_x.startswith("чем у"):              # порог – значение другого газа: какое оно, видно в куске
            notes.append(f"Порог – {label_of(k)} {shown_x.removeprefix('чем ')}.")
        used.append(k)
    return gs, used, notes


def _done(op: str, said: list[str], text: str, gs: list[dict], notes: list[str], keys: list[str] | None = None):
    understood = f"{OP_TEXT[op]}: " + "; ".join(said)
    if notes:
        text += "\n\n" + "\n".join(f"_{n}_" for n in dict.fromkeys(notes))
    rows = [{"name": g["name"], "path": g["path"], **{label_of(k): shown(g, k) for k in (keys or [])}} for g in gs]
    return {"answer": text, "understood": understood, "total": len(gs), "rows": rows}


def _compare(gs, extra, said, notes, ppm, keys, by):
    if not gs:
        return _done("compare", said, "**Сравнивать нечего: газов с составом нет.**", [], notes)
    comps = extra or list(dict.fromkeys(n for g in gs for n in g["parts"] if n != g["main_name"]))
    cols = [g["name"] for g in gs]
    lines = ["| | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1),
             "| Чистота (основной компонент) | " + " | ".join(shown(g, MAIN) for g in gs) + " |"]
    for c in comps:
        lines.append(f"| {label_of(c)} | " + " | ".join(_cell(g, c, ppm) for g in gs) + " |")
    if not extra:
        lines.append("| Сумма примесей (все «не более»), % | "
                     + " | ".join(dec_text(g["sum"]) if g["sum"] is not None else "–" for g in gs) + " |")
    said.append("по " + (", ".join(label_of(c) for c in extra) if extra else "всем примесям"))
    text = "\n".join(lines)
    if by.strip():
        k = keys.key(by)
        have = [g for g in gs if value(g, k) is not None]
        if have:
            for side, f in (("больше всего", max), ("меньше всего", min)):
                edge = f(value(g, k) for g in have)
                text += f"\n\n{side.capitalize()} {label_of(k)}: " + _names([g for g in have if value(g, k) == edge])
                text += f" ({dec_text(edge)} %)."
    return _done("compare", said, text, gs, notes, [MAIN] + comps)


def _cell(g: dict, c: str, ppm: bool) -> str:
    p = g["parts"].get(c)
    if p is None:
        return "–"
    t = p["text"]
    if ppm and p["bound"] == "не более" and p["unit"] == "%" and p["value"] is not None:
        t += f" ({dec_text(p['value'] * PPM)} ppm)"
    return t


def _impurities(pick: Pick, said: list[str], notes: list[str]):
    """Примеси групп: у каждой – все примеси её газов (не основной компонент) и у скольких газов; у двух и больше
    групп – общие (есть в каждой группе) и только у одной."""
    groups = []
    for title, gs in pick.sets:
        filled = [g for g in gs if g["filled"]]
        seen: dict[str, int] = {}
        for g in filled:
            for n in g["parts"]:
                if n != g["main_name"]:
                    seen[n] = seen.get(n, 0) + 1
        groups.append((title, filled, seen))
    lines = []
    for title, filled, seen in groups:
        lines.append(f"**{title}** ({_count(len(filled))} с составом) – примеси: "
                     + (", ".join(f"{n} (у {c})" for n, c in seen.items()) if seen else "нет") + ".")
    if len(groups) > 1:
        common = [n for n in groups[0][2] if all(n in s for _, _, s in groups[1:])]
        lines.append("\n**Общие примеси (есть в каждой группе):** " + (", ".join(common) if common else "нет") + ".")
        for title, _, seen in groups:
            only = [n for n in seen if not all(n in s for _, _, s in groups)]
            lines.append(f"Только у {title}: " + (", ".join(only) if only else "нет") + ".")
        said.append("общие примеси и только у одной группы")
    notes.append("Строки вида «O + Ar (суммарно)» – отдельная запись в базе, с отдельными O и Ar не сливаются.")
    return _done("impurities", said, "\n".join(lines), [g for _, gs, _ in groups for g in gs], notes)


def _errors(gs: list[dict], said: list[str], notes: list[str]):
    bad = [g for g in gs if g["filled"] and g["sum"] is not None and g["rest"] is not None and g["sum"] > g["rest"]]
    if not bad:
        return _done("errors", said, "**Ошибок в составе (сумма примесей больше остатка до 100 %) нет.**", [], notes)
    lines = ["**Ошибки в составе – сумма примесей больше остатка до 100 %:**"]
    for g in bad:
        top = max((p for n, p in g["parts"].items() if value(g, n) is not None), key=lambda p: p["value"])
        lines.append(f"- **{g['name']}** – {shown(g, MAIN)}, сумма примесей {dec_text(g['sum'])} %, остаток до 100 % "
                     f"– {dec_text(g['rest'])} % (больше всего – {top['name']} {top['text']})"
                     + (" – вероятна ошибка ввода (не та единица)" if g["odd"] else ""))
    return _done("errors", said, "\n".join(lines), bad, notes, [MAIN, SUM])


def _classes(rows: list[dict]):
    by: dict[tuple[str, str], list[dict]] = {}
    for g in rows:
        by.setdefault((g["section"], g["group"]), []).append(g)
    order = sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0][1]))
    most = len(order[0][1]) if order else 0
    lines = [f"**Всего газов: {len(rows)}, классов: {len(by)}.**"]
    for (sec, grp), gs in order:
        filled = sum(1 for g in gs if g["filled"])
        lines.append(f"- {grp} – {_count(len(gs))}" + (f" (с составом {filled})" if filled != len(gs) else "")
                     + (" – больше всего" if len(gs) == most else ""))
    return {"answer": "\n".join(lines), "understood": "классы газов: сколько газов в каждом, по убыванию",
            "total": len(rows), "rows": [{"group": k[1], "section": k[0], "gases": len(v)} for k, v in order]}


def _mixtures(ir: dict, gases: str):
    mixes = mixture_table(ir)
    if words := [w.lower() for w in split_items(gases)]:
        mixes = [m for m in mixes if any(all(t in m["name"].lower() for t in re.findall(r"\w+", w)) for w in words)] \
            or mixes
    lines = [f"**Газовых смесей в базе: {len(mixes)}.**"] if not words else []
    for m in mixes:
        lines.append(f"**{m['name']}** (класс «{m['group']}») – компоненты:")
        for c in m["components"]:
            lines.append(f"- {c['gas']} – доля в смеси {c['share']} %" + (f" (сам газ: {c['purity']})" if c["purity"]
                                                                          else ""))
    lines.append("\n_Общей чистоты и суммы примесей смеси в базе нет._")
    return {"answer": "\n".join(lines) if mixes else "**Газовых смесей в базе нет.**",
            "understood": "газовые смеси: из каких газов и в каких долях", "total": len(mixes),
            "rows": [{"name": m["name"], "path": m["path"],
                      "components": "; ".join(f"{c['gas']} {c['share']} %" for c in m["components"])} for m in mixes]}


# ------------------------------------------------------------------ калькулятор
REF = re.compile(r"\[\s*([^\[\]|]+?)\s*\|\s*([^\[\]|]+?)\s*\]")
EXPR = re.compile(r"(?:\s*(?:\[[^\[\]|]+\|[^\[\]|]+\]|\d+(?:[.,]\d+)?|[-+*/()]))+")


def calc(ir: dict, expr: str, unit: str = "%", synonyms: dict[str, str] | None = None) -> dict[str, Any]:
    """Выражение с числами и ссылками «[газ | что]» (+ - * / скобки) – значения берёт код из базы, считает точно
    (Decimal). «[Аргон 4.8 | азот] / [Аргон 6.0 | азот]», «17 * 23»."""
    # слабая модель оборачивает выражение в кавычки («"[Аргон 4.8 | азот] / …"») или дописывает служебный мусор
    # вызова («…"}}![]() ассистент function call…») – берётся само выражение: ссылки, числа, знаки, скобки
    expr = expr.strip().strip("\"'«»“”`").strip()
    if m := EXPR.match(expr):
        expr = m[0].strip()
    rows = gas_table(ir)
    keys = Keys(ir, rows, synonyms)
    used, vals = [], []

    def put(m: re.Match) -> str:
        found = find_elements(ir, [m[1]], (GAS,), synonyms=synonyms)
        many = [x for v in found["ambiguous"].values() for x in v]
        if found["missing"] or many or len(found["paths"]) != 1:
            names = ", ".join(f"«{split_labels(x)[-1]}»" for x in many[:8])
            raise AnswerError(f"«{m[1]}» – " + (f"подходят несколько: {names}. Назови газ точнее или спроси "
                                                 "пользователя." if many else "такого газа в базе нет."))
        g = next(r for r in rows if r["path"] == found["paths"][0])
        k = keys.key(m[2])
        v = value(g, k)
        if v is None:
            raise AnswerError(f"У «{g['name']}» {label_of(k)} отдельным числом не указан ({shown(g, k)}) – "
                              "посчитать нельзя, так и скажи.")
        used.append(f"{label_of(k)} в «{g['name']}» – {shown(g, k)}")
        named.append(f"{label_of(k)} у «{g['name']}»")
        vals.append(v)
        return f"({v})"
    named: list[str] = []
    plain = REF.sub(put, expr)
    if not re.fullmatch(r"[\d.,\s+\-*/()]+", plain):
        raise AnswerError("В выражении – только числа, + - * / скобки и ссылки [газ | что]. Пример: "
                          "'[Аргон 4.8 | азот] / [Аргон 6.0 | азот]'.")
    result = _eval(plain.replace(",", "."))
    pretty = REF.sub(lambda m: "{}", expr)
    pretty = pretty.format(*[dec_text(v) for v in vals]) if vals else expr
    ratio = "/" in expr and vals and len(vals) == 2 and re.fullmatch(r"\s*\[[^\]]+\]\s*/\s*\[[^\]]+\]\s*", expr)
    res = answers._fmt(result) if ratio else dec_text(result.quantize(Decimal("0.0000000001")).normalize()) \
        if result != result.to_integral() else dec_text(result)
    # сумма и разность значений базы – тоже в %; отношение – число раз
    percent = bool(vals) and not re.search(r"[*/]", plain)
    text = ("\n".join(f"- {u}" for u in used) + "\n\n" if used else "") + f"**{pretty} = {res}{' %' if percent else ''}**"
    if ratio and result != 0:
        text += (f" – больше в **{res}** раза." if result >= 1 else f" – меньше в **{answers._fmt(1 / result)}** раза.")
    if unit.strip().lower() == "ppm" and vals:
        text += f" (в ppm: {dec_text(result * PPM)})"
    names = iter(named)
    understood = "расчёт: " + REF.sub(lambda m: next(names), expr)
    return {"answer": text, "understood": understood, "total": len(vals), "rows": [{"value": u} for u in used],
            "result": res}


def _eval(s: str) -> Decimal:
    """Арифметика без eval: + - * / и скобки, точные Decimal."""
    toks = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", s)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom() -> Decimal:
        t = take() if peek() is not None else None
        if t == "(":
            v = expr_()
            if take() != ")":
                raise AnswerError("Скобки не закрыты.")
            return v
        if t == "-":
            return -atom()
        if t is None or not re.fullmatch(r"\d+(?:\.\d+)?", t):
            raise AnswerError("Выражение не понятно.")
        return Decimal(t)

    def term() -> Decimal:
        v = atom()
        while peek() in ("*", "/"):
            if take() == "*":
                v *= atom()
            else:
                d = atom()
                if d == 0:
                    raise AnswerError("Деление на 0.")
                v /= d
        return v

    def expr_() -> Decimal:
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if take() == "+" else v - term()
        return v
    v = expr_()
    if pos != len(toks):
        raise AnswerError("Выражение не понятно.")
    return v


__all__ = ["OPS", "run", "calc", "block_id", "data_version", "gas_table", "mixture_table", "comp_key"]
