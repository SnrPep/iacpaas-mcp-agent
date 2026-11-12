"""Чистые функции над JSON инфоресурса в универсальном формате (universal): обход, подсчёт,
поиск, плоское представление для модели, навигация, проверка структуры, упрощённое представление."""
from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from .paths import join_labels, split_labels

TYPES = ("КОРЕНЬ", "НЕТЕРМИНАЛ", "ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ")
VALTYPES = ("STRING", "INTEGER", "REAL", "BOOLEAN", "DATE", "BLOB")


def label(node: dict) -> str:
    """Метка вершины: имя, а у ТЕРМИНАЛ-ЗНАЧЕНИЕ – значение. У ссылки – последняя метка её пути."""
    if "name" in node:
        return str(node["name"])
    if "value" in node:
        v = node["value"]
        return str(v).lower() if isinstance(v, bool) else str(v)
    if "link" in node:
        tail = str(node["link"]).rstrip(";").split("$")[-1]
        parts = split_labels(tail) if tail else []
        return parts[-1] if parts else str(node["link"])
    return ""


def walk(node: dict, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Обход в глубину: (метки от корня, вершина). Ссылки не раскрываются – циклов не бывает."""
    here = path + (label(node),)
    yield here, node
    if "link" in node:
        return
    for child in node.get("successors", []) or []:
        yield from walk(child, here)


def bfs(node: dict, depth_limit: int | None = None) -> Iterator[tuple[tuple[str, ...], dict, int]]:
    """Обход в ширину с глубиной: удобно отдавать модели «верхние» уровни первыми."""
    queue: deque[tuple[tuple[str, ...], dict, int]] = deque([((label(node),), node, 0)])
    while queue:
        path, n, d = queue.popleft()
        yield path, n, d
        if "link" in n or (depth_limit is not None and d >= depth_limit):
            continue
        for child in n.get("successors", []) or []:
            queue.append((path + (label(child),), child, d + 1))


def count(ir: dict) -> dict[str, int]:
    """Вершины = собственные вершины фрагмента; дуги = все рёбра, включая ведущие по ссылкам."""
    vertices = arcs = links = terminals = 0
    for path, node in walk(ir):
        if len(path) > 1:
            arcs += 1
        if "link" in node:
            links += 1
            continue
        vertices += 1
        if node.get("type") in ("ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ"):
            terminals += 1
    return {"vertices": vertices, "arcs": arcs, "links": links, "terminals": terminals}


def sections(node: dict) -> list[dict[str, Any]]:
    """Разделы первого уровня с размерами (как count_graph с root раздела), по убыванию вершин."""
    out = [{"label": label(c), **{k: v for k, v in count(c).items() if k in ("vertices", "arcs")}}
           for c in node.get("successors", []) or [] if "link" not in c]
    return sorted(out, key=lambda s: -s["vertices"])


def largest(secs: list[dict[str, Any]]) -> str:
    """«Самый большой раздел» – два смысла: по вершинам и по дугам; готовая фраза, чтобы модель назвала оба."""
    if not secs:
        return ""
    v = max(secs, key=lambda s: s["vertices"])
    a = max(secs, key=lambda s: s["arcs"])
    if v is a:
        return f"Самый большой раздел – «{v['label']}»: и по вершинам ({v['vertices']}), и по дугам ({v['arcs']})."
    return (f"Самый большой раздел по вершинам – «{v['label']}» ({v['vertices']}), по дугам – «{a['label']}» "
            f"({a['arcs']}). Называйте оба.")


def find_node(ir: dict, concept_path: str) -> dict | None:
    """Вершина по пути меток от корня (корень не указывается). Пустой путь = корень."""
    node = ir
    for lab in split_labels(concept_path) if concept_path else []:
        nxt = next((c for c in node.get("successors", []) or [] if label(c) == lab), None)
        if nxt is None:
            return None
        node = nxt
    return node


def children_labels(node: dict, limit: int = 30) -> list[str]:
    return [label(c) for c in (node.get("successors", []) or [])[:limit]]


def _show(v: Any) -> str:
    s = str(v)
    if "e" in s.lower() and isinstance(v, float):
        s = format(Decimal(s), "f")
    return s


def tree_text(base: dict, max_depth: int = 50, max_lines: int = 300) -> str:
    """Подграф готовым деревом ├── └── со значениями: «82.0  [Числовое значение]», «→ Аргон … [Газ] (ссылка)»,
    «(пусто)» у вершины без частей. Модель переписывает его как есть."""
    lines: list[str] = []

    def text(n: dict, depth: int) -> str:
        meta, lab = n.get("meta"), label(n)
        if "link" in n:
            return f"→ {lab}" + (f"  [{meta}]" if meta else "") + "  (ссылка)"
        out = _show(n["value"]) if "value" in n else lab
        if meta and meta != out and n.get("type") != "КОРЕНЬ":
            out += f"  [{meta}]"
        kids = n.get("successors") or []
        if not kids and n.get("type") == "НЕТЕРМИНАЛ":
            out += "  (пусто)"
        elif kids and depth >= max_depth:
            out += f"  (+{len(kids)} внутри)"
        return out

    def walk_(n: dict, depth: int, pad: str) -> None:
        kids = (n.get("successors") or []) if "link" not in n and depth < max_depth else []
        for i, k in enumerate(kids):
            if len(lines) >= max_lines:
                return
            last = i == len(kids) - 1
            lines.append(pad + ("└── " if last else "├── ") + text(k, depth + 1))
            walk_(k, depth + 1, pad + ("    " if last else "│   "))

    lines.append(text(base, 0))
    walk_(base, 0, "")
    if len(lines) >= max_lines:
        lines.append("… (дерево обрезано: возьмите root глубже)")
    return "\n".join(lines)


def top_depth(base: dict, max_lines: int = 60) -> int:
    """Сколько верхних уровней большого подграфа помещается в max_lines строк дерева (не меньше 1)."""
    d = 1
    while d < 8:
        c = count(trim(base, d + 1))
        if c["vertices"] + c["links"] > max_lines or c == count(trim(base, d)):
            break
        d += 1
    return d


def ir_name(path: str) -> str:
    """Имя ИР из пути ссылки или клона: «<e-mail> / Мой Фонд / Загрузки / База …$/раздел/…;» → «База …»."""
    head = str(path).split("$", 1)[0]
    return head.replace(" / ", "/").rstrip("/").rsplit("/", 1)[-1].strip()


def external_refs(ir: dict, max_examples: int = 3) -> dict[str, dict[str, Any]]:
    """Связи ИР с другими ИР по имени ИР: links – ссылки на их вершины, clones – вершины-клоны (поле original:
    значение скопировано оттуда, например обозначение «He» или единица «%»), examples – несколько меток."""
    own = str(ir.get("title") or ir.get("name") or "")
    out: dict[str, dict[str, Any]] = {}
    for _, node in walk(ir):
        for key, kind in (("link", "links"), ("original", "clones")):
            target = ir_name(node[key]) if isinstance(node.get(key), str) else ""
            if not target or target == own:
                continue
            d = out.setdefault(target, {"links": 0, "clones": 0, "examples": []})
            d[kind] += 1
            if (ex := label(node)) not in d["examples"] and len(d["examples"]) < max_examples:
                d["examples"].append(ex)
    return out


def flatten(ir: dict, start: str = "", max_nodes: int = 400,
            max_depth: int | None = None) -> tuple[list[dict[str, Any]], bool]:
    """Плоский список вершин подграфа (BFS) – компактно и однозначно для модели.
    Путь у каждой вершины указан от корня ИР, его можно сразу передавать в другие инструменты."""
    base = find_node(ir, start) if start else ir
    if base is None:
        return [], False
    prefix = split_labels(start) if start else []
    out: list[dict[str, Any]] = []
    truncated = False
    for path, node, depth in bfs(base, max_depth):
        if len(out) >= max_nodes:
            truncated = True
            break
        rel = prefix + list(path[1:])
        item: dict[str, Any] = {"path": join_labels(rel), "label": label(node), "depth": depth + len(prefix),
                                "type": node.get("type"), "meta": node.get("meta")}
        if "valtype" in node:
            item["valtype"] = node["valtype"]
        if "value" in node:
            item["value"] = node["value"]
        if "link" in node:
            item["link"] = node["link"]
        if node.get("comment"):
            item["comment"] = node["comment"]
        succ = node.get("successors")
        if succ:
            item["children"] = len(succ)
        out.append(item)
    return out, truncated


def trim(node: dict, depth: int) -> dict:
    """Копия подграфа до глубины depth (0 – только сама вершина); исходный граф не меняется."""
    out = {k: v for k, v in node.items() if k != "successors"}
    if depth > 0 and node.get("successors") and "link" not in node:
        out["successors"] = [trim(c, depth - 1) for c in node["successors"]]
    return out


def find(ir: dict, *, text: str | None = None, meta: str | None = None, vtype: str | None = None,
         limit: int = 50) -> tuple[list[dict[str, Any]], int]:
    """Поиск вершин. Возвращает (первые limit совпадений, сколько совпадений всего)."""
    matches: list[dict[str, Any]] = []
    total = 0
    t = text.lower() if text else None
    for path, node in walk(ir):
        if t and t not in label(node).lower():
            continue
        if meta and node.get("meta") != meta:
            continue
        if vtype and node.get("type") != vtype:
            continue
        total += 1
        if len(matches) < limit:
            matches.append({"path": join_labels(list(path[1:])), "label": label(node),
                            "type": node.get("type"), "meta": node.get("meta"),
                            **({"link": node["link"]} if "link" in node else {})})
    return matches, total


# служебные слова вопроса («газ на основе гелия», «найди элемент») – в именах вершин их нет или они везде
GENERIC_STEMS = {"основе", "элемен", "понят", "вершин", "найди", "покажи"}


def word_stems(text: str) -> list[str]:
    """Основы слов для поиска по частям: первые 6 букв слов от 5 букв («ограниченности» → «ограни»)."""
    stems = [w[:6] for w in re.findall(r"\w+", text.lower()) if len(w) >= 5]
    return list(dict.fromkeys(s for s in stems if s not in GENERIC_STEMS))


def find_words(ir: dict, stems: list[str], limit: int = 50) -> tuple[list[dict[str, Any]], int]:
    """Поиск, когда фраза целиком не нашлась: вершины с наибольшим числом основ слов в метке (ссылки – нет)."""
    scored = []
    for path, node in walk(ir):
        if "link" in node:
            continue
        lab = label(node).lower()
        if k := sum(s in lab for s in stems):
            scored.append((-k, len(scored), {"path": join_labels(list(path[1:])), "label": label(node),
                                             "type": node.get("type"), "meta": node.get("meta")}))
    if not scored:
        return [], 0
    best = min(s[0] for s in scored)                   # только лучшие: со всеми найденными основами сразу
    top = [m for k, _, m in sorted(scored, key=lambda s: s[:2]) if k == best]
    return top[:limit], len(top)


def meta_stats(ir: dict, limit: int = 50) -> list[dict[str, Any]]:
    """Сколько вершин каждого метапонятия: быстрый «портрет» ИР для модели."""
    stats: dict[str, int] = {}
    for _, node in walk(ir):
        if "link" in node:
            continue
        m = str(node.get("meta"))
        stats[m] = stats.get(m, 0) + 1
    return [{"meta": k, "count": v} for k, v in sorted(stats.items(), key=lambda kv: -kv[1])[:limit]]


def to_simple(node: dict) -> Any:
    """«Упрощённый» формат (simple): метки-ключи, без служебных полей. Строится из universal и в mock, и для
    платформы: iacpaas.dvo.ru отдаёт simple/meta некорректным JSON и игнорирует start-target-concept-path."""
    succ = node.get("successors") or []
    if not succ:
        return node.get("value", label(node))
    result: dict[str, Any] = {}
    for c in succ:
        key = label(c)
        val = to_simple(c) if c.get("successors") else c.get("value", c.get("link", key))
        if key in result:
            if not isinstance(result[key], list):
                result[key] = [result[key]]
            result[key].append(val)
        else:
            result[key] = val
    return result


def to_meta(node: dict) -> dict:
    """Онтологоориентированный формат (meta): атрибуты с '$', потомки сгруппированы по метке метавершины.
    Строится из universal (см. to_simple – почему не берётся с платформы)."""
    out = {f"${k}": v for k, v in node.items() if k != "successors"}
    for child in node.get("successors", []) or []:
        out.setdefault(str(child.get("meta")), []).append(to_meta(child))
    return out


def validate(ir: Any) -> list[str]:
    """Проверка структуры ИР в universal-формате до импорта. Официальная схема:
    https://iacpaas.dvo.ru/docs/Json/target-ir.schema.json (в ней нет id/context, поэтому она строже экспорта)."""
    if not isinstance(ir, dict):
        return ["ИР должен быть JSON-объектом"]
    errors: list[str] = []
    for key in ("title", "json_type", "ontology", "type", "meta"):
        if key not in ir:
            errors.append(f"корень: нет обязательного поля '{key}'")
    if ir.get("json_type") not in (None, "universal"):
        errors.append("корень: для импорта через этот инструмент нужен json_type='universal'")
    if ir.get("type") not in (None, "КОРЕНЬ"):
        errors.append("корень: type должен быть 'КОРЕНЬ'")
    if "name" in ir and "title" in ir and ir["name"] != ir["title"]:
        errors.append("корень: name должен совпадать с title")
    if "path" in ir and not str(ir["path"]).rstrip().endswith("$;"):
        errors.append("корень: path должен заканчиваться на '$;'")
    if "ontology" in ir and not str(ir["ontology"]).rstrip().endswith("$;"):
        errors.append("корень: ontology (путь к метаинформации) должен заканчиваться на '$;'")
    for path, node in walk(ir):
        where = "/".join(path)
        if not isinstance(node, dict):
            errors.append(f"{where}: вершина должна быть объектом")
            continue
        if "link" in node:
            if not str(node["link"]).rstrip().endswith(";"):
                errors.append(f"{where}: link должен быть полным путём, оканчивающимся на ';'")
            continue
        t = node.get("type")
        if t not in TYPES:
            errors.append(f"{where}: неизвестный type {t!r}; допустимо {list(TYPES)}")
        if "meta" not in node:
            errors.append(f"{where}: нет поля meta (метка метавершины)")
        if t in ("ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ") and node.get("valtype") not in VALTYPES:
            errors.append(f"{where}: у терминала нужен valtype из {list(VALTYPES)}")
        if t == "ТЕРМИНАЛ-ЗНАЧЕНИЕ" and "value" not in node:
            errors.append(f"{where}: у ТЕРМИНАЛ-ЗНАЧЕНИЕ нет value")
        if t in ("НЕТЕРМИНАЛ", "КОРЕНЬ", "ТЕРМИНАЛ-СОРТ") and "name" not in node:
            errors.append(f"{where}: нет name")
        if t in ("ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ") and node.get("successors"):
            errors.append(f"{where}: у терминала не может быть потомков")
        if "successors" in node and not isinstance(node["successors"], list):
            errors.append(f"{where}: successors должен быть массивом")
        if len(errors) > 50:
            errors.append("… ошибок больше 50, остальные не показаны")
            break
    return errors


SPECIFIERS = {
    "copy": "копия", "copymm": "копия, может отсутствовать", "one": "ровно один", "onemm": "ноль или один",
    "list": "непустой список", "listmm": "возможно пустой список", "set": "непустое множество",
    "setmm": "возможно пустое множество", "seq": "непустая последовательность",
    "seqmm": "возможно пустая последовательность", "proxy": "пропуск",
}
RESTRICTORS = {
    "new": "новая вершина", "ref": "ссылка на существующую", "clone": "зависимый клон",
    "ref-new": "ссылка или новая", "ref-clone": "ссылка или клон", "new-clone": "новая или клон",
    "all": "любой способ", "none": "без ограничений",
}


def describe_ontology(onto: dict, max_items: int = 300, root: str = "",
                      max_depth: int | None = None) -> tuple[list[dict[str, Any]], bool]:
    """Метапонятия онтологии: путь, тип, тип значения, сколько может быть (спецификатор) и как порождается
    (ограничитель). Спецификаторы в экспорте бывают и кодами (listmm), и словами («возможно пустой список»)."""
    out: list[dict[str, Any]] = []
    base = find_node(onto, root) if root else onto
    if base is None:
        return out, False
    prefix = split_labels(root) if root else []
    for path, node, depth in bfs(base, max_depth):
        if depth == 0:
            continue
        if len(out) >= max_items:
            return out, True
        path = ("",) + tuple(prefix) + tuple(path[1:])
        spec, restr = node.get("specifier"), node.get("restrictor")
        item: dict[str, Any] = {"path": join_labels(list(path[1:])), "concept": label(node), "type": node.get("type")}
        if node.get("valtype"):
            item["valtype"] = node["valtype"]
        if spec:
            item["cardinality"] = SPECIFIERS.get(spec, spec)
        if restr:
            item["creation"] = RESTRICTORS.get(restr, restr)
        if node.get("meta") == "описать вариант альтернативы":
            item["choice"] = "вариант (из соседних вариантов выбирается один)"
        if node.get("comment"):
            item["comment"] = node["comment"]
        if "link" in node:
            item["link"] = node["link"]
        out.append(item)
    return out, False
