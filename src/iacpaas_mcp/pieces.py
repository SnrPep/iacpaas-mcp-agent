"""Готовые куски ответа для инструментов платформы: текст пишет код, модель вставляет метку куска (наш агент) или
показывает его дословно (VS Code). Так и о газах, и о платформе всё, что из данных, человеку пишет код."""
from __future__ import annotations

from typing import Any

SHOW = 40                                   # сколько строк списка в куске; остальное – «и ещё N»


def _more(n: int) -> str:
    return f"\n- … и ещё {n}" if n > 0 else ""


def ir_list(r: Any) -> str:
    head = f"**Всего ИР: {r.total} – выданных {r.given} (только чтение), своих {r.own}.**"
    rows = [f"- «{i.title}» – {i.kind or 'ИР'}{', свой' if i.origin == 'свой' else ''} (`{i.path}`)"
            for i in r.items[:SHOW]]
    return "\n".join([head, *rows]) + _more(len(r.items) - SHOW)


def ir_info(r: Any) -> str:
    out = [f"**«{r.title or r.path}»** (`{r.path}`)"]
    if r.ontology:
        out.append(f"- онтология: «{r.ontology.rsplit('/', 1)[-1]}» (`{r.ontology}`)")
    if r.modified:
        out.append(f"- изменён: {r.modified}")
    if r.created:
        out.append(f"- создан: {r.created}")
    if r.top_level:
        out.append("- первый уровень: " + ", ".join(f"«{x}»" for x in r.top_level))
    return "\n".join(out)


def counts(r: Any) -> str:
    return (f"**{r.scope[:1].upper() + r.scope[1:]}: вершин {r.vertices}, дуг {r.arcs}** (дуг по ссылкам на "
            f"существующие вершины – {r.links}, терминальных вершин – {r.terminals}).")


def summary(r: Any) -> str:
    c = r.counts
    out = [f"**«{r.title or r.path}»**" + (f" – онтология «{r.ontology.rsplit('/', 1)[-1]}»" if r.ontology else ""),
           f"- всего: вершин {c.vertices}, дуг {c.arcs} (по ссылкам – {c.links}, терминальных – {c.terminals})"]
    if r.sections:
        out.append("- разделы: " + "; ".join(f"«{s['label']}» – вершин {s['vertices']}, дуг {s['arcs']}"
                                             for s in r.sections))
    if r.largest:
        out.append("- " + r.largest.replace(" Называйте оба.", ""))
    if r.groups_hint:
        out.append("- " + r.groups_hint)
    return "\n".join(out)


def matches(r: Any) -> str:
    if not r.total:
        return "**Ничего не найдено.**"
    head = f"**Найдено: {r.total}**" + (f" (показано {r.returned})" if r.returned < r.total else "") + ":"
    rows = [f"- «{m.label}»" + (f" [{m.meta}]" if m.meta else "") + f" – `{m.path}`"
            + (f" → `{m.link}`" if m.link else "") for m in r.matches[:SHOW]]
    return "\n".join([head, *rows]) + _more(min(r.returned, len(r.matches)) - SHOW)


def nodes(r: Any) -> str:
    if r.tree or not r.nodes:
        return ""
    head = f"**Вершины от «{r.root or 'корня'}», глубина {r.depth}: {r.returned}**" + (" (не все)" if r.truncated else "") + ":"
    rows = []
    for n in r.nodes[:SHOW]:
        val = f" = {n.value}" if n.value not in (None, "") else ""
        rows.append("  " * max(n.depth - 1, 0) + f"- «{n.label}»" + (f" [{n.meta}]" if n.meta else "") + val
                    + (f" → `{n.link}`" if n.link else "") + (f" (частей: {n.children})" if n.children else ""))
    if r.by_meta:
        rows.append("Прямых потомков по видам: " + ", ".join(f"«{k}» – {v}" for k, v in r.by_meta.items()))
    return "\n".join([head, *rows]) + _more(len(r.nodes) - SHOW)


def ontology(r: Any) -> str:
    head = f"**Онтология «{r.title or r.ontology_path}»**" + (f" (для «{r.for_inforesource}»)" if r.for_inforesource else "") + ":"
    rows = []
    # понятия приходят по уровням; в куске – деревом: под родителем его дети (порядок братьев – как в онтологии)
    pos = {x.path: i for i, x in enumerate(r.concepts)}
    tree = sorted(r.concepts, key=lambda x: [pos.get("/".join(x.path.split("/")[:k + 1]), -1)
                                             for k in range(x.path.count("/") + 1)])
    for x in tree[:SHOW * 2]:
        depth = x.path.count("/")
        extra = ", ".join(v for v in (x.cardinality, x.creation, x.choice, x.valtype) if v)
        rows.append("  " * depth + f"- «{x.concept}»" + (f" ({extra})" if extra else "")
                    + (f" → `{x.link}`" if x.link else ""))
    return "\n".join([head, *rows]) + (_more(len(r.concepts) - SHOW * 2) if len(r.concepts) > SHOW * 2 else "") \
        + ("\n_(онтология большая – показана не целиком)_" if r.truncated else "")


def check_report(r: Any) -> str:
    if not r.problems_total:
        return f"**Нарушений нет:** «{r.path}» соответствует ограничениям онтологии «{r.ontology}»."
    head = f"**Нарушений {r.problems_total} в {r.places} вершинах** (онтология «{r.ontology}»):"
    rows = [f"- `{p['path']}`: " + "; ".join(p["problems"]) for p in r.problems[:SHOW]]
    return "\n".join([head, *rows]) + _more(r.places - min(len(r.problems), SHOW))


def status(r: Any) -> str:
    write = {"off": "выключена", "sandbox": "в песочнице", "remote": "на платформу – только своя папка"}[r.write_tools]
    return (f"**Сервер:** {'реальная платформа' if r.backend == 'http' else 'песочница'} ({r.platform_url}); вход – "
            f"{r.auth_mode}; запись – {write}; известных ИР – {r.known_inforesources}.")


def export(r: Any) -> str:
    return f"**Выгружено «{r.path}»** (формат {r.json_type}, {r.chars} символов JSON)."
