"""MCP-слой: инструменты, промпты и ресурс поверх шлюза к IACPaaS.
Один объект сервера запускается и через stdio, и через Streamable HTTP (см. __main__.py).

Правила, проверенные на практике:
- в модуле НЕТ `from __future__ import annotations` (иначе SDK не вычислит аннотации с локальными лимитами);
- необязательные аргументы – `str = ""`, а не `str | None` (GigaChat не принимает anyOf в схемах);
- в логах не используются зарезервированные имена LogRecord (args, msg, name...)."""

import asyncio
import collections
import difflib
import functools
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.elicitation import (AcceptedElicitation, CancelledElicitation, DeclinedElicitation,
                                    ElicitationResult)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resolve import Elicit, Resolve
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from . import answers, graph, knowledge, pieces, query
from .backend.base import BackendError, IacpaasBackend
from .config import Settings, load_synonyms
from .gateway import Gateway
from .logs import request_id
from .paths import PathError, join_labels, normalize_concept_path, normalize_ir_path, split_labels
from .policy import WritePolicy
from .registry import IrEntry, load_registry

log = logging.getLogger("iacpaas_mcp.tools")

MAX_IMPORT_CHARS = 2_000_000          # защита памяти: импорт/проверка JSON больше 2 млн символов отклоняется

READ = ToolAnnotations(read_only_hint=True, open_world_hint=True)
LOCAL = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True)

TREE_MAX_LINES = 300                  # больше вершин – дерево только верхними уровнями

IrPath = Annotated[str, Field(min_length=1, max_length=500, description=(
    "Путь ИР в «Мой Фонд»: 'Папка/Имя ИР' – из list_inforesources, не выдумывать"))]
ConceptPath = Annotated[str, Field(max_length=1000, description=(
    "Путь вершины от корня ИР через '/', например 'Моногазы/Газы на основе Аргона'; метку с '/' – в ‴…‴; "
    "пусто – корень"))]
VertexType = Literal["", "КОРЕНЬ", "НЕТЕРМИНАЛ", "ТЕРМИНАЛ-СОРТ", "ТЕРМИНАЛ-ЗНАЧЕНИЕ"]


def _joined(v: Any) -> Any:
    """Список путей от старых клиентов – той же строкой «путь 1; путь 2» (в схеме – строка)."""
    return "; ".join(str(x) for x in v) if isinstance(v, list) else v


# Газы строкой через «;»: GigaChat в списке путей передавал только первый (проба: 0 из 10, строкой – 4 из 4).
# Пути по названиям находит сервер (knowledge.find_elements); список тоже принимается.
GasList = Annotated[str, BeforeValidator(_joined)]


def resolve_gases(ir: dict, value: str, metas: tuple[str, ...], *, groups: bool = False,
                  write: bool = False) -> tuple[list[str], list[str]]:
    """Газы из строки «аргон 4.8; аргон 6.0» → настоящие пути и строки «как сопоставлено». Код не угадывает: нет
    газа – ошибка с похожими; несколько подходят – при записи ошибка (спросить человека), при чтении – все."""
    items = knowledge.split_items(value)
    r = knowledge.find_elements(ir, items, metas, groups=groups, synonyms=load_synonyms())
    short = lambda p: split_labels(p)[-1]                           # noqa: E731
    if r["missing"]:
        bits = [f"«{q}» – в базе нет" + (f" (похожие: {', '.join(f'«{short(x)}»' for x in near)})" if near else "")
                for q, near in r["missing"].items()]
        raise ToolError(f"[not_found] {'; '.join(bits)}. Названия – из gas_query(op=\"list\"); если такого газа нет – "
                        "так и скажите пользователю. Ничего не выполнено.")
    found = [f"«{x['asked']}» → «{short(x['path'])}»" for x in r["picked"] if x["how"] in ("words", "fuzzy")]
    paths = list(r["paths"])
    for q, many in r["ambiguous"].items():
        if write or len(many) > 8:
            raise ToolError(f"[ambiguous] «{q}» – в базе {len(many)} подходящих: "
                            + "; ".join(f"«{short(x)}»" for x in many[:12])
                            + ". Не выбирай сам: спроси пользователя, какой нужен (или все). Ничего не выполнено.")
        paths += [x for x in many if x not in paths]
        found.append(f"«{q}» – в базе {len(many)} подходящих: " + ", ".join(f"«{short(x)}»" for x in many)
                     + " – взяты все")
    return paths, found
def find_root(ir: dict, root: str) -> tuple[str, dict | None]:
    """Вершина root: путь как есть, иначе – название газа или смеси («Аргон 6.0» → его путь, если такой один)."""
    if not root:
        return root, ir
    if (node := graph.find_node(ir, root)) is not None:
        return root, node
    found = knowledge.find_elements(ir, [split_labels(root)[-1]], ("Газ", "Газовая смесь"), synonyms=load_synonyms())
    if len(found["paths"]) == 1 and not found["ambiguous"]:
        return found["paths"][0], graph.find_node(ir, found["paths"][0])
    return root, None


JsonType = Literal["universal", "meta", "simple"]
DeleteMode = Literal["deleteIR", "trashIR", "clearIR", "deleteSubGraph", "clearSubGraph"]


# ------------------------------------------------------------------ результаты
class Piece(BaseModel):
    """Готовый кусок ответа: текст из данных пишет код (pieces.py), модель его не переписывает."""
    block: str = Field("", description="Метка куска (её подставляет только агент IACPaaS; как показать – в hint)")
    answer: str = Field("", description="Готовый кусок ответа – его пишет код")
    hint: str = ""


class Status(Piece):
    backend: str = Field(description="mock – песочница в памяти, http – реальная платформа")
    platform_url: str
    auth_mode: str
    write_tools: Literal["off", "sandbox", "remote"]
    limits: dict[str, Any]
    known_inforesources: int
    stats: dict[str, int]


class IrItem(BaseModel):
    path: str
    title: str
    kind: str = ""
    origin: Literal["выдан", "свой", ""] = Field("", description=(
        "выдан – выданный ИР (только чтение); свой – создан агентом в своей папке"))
    description: str = ""


class IrList(Piece):
    total: int = Field(description="Сколько всего ИР подходит под фильтр – называйте это число")
    given: int = Field(0, description="Из них выданных (только чтение)")
    own: int = Field(0, description="Из них своих (созданы в своей папке)")
    returned: int
    hint: str = Field("", description="Итог одной фразой: сколько выданных и своих и где они лежат")
    items: list[IrItem]


class IrLink(BaseModel):
    source: str = Field(description="ИР, от которого связь")
    target: str = Field(description="ИР, на который связь")
    kind: Literal["онтология", "ссылки", "клоны"] = Field(description=(
        "онтология – source построен по target; ссылки – дуги на вершины target; клоны – вершины, "
        "скопированные из target (обозначения, единицы)"))
    count: int = 0
    examples: list[str] = Field(default_factory=list)
    in_set: bool = Field(True, description="false – target не входит в набор ИР (к нему нет доступа)")


class IrMap(Piece):
    given: int
    own: int
    irs: list[IrItem]
    links: list[IrLink]
    outside: list[str] = Field(default_factory=list, description="ИР вне набора, на которые есть связи")
    unreadable: list[str] = Field(default_factory=list, description="ИР, которые не удалось прочитать")
    scheme: str = Field(description="Готовая схема набора ИР и связей (как показать – в hint)")
    hint: str = ""


class IrInfo(Piece):
    path: str
    title: str | None = None
    ontology: str | None = Field(None, description="Путь к метаинформации (онтологии), по которой построен ИР")
    modified: str | None = None
    created: str | None = None
    top_level: list[str] = Field(default_factory=list, description="Метки вершин первого уровня")


class NodeItem(BaseModel):
    path: str = Field(description="Путь от корня ИР – его можно передавать в root/concept_path")
    label: str
    depth: int
    type: str | None = None
    meta: str | None = Field(None, description="Метка метавершины (тип понятия)")
    valtype: str | None = None
    value: Any = None
    link: str | None = Field(None, description="Если это ссылка – полный путь вершины, на которую она указывает")
    comment: str | None = None
    children: int = 0


class Counts(Piece):
    scope: str
    vertices: int
    arcs: int
    links: int = Field(description="Дуг, ведущих по ссылкам на уже существующие вершины")
    terminals: int


class Subgraph(Piece):
    path: str
    root: str
    depth: int
    # чисел вершин/дуг здесь нет: размер считает count_graph
    returned: int
    truncated: bool
    hint: str = ""
    by_meta: dict[str, int] = Field(default_factory=dict, description="Сколько прямых потомков root каждого "
                                    "метапонятия – готовый подсчёт элементов группы (группы-соседи, например "
                                    "«Следствия», считаются отдельно)")
    nodes: list[NodeItem]
    tree: str = Field(default="", description="Готовое дерево (при as_tree=true; как показать – в hint)")
    levels: int = Field(0, description="Большой подграф: в дереве только столько верхних уровней")


class Match(BaseModel):
    path: str
    label: str
    type: str | None = None
    meta: str | None = None
    link: str | None = None


class QueryResult(Piece):
    total: int
    returned: int
    hint: str = Field("", description="Фраза целиком не нашлась – найдено по частям слов")
    matches: list[Match]


class Summary(Piece):
    path: str
    title: str | None
    ontology: str | None
    counts: Counts
    meta_stats: list[dict[str, Any]] = Field(description="Сколько вершин каждого метапонятия")
    top_level: list[NodeItem]
    sections: list[dict[str, Any]] = Field(default_factory=list, description=(
        "Разделы первого уровня: вершины и дуги (как count_graph с root раздела), по убыванию вершин"))
    largest: str = Field("", description="Самый большой раздел – по вершинам и по дугам")
    groups: list[dict[str, Any]] = Field(default_factory=list, description=(
        "Группы второго уровня (например, классы газов): сколько в каждой элементов, по убыванию"))
    groups_hint: str = Field("", description="Итог по группам одной фразой: сколько элементов и где их больше всего")


BLOCK_HINT = ("Вставь в ответ метку {block} – код подставит готовый текст (answer) целиком. Газы и числа сам не "
              "пиши – только связку.")
# клиент без подстановки меток (VS Code, Claude Code): метку человек увидел бы как есть
TEXT_HINT = ("Покажи пользователю answer дословно и целиком (список и числа без изменений), метку block не пиши; "
             "ниже – строка «Понял вопрос так: <understood>». Газы и числа от себя не добавляй.")
SCHEME_HINT = "Вставь в ответ метку {block} – код подставит готовую схему целиком. Схему сам не переписывай."
SCHEME_TEXT_HINT = "Покажи пользователю answer дословно (готовая схема в блоке ```), метку block не пиши."
PIECE_HINT = ("Вставь в ответ метку {block} – код подставит готовый текст (answer). Сам его не переписывай, числа и "
              "названия от себя не пиши – только связку.")
PIECE_TEXT_HINT = "Покажи пользователю answer дословно, метку block не пиши; числа и названия от себя не добавляй."


class GasQueryResult(BaseModel):
    block: str = Field(description="Метка куска ответа (её подставляет только агент IACPaaS; как показать – в hint)")
    understood: str = Field(description="Как код понял запрос – строка «Понял вопрос так» для человека")
    answer: str = Field(description="Готовый кусок ответа, его считает и пишет код")
    total: int = Field(0, description="Сколько газов (значений) в куске")
    result: str = Field("", description="Итог расчёта (op=calc)")
    rows: list[dict[str, Any]] = Field(default_factory=list)
    data_version: str = Field("", description="Дата изменения ИР – версия данных")
    hint: str = ""


GAS_BASE = "Загрузки/База технологических газов"


class CheckReport(Piece):
    path: str
    ontology: str
    ontologies_used: list[str] = Field(description="Онтологии, по которым шла проверка (с теми, на которые ссылается)")
    problems_total: int
    places: int = Field(description="Сколько вершин с нарушениями")
    by_kind: dict[str, int]
    problems: list[dict[str, Any]] = Field(description="Вершина и её нарушения")
    truncated: bool = False
    hint: str = ""


class OntologyConcept(BaseModel):
    path: str
    concept: str
    type: str | None = None
    valtype: str | None = None
    cardinality: str | None = Field(None, description="Сколько может быть (спецификатор дуги)")
    creation: str | None = Field(None, description="Как порождается (ограничитель дуги)")
    choice: str | None = Field(None, description="«вариант»: одно из альтернатив – в элементе выбирается ровно один")
    comment: str | None = None
    link: str | None = None


class Ontology(Piece):
    ontology_path: str = Field(description="Путь онтологии внутри «Мой Фонд» (или полный путь, если она не в фонде)")
    for_inforesource: str | None = None
    title: str | None = None
    concepts: list[OntologyConcept]
    truncated: bool = False


class ExportResult(Piece):
    path: str
    json_type: str
    chars: int
    data: dict[str, Any]


class ValidationReport(BaseModel):
    valid: bool
    errors: list[str]
    counts: dict[str, int] = Field(default_factory=dict)


class HumanDecision(BaseModel):
    """Ответ человека в окне подтверждения клиента (MCP elicitation)."""
    approve: bool = Field(True, title="Разрешить изменение",
                          description="Нажмите «Отклонить» (или снимите отметку), чтобы ничего не менять")


class WriteResult(BaseModel):
    status: Literal["pending_confirmation", "done", "rejected", "cancelled"]
    operation: str
    target: str
    preview: dict[str, Any]
    confirm_token: str | None = None
    expires_in_s: int | None = None
    message: str


class ImportItem(BaseModel):
    path: str = Field(description="Папка в «Мой Фонд», куда импортировать (без имени ИР)")
    ir_json: dict[str, Any] = Field(description="ИР в универсальном JSON-формате")
    clear_if_exists: bool = False


class Note(BaseModel):
    path: str = Field(max_length=1000, description="Путь элемента в подборке (тот же, что в elements)")
    text: str = Field(min_length=1, max_length=1000, description="Пометка, например «нет состава и стандарта»")


class MixPart(BaseModel):
    gas: str = Field(min_length=1, max_length=1000, description="Путь газа от корня базы, например "
                                                                "'Моногазы/Газы на основе Аргона/Аргон газообразный высший сорт'")
    percent: float = Field(gt=0, le=100, description="Доля в смеси, %")


class ExportItem(BaseModel):
    path: str = Field(description="Путь ИР для экспорта после работы сервиса")
    json_type: JsonType = "universal"
    root: str = ""
    depth: int = Field(-1, description="-1 – весь подграф")


# ------------------------------------------------------------------ сборка
def create_backend(settings: Settings) -> IacpaasBackend:
    if settings.backend == "http":
        from .backend.http import HttpBackend
        return HttpBackend(settings.base_url, api_key=settings.api_key, jwt=settings.jwt,
                           username=settings.username, password=settings.password,
                           timeout_s=settings.timeout_s, write_enabled=settings.allow_write and settings.remote_write)
    if settings.backend == "mock":
        from .backend.mock import MockBackend
        return MockBackend(settings.fixtures_dir)
    raise ValueError(f"IACPAAS_BACKEND должен быть mock или http, а не {settings.backend!r}")


def _drop_empty(v: Any, drop_label: bool = True) -> Any:
    if isinstance(v, dict):
        d = {k: _drop_empty(x, drop_label) for k, x in v.items() if x is not None and x != ""}
        if drop_label and isinstance(d.get("path"), str) and d.get("label") == d["path"].rsplit("/", 1)[-1]:
            del d["label"]                       # метка = последняя часть пути, дубль не нужен
        return d
    if isinstance(v, list):
        return [_drop_empty(x, drop_label) for x in v]
    return v


def short_ir(path: str | None) -> str | None:
    """Путь ИР для ответа модели: без «<e-mail> / Мой Фонд /» и «$;» – короче и без персональных данных."""
    if not path:
        return path
    try:
        return normalize_ir_path(path)
    except PathError:
        return path


def compact(fn):
    """SDK пишет в ответ JSON с отступами и всеми null. LangChain MCPAdapter отдаёт модели текстовую часть, а клиенты
    вроде Claude Code – structuredContent. Текст – компактный JSON без пустых полей и без дубля label;
    structuredContent – без полей со значением по умолчанию (exclude_defaults: ответ остаётся валидным по схеме)."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        res = await fn(*args, **kwargs)
        if isinstance(res, BaseModel):
            data = res.model_dump(mode="json", by_alias=True)
            structured = res.model_dump(mode="json", by_alias=True, exclude_defaults=True)
        elif isinstance(res, list) and all(isinstance(x, BaseModel) for x in res):
            data = [x.model_dump(mode="json", by_alias=True) for x in res]
            structured = {"result": [x.model_dump(mode="json", by_alias=True, exclude_defaults=True) for x in res]}
        else:
            return res
        text = json.dumps(_drop_empty(data), ensure_ascii=False, separators=(",", ":"))
        return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)
    return wrapper


def _strip_titles(schema: Any, in_props: bool = False, keep: bool = False) -> Any:
    """Без «title» у параметров («Path», «Root»): модели они не нужны, а схемы уходят ей на каждом шаге.
    У параметра с именем title заголовок остаётся: без него LangChain выбрасывает сам параметр."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v, k == "properties", in_props and k == "title") for k, v in schema.items()
                if keep or not (k == "title" and isinstance(v, str))}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


NO_DIALOG = ("[confirmation] Этот клиент не умеет показывать окно подтверждения (MCP elicitation), поэтому запись здесь "
             "выключена: иначе модель могла бы подтвердить изменение сама. Ничего не изменено. Сделайте это в веб-чате "
             "или консольном агенте IACPaaS (там подтверждает человек) или в клиенте с окнами подтверждения.")


def preview_text(preview: dict) -> str:
    """Предпросмотр для окна подтверждения: строки «поле: значение», списки – через «; », не длиннее 1500 знаков."""
    def val(v: Any) -> str:
        if isinstance(v, list):
            return f"{len(v)}: " + "; ".join(map(str, v)) if len(v) > 1 else "; ".join(map(str, v))
        if isinstance(v, dict):
            return "; ".join(f"{k} – {x}" for k, x in v.items())
        return {True: "да", False: "нет"}.get(v, str(v)) if isinstance(v, bool) else str(v)
    text = "\n".join(f"{k}: {val(v)}" for k, v in preview.items() if v not in ("", None, [], {}))
    return text if len(text) <= 1500 else text[:1500] + " …"


NAME_OK = re.compile(r"[А-Яа-яЁёA-Za-z0-9 .\-]+")
NAME_RULE = "только буквы, цифры, пробел, точка и дефис"


def check_name(title: str) -> str:
    """Имя нового ИР – по простому правилу: со знаками вроде «%», «+», «&», «"» платформа и её сайт работают
    с ошибками. Правило есть и в инструкции агента; здесь – вторая проверка."""
    title = title.strip()
    if not NAME_OK.fullmatch(title):
        # готовое имя по правилу: модель трижды придумывала новое имя с тем же «%» или «≥»
        fixed = " ".join(re.sub(r"[^А-Яа-яЁёA-Za-z0-9 .\-]", " ", title).split()).strip(" .-")
        raise ToolError(f"[invalid_request] Имя ИР «{title}» не подходит: {NAME_RULE}. "
                        + (f"Возьмите «{fixed}». " if fixed else
                           "Придумайте имя по этому правилу, например «Газы высокой чистоты не менее 99.999». ")
                        + "Ничего не создано.")
    return title


def size_text(c: dict[str, int]) -> str:
    """Размер для предпросмотра человеку: «249 вершин, 248 дуг»."""
    return f"{c.get('vertices', 0)} вершин, {c.get('arcs', 0)} дуг"


def write_mode(settings: Settings, backend: IacpaasBackend) -> str:
    if not settings.allow_write:
        return "off"
    if backend.name == "mock":
        return "sandbox"
    # на реальную платформу – только при заданных папках для записи: без них запись туда не включается вовсе
    return "remote" if settings.remote_write and backend.writable and settings.write_paths else "off"


def build_server(settings: Settings, backend: IacpaasBackend, **auth: Any) -> MCPServer:
    gw = Gateway(backend, cache_ttl_s=settings.cache_ttl_s, retries=settings.retries,
                 max_concurrency=settings.max_concurrency)
    sandbox = list(getattr(backend, "store", {}).keys())
    base: list[IrEntry] = load_registry(settings.registry_file, settings.known_irs, sandbox)
    own_file = settings.own_registry_file if backend.name == "http" else None
    own_mem: list[IrEntry] = []                         # песочница: свои ИР – только в памяти

    def read_own() -> list[IrEntry]:
        """Свои ИР (API платформы фонд не перечисляет) – из файла при каждом обращении. Файл общий у всех
        процессов сервера (веб-чат, VS Code): копия в памяти расходилась, и удалённое в одном окне другое
        записывало обратно."""
        if not own_file:
            return list(own_mem)
        try:
            data = json.loads(own_file.read_text(encoding="utf-8")) if own_file.is_file() else []
            return [IrEntry(path=i["path"], title=i.get("title", ""), description=i.get("description", ""),
                            kind=i.get("kind", "")) for i in data if i.get("path")]
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            log.warning("own_registry_unreadable", extra={"file": str(own_file)})
            return []

    def write_own(items: list[IrEntry]) -> None:
        if not own_file:
            own_mem[:] = items
            return
        own_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = own_file.with_suffix(".tmp")                # целиком или никак: файл читают другие процессы
        tmp.write_text(json.dumps([e.__dict__ for e in items], ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(own_file)

    def entries() -> list[IrEntry]:
        return base + [e for e in read_own() if all(b.path != e.path for b in base)]

    def remember_own(path: str, title: str, description: str) -> None:
        """Созданный агентом ИР – в список; меняется только эта запись."""
        write_own([e for e in read_own() if e.path != path] + [IrEntry(
            path=path, title=title, description=description, kind="своя подборка (создана агентом)")])

    def forget_own(path: str) -> bool:
        own = read_own()
        if all(e.path != path for e in own):
            return False
        write_own([e for e in own if e.path != path])
        return True

    async def drop_deleted_own() -> None:
        """Свои ИР, удалённые на сайте или в другом окне, – убрать из списка. API папку не перечисляет, поэтому
        каждый свой ИР проверяется на платформе (лёгкий экспорт корня, мимо кэша). Пользователь удалил подборки на
        сайте, а модель ещё называла их «уже есть в папке»."""
        async def gone(e: IrEntry) -> bool:
            try:
                await backend.export(e.path, "universal", None, 0)
                return False
            except BackendError as err:
                return err.kind == "not_found"          # сеть и прочие сбои – запись остаётся
        own = read_own()
        for e, dead in zip(own, await asyncio.gather(*(gone(e) for e in own))):
            if dead:
                forget_own(e.path)
                gw.invalidate(e.path)
                log.info("own_ir_gone", extra={"target": e.path})

    def ir_items(items: list[IrEntry]) -> list[IrItem]:
        """С пометкой origin: выданный (только чтение) или свой – из списка своих или из папки записи."""
        own = {e.path for e in read_own()}
        return [IrItem(path=e.path, title=e.title, kind=e.kind, description=e.description,
                       origin="свой" if e.path in own or any(e.path.startswith(w.rstrip("/") + "/")
                                                             for w in settings.write_paths) else "выдан")
                for e in items]

    def origin_hint(items: list[IrItem]) -> str:
        """«Всего 11: выданных 9 (…) и своих 2 в папке «Песочница ИИ»…» – на сайте видно 9 ИР и папку, модель
        называла 11 без пояснения и на «почему не 9» отвечала общими словами."""
        given = [i for i in items if i.origin == "выдан"]
        own = [i for i in items if i.origin == "свой"]
        kinds = collections.Counter(i.kind.split(" (")[0] or "ИР" for i in given)
        folders = ", ".join(f"«{f}»" for f in sorted({i.path.rsplit("/", 1)[0] for i in given if "/" in i.path}))
        own_dirs = ", ".join(f"«{d}»" for d in sorted({i.path.rsplit("/", 1)[0].rsplit("/", 1)[-1] for i in own}))
        if not given:
            return (f"Своих ИР: {len(own)}" + (f" в папке {own_dirs} (созданы агентом)" if own else "")
                    + ". Список сверен с платформой: удалённые на сайте уже убраны.")
        text = (f"Всего {len(items)}: выданных {len(given)} (только чтение: "
                + ", ".join(f"{k} – {n}" for k, n in kinds.items()) + ")"
                + f" и своих {len(own)}" + (f" в папке {own_dirs} (созданы агентом)" if own else "") + ".")
        if given and own:
            text += (f" На сайте в {folders} видно {len(given)} выданных ИР и папку {own_dirs}: свои ИР лежат "
                     "внутри неё, поэтому всего больше. Называйте оба числа раздельно.")
        return text
    mode = write_mode(settings, backend)
    policy = WritePolicy(enabled=mode != "off", ttl_s=settings.confirm_ttl_s)
    max_depth, max_nodes = settings.max_depth, settings.max_nodes

    where = {"off": ("Изменяющих инструментов нет: только чтение. Просят изменить (переименовать, удалить, добавить) – "
                     "найдите этот элемент (query_graph), покажите его путь и скажите, что запись на этом сервере "
                     "выключена."),
             "sandbox": "Изменяющие инструменты работают в ПЕСОЧНИЦЕ (локальная копия), реальная платформа не меняется.",
             "remote": ("Изменяющие инструменты меняют данные на РЕАЛЬНОЙ платформе – только внутри папок "
                        + ", ".join(f"«{w}»" for w in settings.write_paths) + "; всё остальное только читается. "
                        "В свою папку: подборка элементов (save_selection, пометки – notes), черновик смеси "
                        "(save_mixture), удаление своего ИР (delete_from_fund). Просят изменить выданную базу – "
                        "найдите элемент и предложите сохранить копию с пометкой в свою папку.")}[mode]
    known = "; ".join(f"«{e.title}» ({e.path}{': ' + e.description[:120] if e.description else ''})"
                      for e in entries()[:8]) or "см. list_inforesources"
    mcp = MCPServer(
        "iacpaas",
        title="IACPaaS",
        version="1.0.0",
        instructions=(
            "Доступ к инфоресурсам (ИР) платформы IACPaaS. ИР – граф: вершины-понятия и дуги-отношения; "
            "у каждой вершины есть метавершина (поле meta) из онтологии ИР.\n"
            f"Подключённые ИР: {known}. Вопросы пользователя о том, что есть в этих базах (их предметная "
            "область, разделы, элементы, числа, значения), решайте через инструменты этого сервера и отвечайте "
            "по их результатам, а не по общим знаниям: в базе может быть иначе. Сначала найдите нужное "
            "(get_inforesource – разделы, query_graph – поиск по словам), затем читайте (get_subgraph). "
            "Названия, значения, обозначения и стандарты приводите из базы как есть и не дополняйте своими; "
            "незаполненное в базе так и называйте. Запрашивайте ровно нужное: узкий root, небольшие depth и limit.\n"
            "Порядок работы: 1) list_inforesources – найти путь ИР (не выдумывайте пути; если не нашли – "
            "спросите пользователя); 2) get_inforesource или summarize_inforesource – обзор; "
            "3) get_subgraph с небольшой depth и уточнение через root; count_graph – числа; query_graph – поиск.\n"
            "Любой вопрос о газах базы (какие есть, состав, отбор по условиям, самые / хуже / лучше, сортировка, "
            "сравнение, общие примеси групп, ошибки состава, смеси, расчёт «во сколько раз / на сколько / сумма») – "
            "один инструмент gas_query: код считает и пишет готовый кусок ответа (answer) и строку understood (как "
            "понят запрос); перескажите кусок как есть, чисел и газов от себя не добавляйте. "
            "«что заполнено не так / чего не хватает» – check_inforesource (проверка по ограничениям онтологии).\n"
            "«Сколько / какие ИР» – list_inforesources: выданные и свои называйте раздельно (готовый итог в hint). "
            "«Нарисуй / покажи структуру»: набора ИР и связей между ними – map_inforesources (готовая схема); "
            "одной базы – верхние уровни (summarize_inforesource или get_subgraph с as_tree), конкретная группа – "
            "as_tree с root; онтологии – get_ontology. Неясно, какая структура нужна, – спросите с вариантами.\n"
            "У инструментов чтения тоже есть готовый кусок answer (список ИР, вершины и дуги, схема, дерево, "
            "онтология, проверка) – его пишет код: показывайте, как сказано в hint, сами не переписывайте.\n"
            "Ошибка инструмента – подсказка: прочитайте её, исправьте аргумент, не повторяйте тот же вызов.\n"
            f"{where} Запись всегда в два шага: вызов без confirm_token даёт предпросмотр и токен; "
            "покажите предпросмотр пользователю и только после явного согласия повторите вызов с теми же "
            "аргументами и confirm_token."),
        **auth,
    )

    # ---------------------------------------------------------------- utils
    def ir_path(p: str) -> str:
        try:
            q = normalize_ir_path(p)
        except PathError as e:
            raise ToolError(f"[invalid_path] {e}") from e
        known = [e.path for e in entries()]
        if not known or q in known:
            return q
        # неполное имя ИР («Загрузки/База газов»): все его слова целиком – в имени ровно одного известного ИР;
        # слово с опечаткой («наблюдени») не совпадёт – тогда ошибка с подсказкой, как раньше
        def words(s: str) -> set[str]:
            return {w for w in re.findall(r"\w+", s.rsplit("/", 1)[-1].lower()) if len(w) >= 3}
        want = words(q)
        hits = [k for k in known if want and want <= words(k)]
        if len(hits) == 1:
            log.info("ir_path_resolved", extra={"given": q, "path": hits[0]})
            return hits[0]
        return q

    def concept(p: str) -> str:
        try:
            return normalize_concept_path(p)
        except PathError as e:
            raise ToolError(f"[invalid_path] {e}") from e

    email = (settings.user_email or "").strip()
    own_fund = re.compile(re.escape(email) + r"\s*[,/]\s*Мой Фонд\s*[,/]\s*") if email else None

    def scrub(text: str) -> str:
        """Сообщения платформы содержат путь с e-mail владельца – модели и пользователю он не нужен."""
        if not email:
            return text
        return own_fund.sub("", text).replace(email, "<пользователь>")

    def groups_phrase(groups: list[dict[str, Any]]) -> str:
        """«Моногазы: 48 элементов «Газ» в 12 группах; … Больше всего: …» – готовый ответ на «сколько и где больше»."""
        if not groups:
            return ""
        per: dict[str, list[dict[str, Any]]] = {}
        for g in groups:
            per.setdefault(g["section"], []).append(g)
        parts = [f"«{sec}»: {sum(g['elements'] for g in gs)} элем. «{gs[0]['element_meta']}» в "
                 f"{len(gs)} {'группе' if len(gs) % 10 == 1 and len(gs) % 100 != 11 else 'группах'}."
                 for sec, gs in per.items()]
        top = sorted(groups, key=lambda g: -g["elements"])[:3]
        return " ".join(parts) + " Больше всего: " + ", ".join(f"«{g['group']}» – {g['elements']}" for g in top) + "."

    async def ontologies_for(p: str) -> tuple[str, dict[str, dict]]:
        """Онтология ИР и онтологии, на которые она ссылается (например, «…характеристик и их значений»)."""
        head = await gw.export(p, depth=0)
        if not head.get("ontology"):
            raise ToolError(f"[invalid_request] У ИР '{p}' не указана онтология.")
        onto = await gw.export(ir_path(head["ontology"]))
        title = str(onto.get("title") or "")
        found, queue = {title: onto}, knowledge.linked_ontologies(onto)
        while queue:
            lp = queue.pop(0)
            try:
                o = await gw.export(ir_path(lp))
            except (BackendError, ToolError):
                continue                                  # нет доступа – проверка этой части пропускается
            t = str(o.get("title") or "")
            if t and t not in found:
                found[t] = o
                queue += knowledge.linked_ontologies(o)
        return title, found

    VERTEX_MISSING = re.compile(r"начальное понятие|^Вершина ", re.IGNORECASE)

    def suggest(path: str) -> str:
        names = [e.path for e in entries()]
        close = difflib.get_close_matches(path, names, n=3, cutoff=0.4)
        if not close:
            low = path.lower().split("/")[-1]
            close = [n for n in names if low and low in n.lower()][:3]
        return f" Похожие ИР: {close}." if close else " Вызовите list_inforesources, чтобы увидеть доступные ИР."

    @asynccontextmanager
    async def traced(tool: str, ctx: Context, args: dict, path: str | None = None):
        token = request_id.set(str(ctx.request_id))
        start = time.perf_counter()
        log.info("tool_call", extra={"tool": tool, "tool_args": args})
        try:
            yield
            log.info("tool_ok", extra={"tool": tool, "ms": round((time.perf_counter() - start) * 1000)})
        except BackendError as e:
            folder = (path or "").strip("/")
            if e.kind == "not_found" and path and forget_own(folder):
                # свой ИР удалён в другом окне (веб-чат, VS Code, сайт) – убираем из списка, а не пугаем ошибкой
                raise ToolError(f"[not_found] ИР «{folder}» на платформе уже нет (удалён раньше, например в другом "
                                "окне) – убран из списка своих ИР. Делать с ним нечего.") from e
            inside = [x.path for x in entries() if folder and x.path.startswith(folder + "/")]
            if inside or folder in settings.write_paths:
                log.warning("tool_error", extra={"tool": tool, "kind": "folder", "error": str(e)})
                names = ", ".join(f"«{x}»" for x in inside[:12]) or "пока нет"
                raise ToolError(f"[invalid_path] «{folder}» – это папка, а не ИР: читать можно только ИР. ИР в ней: "
                                f"{names}. Вызовите инструмент с путём ИР (если нужно несколько – по очереди).") from e
            hint = ""
            if e.kind == "not_found" and VERTEX_MISSING.search(str(e)):
                hint = (" Такой вершины нет: проверьте путь вершины (метки от корня через '/'); существующие "
                        "пути возьмите из get_inforesource (top_level) или get_subgraph.")
            elif e.kind == "not_found" and path:
                hint = suggest(path)
            elif e.retryable:
                hint = " Ошибка временная – можно повторить позже."
            elif e.kind == "auth":
                hint = " Это не исправить аргументами: сообщите пользователю."
            log.warning("tool_error", extra={"tool": tool, "kind": e.kind, "error": str(e),
                                             "ms": round((time.perf_counter() - start) * 1000)})
            raise ToolError(f"[{e.kind}] {scrub(str(e))}.{hint}") from e
        except ToolError as e:
            log.warning("tool_error", extra={"tool": tool, "kind": "invalid_request", "error": str(e)})
            raise
        finally:
            request_id.reset(token)

    def pieced(render):
        """Результат инструмента + готовый кусок ответа (render пишет текст из данных) и подсказка, как его показать:
        метка – нашему агенту (pieces=marks), дословно – клиентам без подстановки меток."""
        def deco(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                res = await fn(*args, **kwargs)
                if isinstance(res, Piece) and not res.answer and (text := render(res)):
                    res.block = query.block_id(fn.__name__, {k: v for k, v in kwargs.items() if k != "ctx"})
                    res.answer = text
                    how = PIECE_HINT.format(block=res.block) if settings.pieces == "marks" else PIECE_TEXT_HINT
                    res.hint = f"{res.hint} {how}".strip()
                return res
            return wrapper
        return deco

    # ================================================================ READ
    @mcp.tool(title="Состояние сервера", annotations=LOCAL)
    @compact
    @pieced(pieces.status)
    async def iacpaas_status(ctx: Context) -> Status:
        """Режим работы сервера: песочница или реальная платформа, способ входа, включена ли запись, лимиты."""
        async with traced("iacpaas_status", ctx, {}):
            return Status(backend=backend.name, platform_url=settings.base_url, auth_mode=settings.auth_mode,
                          write_tools=mode, known_inforesources=len(entries()), stats=dict(gw.stats),
                          limits={"max_depth": max_depth, "max_nodes": max_nodes,
                                  "max_result_chars": settings.max_result_chars,
                                  "cache_ttl_s": settings.cache_ttl_s, "platform_timeout_s": settings.timeout_s})

    @mcp.tool(title="Найти инфоресурсы", annotations=LOCAL)

    @compact
    @pieced(pieces.ir_list)
    async def list_inforesources(
        ctx: Context,
        filter: Annotated[str, Field(max_length=200, description="Подстрока в пути, названии или описании; пусто – все")] = "",
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> IrList:
        """Известные ИР пользователя с путями и их общее число (total). Начинайте работу отсюда: путь ИР для
        остальных инструментов берите из этого списка. Сюда же попадают ИР, созданные агентом в своей папке:
        origin «выдан» (только чтение) или «свой»; given и own – сколько каких, hint – готовый итог.
        (REST API платформы не перечисляет фонд – список задан в конфигурации.)"""
        async with traced("list_inforesources", ctx, {"filter": filter, "limit": limit}):
            await drop_deleted_own()
            f = filter.lower().strip()
            items = ir_items([e for e in entries()
                              if not f or f in f"{e.path} {e.title} {e.description} {e.kind}".lower()])
            return IrList(total=len(items), given=sum(i.origin == "выдан" for i in items),
                          own=sum(i.origin == "свой" for i in items), returned=min(len(items), limit),
                          hint=origin_hint(items), items=items[:limit])

    @mcp.tool(title="Схема ИР и связей", annotations=READ)

    @compact
    async def map_inforesources(
        ctx: Context,
        path: Annotated[str, Field(max_length=500, description=(
            "Путь ИР – только его связи (от него и к нему); пусто – весь набор"))] = "",
    ) -> IrMap:
        """Схема набора ИР: какие ИР выданы и какие свои, по какой онтологии построен каждый, и связи между ИР –
        ссылки на вершины другого ИР и клоны (значения, скопированные из другого ИР: химические обозначения,
        единицы). Для «нарисуй / покажи структуру» набора ИР, «как связаны ИР», «на чём построена база».
        scheme – готовая схема (как показать – в hint)."""
        p = ir_path(path) if path.strip() else ""
        async with traced("map_inforesources", ctx, {"path": p}, path=p or None):
            await drop_deleted_own()
            items = ir_items(entries())
            by_name = {i.title: i for i in items}
            by_path = {i.path: i for i in items}
            if p and p not in by_path:
                raise ToolError(f"[not_found] ИР «{p}» нет в списке.{suggest(p)}")

            async def load(i: IrItem):
                try:
                    return i, await gw.export(i.path)
                except BackendError as e:
                    log.warning("map_unreadable", extra={"tool": "map_inforesources", "error": str(e)})
                    return i, None
            loaded = await asyncio.gather(*(load(i) for i in items))
            links: list[IrLink] = []
            unreadable: list[str] = []
            for i, ir in loaded:
                if ir is None:
                    unreadable.append(i.title)
                    continue
                onto = short_ir(ir.get("ontology")) or ""
                if onto and ir.get("meta") != "Язык ИРУО":       # у онтологий – язык платформы, не ИР набора
                    name = by_path[onto].title if onto in by_path else onto.rsplit("/", 1)[-1]
                    links.append(IrLink(source=i.title, target=name, kind="онтология", count=1,
                                        in_set=name in by_name))
                for name, ref in graph.external_refs(ir).items():
                    for kind, key in (("ссылки", "links"), ("клоны", "clones")):
                        if ref[key]:
                            links.append(IrLink(source=i.title, target=name, kind=kind, count=ref[key],
                                                examples=ref["examples"], in_set=name in by_name))
            if p:
                me = by_path[p].title
                links = [x for x in links if me in (x.source, x.target)]
                items = [i for i in items if i.title == me or any(i.title in (x.source, x.target) for x in links)]
            outside = sorted({x.target for x in links if not x.in_set})
            given = [i for i in items if i.origin == "выдан"]
            own = [i for i in items if i.origin == "свой"]
            scheme = map_scheme(given, own, links, f"Связи ИР «{by_path[p].title}»" if p else "")
            block, answer, how = scheme_piece("map_inforesources", {"path": p}, scheme)
            return IrMap(given=len(given), own=len(own), irs=items, links=links, outside=outside,
                         unreadable=unreadable, hint=(
                             "онтология – по чему построен ИР; ссылки – дуги на вершины другого ИР; клоны – значения, "
                             "скопированные из другого ИР. Свои ИР – копии выданных на той же онтологии."
                             + (f" Вне набора (доступа нет): {', '.join(outside)}." if outside else "") + " " + how),
                         scheme=scheme, block=block, answer=answer)

    def scheme_piece(tool: str, args: dict, text: str) -> tuple[str, str, str]:
        """Схема или дерево готовым куском: метка, кусок в блоке ``` и подсказка, как его показать (метка – нашему
        агенту, дословно – клиентам без подстановки меток)."""
        block = query.block_id(tool, args)
        how = SCHEME_HINT.format(block=block) if settings.pieces == "marks" else SCHEME_TEXT_HINT
        return block, "```\n" + text + "\n```", how

    def map_scheme(given: list[IrItem], own: list[IrItem], links: list[IrLink], title: str = "") -> str:
        """Набор ИР деревом: у каждого ИР – онтология, ссылки и клоны с числом и примерами."""
        lines = [title or f"Набор ИР: выданных {len(given)} (только чтение), своих {len(own)}"]

        def node(i: IrItem, pad: str, last: bool) -> None:
            lines.append(pad + ("└── " if last else "├── ") + f"{i.title}  [{i.kind.split(' (')[0] or 'ИР'}]")
            sub = pad + ("    " if last else "│   ")
            mine = [x for x in links if x.source == i.title]
            for n, x in enumerate(mine):
                more = "" if x.in_set else "  (вне набора)"
                text = (f"онтология → {x.target}" if x.kind == "онтология" else
                        f"{x.kind} → {x.target}: {x.count}" + (f" ({', '.join(x.examples)})" if x.examples else ""))
                lines.append(sub + ("└── " if n == len(mine) - 1 else "├── ") + text + more)
        for n, i in enumerate(given):
            node(i, "", n == len(given) - 1 and not own)
        if own:
            folder = own[0].path.rsplit("/", 1)[0].rsplit("/", 1)[-1]
            lines.append(f"└── {folder} – свои ИР")
            for n, i in enumerate(own):
                node(i, "    ", n == len(own) - 1)
        return "\n".join(lines)

    @mcp.tool(title="Метаданные ИР", annotations=READ)

    @compact
    @pieced(pieces.ir_info)
    async def get_inforesource(path: IrPath, ctx: Context) -> IrInfo:
        """Название, онтология (метаинформация), даты и вершины первого уровня. Дёшево: грузится 1 уровень."""
        p = ir_path(path)
        async with traced("get_inforesource", ctx, {"path": p}, path=p):
            ir = await gw.export(p, depth=1)
            return IrInfo(path=p, title=ir.get("title"), ontology=short_ir(ir.get("ontology")), modified=ir.get("date"),
                          created=ir.get("creation"), top_level=graph.children_labels(ir, limit=100))

    @mcp.tool(title="Подграф ИР", annotations=READ)

    @compact
    @pieced(pieces.nodes)
    async def get_subgraph(
        path: IrPath,
        ctx: Context,
        root: ConceptPath = "",
        depth: Annotated[int, Field(ge=0, le=max_depth, description="Глубина от root: 0 – только сама вершина. "
                                                                    "Начинайте с 1–2")] = 2,
        max_nodes: Annotated[int, Field(ge=1, le=max_nodes, description="Сколько вершин вернуть максимум")] = 100,
        as_tree: Annotated[bool, Field(description=(
            "true – вместо списка готовое дерево со значениями, всегда целиком (depth не нужен); для «нарисуй», "
            "«покажи дерево»"))] = False,
    ) -> Subgraph:
        """Фрагмент графа от вершины root глубиной depth: плоский список вершин с путями (метка вершины –
        последняя часть path), типами, метапонятиями и числом потомков (children). as_tree=true – готовое
        дерево со значениями, ссылками «→» и пометками «(пусто)» готовым куском (как показать – в hint). Если вершин больше
        max_nodes – ответ обрезается (truncated) с подсказкой. Сколько элементов в группе (например, газов в
        классе) – root=группа, depth=1: by_meta – сколько прямых потомков каждого метапонятия. Чисел
        вершин/дуг в ответе нет – их считает count_graph."""
        p, r = ir_path(path), concept(root)
        async with traced("get_subgraph", ctx, {"path": p, "root": r, "depth": depth, "max_nodes": max_nodes}, path=p):
            # полный экспорт (кэшируется), глубина режется здесь: во фрагменте платформа разворачивает ссылки
            # в копии чужих вершин, а на границе export-depth у вершин не видно, что у них есть потомки
            ir = await gw.export(p)
            r, base = find_root(ir, r)
            if base is None:
                raise ToolError(f"[not_found] Такой вершины нет: '{r}' в ИР '{p}'. Существующие пути возьмите "
                                f"из get_inforesource (top_level) или get_subgraph с меньшей глубиной.")
            c = graph.count(graph.trim(base, depth))
            nodes, truncated = graph.flatten(ir, start=r, max_nodes=max_nodes, max_depth=depth)
            hint = (f"Показано {len(nodes)} из {c['vertices'] + c['links']} элементов. Уменьшите depth или "
                    f"возьмите root глубже (пути вершин есть в ответе).") if truncated else ""
            by_meta = dict(collections.Counter(ch.get("meta") or "–" for ch in base.get("successors") or [])) \
                if depth >= 1 else {}
            if as_tree:
                whole = graph.count(base)
                if whole["vertices"] + whole["links"] <= TREE_MAX_LINES:
                    tree = graph.tree_text(base)
                    block, answer, how = scheme_piece("get_subgraph", {"path": p, "root": r, "tree": True}, tree)
                    return Subgraph(path=p, root=r, depth=depth, returned=0, truncated=False, by_meta=by_meta,
                                    nodes=[], tree=tree, block=block, answer=answer, hint=how)
                # вся база (1478 вершин) резалась на 300 строках, и модель переписывала один газ и «…»
                lv = graph.top_depth(base)
                tree = graph.tree_text(base, max_depth=lv)
                block, answer, how = scheme_piece("get_subgraph", {"path": p, "root": r, "tree": lv}, tree)
                return Subgraph(path=p, root=r, depth=lv, returned=0, truncated=True, by_meta=by_meta, nodes=[],
                                levels=lv, tree=tree, block=block, answer=answer, hint=(
                                    f"Подграф большой ({whole['vertices']} вершин) – целиком не рисуется: в дереве "
                                    f"верхние уровни ({lv}), «(+N внутри)» – сколько частей у вершины. Скажите это "
                                    "пользователю и предложите показать конкретную группу или элемент "
                                    "(as_tree с root=<путь из дерева>). " + how))
            return Subgraph(path=p, root=r, depth=depth, returned=len(nodes), truncated=truncated, hint=hint, by_meta=by_meta,
                            nodes=[NodeItem(**n) for n in nodes])

    @mcp.tool(title="Подсчёт вершин и дуг", annotations=READ)

    @compact
    @pieced(pieces.counts)
    async def count_graph(path: IrPath, ctx: Context, root: ConceptPath = "") -> Counts:
        """Число вершин и дуг во всём ИР или в подграфе от root (по полному экспорту). Ссылки считаются
        дугами, но не вершинами; путь от корня до root в подсчёт не входит."""
        p, r = ir_path(path), concept(root)
        async with traced("count_graph", ctx, {"path": p, "root": r}, path=p):
            # всегда полный экспорт (кэшируется): во фрагмент по start-target-concept-path платформа вкладывает
            # и вершины-цели ссылок из других частей ИР – подсчёт по фрагменту был бы завышен
            ir = await gw.export(p)
            r, base = find_root(ir, r)
            if base is None:
                raise ToolError(f"[not_found] Такой вершины нет: '{r}' в ИР '{p}'. Существующие пути возьмите "
                                f"из get_inforesource (top_level) или get_subgraph с меньшей глубиной.")
            return Counts(scope=r or "весь ИР", **graph.count(base))

    @mcp.tool(title="Поиск вершин", annotations=READ)

    @compact
    @pieced(pieces.matches)
    async def query_graph(
        path: IrPath,
        ctx: Context,
        text: Annotated[str, Field(max_length=200, description="Подстрока в метке (имени/значении), без учёта регистра")] = "",
        meta: Annotated[str, Field(max_length=200, description="Точная метка метавершины, например 'Наблюдение'")] = "",
        vertex_type: VertexType = "",
        limit: Annotated[int, Field(ge=1, le=200, description="Сколько совпадений вернуть; общее число – в total")] = 20,
    ) -> QueryResult:
        """Поиск вершин ИР по тексту, метапонятию и/или типу. Нужен хотя бы один критерий; заданные критерии
        объединяются через И (text и meta вместе сильно сужают поиск – для поиска по словам задавайте только
        text). Пустой результат при нескольких критериях – повторите с одним. Возвращает пути вершин и общее
        число совпадений. Фраза целиком не нашлась – ищет по частям слов (hint); ничего нет – hint об этом."""
        p = ir_path(path)
        async with traced("query_graph", ctx, {"path": p, "text": text, "meta": meta, "type": vertex_type}, path=p):
            if not (text or meta or vertex_type):
                raise ToolError("[invalid_request] Укажите хотя бы один критерий: text, meta или vertex_type.")
            ir = await gw.export(p)
            found, total = graph.find(ir, text=text or None, meta=meta or None, vtype=vertex_type or None,
                                      limit=limit)
            hint = ""
            # фразы нет – сначала газ по названию («азот 6.0» = «Азот высокой чистоты (марка 6.0)»):
            # поиск по частям слов на «аргона 4,8» давал только класс аргонов
            gas = knowledge.find_elements(ir, [text], ("Газ", "Газовая смесь"), synonyms=load_synonyms()) \
                if not found and text and not (meta or vertex_type) else {}
            if not found and not gas.get("picked") and not gas.get("ambiguous") and text \
                    and not (meta or vertex_type) and (stems := graph.word_stems(text)):
                # фраза целиком («гелий высокой чистоты марки 6») не нашлась – ищем по основам слов
                found, total = graph.find_words(ir, stems, limit=limit)
                if found:
                    hint = (f"Точно «{text}» нет; показаны вершины, где есть части слов: "
                            f"{', '.join(stems)}. Проверьте, то ли это.")
            if gas.get("picked") or gas.get("ambiguous"):
                # «азот 6.0»: фразы в названиях нет («Азот высокой чистоты (марка 6.0)»), а газ есть
                hits = [x["path"] for x in gas["picked"]] or gas["ambiguous"][text]
                found = [{"path": x, "label": split_labels(x)[-1], "type": n.get("type"), "meta": n.get("meta")}
                         for x in hits[:limit] if (n := graph.find_node(ir, x)) is not None]
                total = len(hits)
                if len(hits) == 1:
                    hint = (f"Фразы «{text}» в названиях нет, но это газ «{found[0]['label']}»: состав – "
                            f"gas_query(op=\"card\", gases=\"{found[0]['label']}\").")
                else:
                    hint = (f"Фразы «{text}» в названиях нет; подходят {len(hits)} газа(ов): "
                            + "; ".join(f"«{split_labels(x)[-1]}»" for x in hits[:10])
                            + ". Если неясно, какой нужен, – спросите пользователя.")
            elif not found and text:
                # GigaChat пустой поиск пропускал молча и отвечал общими знаниями, как будто из базы
                hint = (f"По «{text}» в этом ИР ничего нет (и по частям слов). Скажите пользователю прямо: «в базе "
                        "этого нет». Объяснение из общих знаний – только с пометкой «не из базы».")
            # «какие газы содержат ксенон»: поиск по названиям не видит вещество в составах («N + Ar + Ne + Xe»)
            known = knowledge.all_components(ir)
            pool = {c for c in known if "+" not in c} | {x for c in known for x in knowledge._parts_of(c)}
            if text and pool and (comp := knowledge.resolve_component(
                    text, pool, {**knowledge.component_names(ir), **load_synonyms()})):
                hint = (hint + " " if hint else "") + (
                    f"«{text}» – вещество ({comp}): здесь только вершины с этим словом в названии. В каких газах оно "
                    f"есть примесью – gas_query(op=\"list\", has=\"{comp}\").")
            return QueryResult(total=total, returned=len(found), hint=hint, matches=[Match(**m) for m in found])

    @mcp.tool(title="Обзор ИР", annotations=READ)

    @compact
    @pieced(pieces.summary)
    async def summarize_inforesource(path: IrPath, ctx: Context) -> Summary:
        """Портрет ИР за один вызов: числа вершин/дуг, сколько вершин каждого метапонятия, первый уровень,
        размеры разделов (sections), самый большой раздел по вершинам и по дугам (largest), группы второго
        уровня с числом элементов (groups, например «сколько газов в каждом классе») и итог по ним (groups_hint)."""
        p = ir_path(path)
        async with traced("summarize_inforesource", ctx, {"path": p}, path=p):
            ir = await gw.export(p)
            top, _ = graph.flatten(ir, max_nodes=60)
            secs = graph.sections(ir)
            groups = knowledge.group_counts(ir)
            return Summary(path=p, title=ir.get("title"), ontology=short_ir(ir.get("ontology")),
                           counts=Counts(scope="весь ИР", **graph.count(ir)), meta_stats=graph.meta_stats(ir),
                           top_level=[NodeItem(**n) for n in top if n["depth"] == 1],
                           sections=secs, largest=graph.largest(secs), groups=groups[:40],
                           groups_hint=groups_phrase(groups))

    @mcp.tool(title="Онтология ИР", annotations=READ)

    @compact
    @pieced(pieces.ontology)
    async def get_ontology(
        path: Annotated[str, Field(min_length=1, max_length=500, description=(
            "Путь ИР (тогда берётся его онтология из поля ontology) или путь самой онтологии"))],
        ctx: Context,
        root: Annotated[str, Field(max_length=1000, description=(
            "Путь понятия онтологии, с которого начать (метки через '/'); пусто – от корня"))] = "",
        depth: Annotated[int, Field(ge=1, le=12, description="Сколько уровней понятий показать от root")] = 4,
    ) -> Ontology:
        """Структура метаинформации: какие понятия могут быть в ИР, сколько их (cardinality: «в точности один»,
        «ноль или один», «непустое множество»…; «копия» – ровно одна вершина с тем же именем; «пропуск» – в ИР
        этой вершины нет, её потомки стоят у родителя), как порождаются (creation: новая, ссылка, клон), какие
        типы значений, choice – вариант альтернативы (из соседних вариантов выбирается один). Помогает понять,
        что вообще можно хранить в базе и что обязательно. Показывает depth уровней от root; глубже – root."""
        p = ir_path(path)
        async with traced("get_ontology", ctx, {"path": p, "root": root, "depth": depth}, path=p):
            head = await gw.export(p, depth=0)
            target, owner = p, None
            if head.get("meta") != "Язык ИРУО" and head.get("ontology"):
                try:
                    target, owner = normalize_ir_path(head["ontology"]), p
                except PathError as e:
                    raise ToolError(f"[invalid_path] У ИР непонятный путь онтологии: {head['ontology']!r}") from e
            onto = await gw.export(target) if owner else await gw.export(p)
            items, cut = graph.describe_ontology(onto, root=concept(root), max_depth=depth)
            if root and not items and graph.find_node(onto, concept(root)) is None:
                raise ToolError(f"[not_found] Понятия '{root}' нет в онтологии '{target}'.")
            return Ontology(ontology_path=target, for_inforesource=owner, title=onto.get("title"),
                            concepts=[OntologyConcept(**i) for i in items], truncated=cut)

    @mcp.tool(title="Запрос к базе газов", annotations=READ)
    @compact
    async def gas_query(
        ctx: Context,
        op: Annotated[Literal["list", "count", "max", "min", "sort", "card", "compare", "impurities", "errors",
                              "unfilled", "classes", "mixtures", "calc"], Field(description=(
            "Что сделать: list – список; count – сколько; max / min – больше / меньше всего по by (все равные, top – "
            "первые N); sort – по порядку by и order; card – состав и данные газа; compare – таблица газов; "
            "impurities – примеси групп (несколько групп в gases – общие и только у одной); errors – ошибки состава; "
            "unfilled – газы только с названием; classes – классы и сколько в них газов; mixtures – газовые смеси; calc – "
            "расчёт по expr"))],
        gases: Annotated[str, Field(max_length=500, description=(
            "Какие газы через «;»: класс («азот», «водород; аргон») или названия («Аргон 4.8; Аргон 6.0»); пусто – "
            "все газы базы"))] = "",
        where: Annotated[str, Field(max_length=300, description=(
            "Условия через «;»: «N > 0.01; O < 1 ppm», «чистота >= 99.999», «чистота > Азот 6.0» (порог – значение "
            "другого газа)"))] = "",
        has: Annotated[str, Field(max_length=120, description="Компоненты, которые должны быть: «водород»")] = "",
        without: Annotated[str, Field(max_length=120, description="Компоненты, которых не должно быть: «метан»")] = "",
        name_has: Annotated[str, Field(max_length=60, description="Часть названия, которая должна быть")] = "",
        name_not: Annotated[str, Field(max_length=120, description="Части названий, которых не должно быть, через «;»")
                            ] = "",
        state: Annotated[Literal["", "жидкий", "газообразный"], Field(description="Только жидкие / газообразные")] = "",
        grade: Annotated[str, Field(max_length=30, description="Марка или сорт: «5.5», «А», «первый», «высший»")] = "",
        standard: Annotated[str, Field(max_length=60, description="Часть стандарта: «ГОСТ 10157»")] = "",
        by: Annotated[str, Field(max_length=60, description=(
            "По чему max / min / sort (и итог compare): компонент («кислород»), «чистота» или «сумма примесей»"))] = "",
        order: Annotated[Literal["desc", "asc"], Field(description="sort: desc – сначала наибольшие, asc – "
                                                                   "наименьшие")] = "desc",
        top: Annotated[int, Field(ge=0, le=50, description="Первые N (0 – все равные крайние / все)")] = 0,
        show: Annotated[str, Field(max_length=120, description="Ещё компоненты в строках или столбцы compare через "
                                                               "«,»")] = "",
        unit: Annotated[Literal["%", "ppm"], Field(description="ppm – показать и в ppm")] = "%",
        expr: Annotated[str, Field(max_length=300, description=(
            "Только для op=calc: числа, + - * / скобки и ссылки [газ | что] (что – компонент, «чистота» или «сумма "
            "примесей»): «[Аргон 4.8 | азот] / [Аргон 6.0 | азот]» – во сколько раз, «[Аргон 6.0 | чистота] - "
            "[Аргон первый сорт | чистота]» – на сколько, «[Аргон 5.5 | кислород] + [Аргон 5.5 | азот]» – сумма, "
            "«17 * 23»"))] = "",
        path: Annotated[str, Field(max_length=500, description="ИР базы газов (пусто – основная)")] = "",
    ) -> GasQueryResult:
        """Любой вопрос о газах базы – одним запросом: код отбирает, считает и пишет готовый кусок ответа (answer) и
        строку understood («понял вопрос так»); расчёт (во сколько раз, на сколько, сумма) – op=calc. Вопрос из
        нескольких частей – по вызову на часть. Кусок показывай, как сказано в hint; газы, числа и составы сам не пиши. Нет
        нужной операции – так и скажи, не придумывай."""
        p = ir_path(path or GAS_BASE)
        # газы или класс в path («Моногазы/Газы на основе Азота») – модель путает path (ИР базы) с gases
        if path.strip() and p not in {e.path for e in entries()}:
            if gases.strip():
                raise ToolError("[invalid_request] path – ИР базы газов (оставьте пустым), а не газ; газы и классы – "
                                "в gases.")
            gases, p = path, ir_path(GAS_BASE)
        args = {"op": op, "gases": gases, "where": where, "has": has, "without": without, "name_has": name_has,
                "name_not": name_not, "state": state, "grade": grade, "standard": standard, "by": by,
                "order": order, "top": top, "show": show, "unit": unit, "expr": expr, "path": p}
        async with traced("gas_query", ctx, args, path=p):
            ir = await gw.export(p)
            try:
                if op == "calc":
                    if not expr.strip():
                        raise answers.AnswerError("Для op=calc нужен expr: «[Аргон 4.8 | азот] / [Аргон 6.0 | азот]».")
                    res = query.calc(ir, expr, unit, synonyms=load_synonyms())
                else:
                    res = query.run(ir, **{k: v for k, v in args.items() if k not in ("path", "expr")},
                                    synonyms=load_synonyms())
            except answers.AnswerError as e:
                raise ToolError(f"[invalid_request] {e}") from None
            block = query.block_id("gas_query", args)
            return GasQueryResult(block=block, understood=res["understood"], answer=res["answer"],
                                  total=res["total"], rows=res["rows"][:60], data_version=query.data_version(ir),
                                  result=str(res.get("result", "")),
                                  hint=BLOCK_HINT.format(block=block) if settings.pieces == "marks" else TEXT_HINT)

    @mcp.tool(title="Проверка ИР по онтологии", annotations=READ)

    @compact
    @pieced(pieces.check_report)
    async def check_inforesource(path: IrPath, ctx: Context, root: ConceptPath = "") -> CheckReport:
        """Проверяет ИР по ограничениям его онтологии: обязательные понятия («в точности один», «копия»,
        «непустое…») есть, «ноль или один» не повторяется, из вариантов альтернативы выбран ровно один, понятия
        и типы значений такие, как в онтологии. Находит незаполненные элементы (например, газы без состава и
        стандарта). root – проверить только эту часть. Ничего не меняет."""
        p, r = ir_path(path), concept(root)
        async with traced("check_inforesource", ctx, {"path": p, "root": r}, path=p):
            ir = await gw.export(p)
            if r and graph.find_node(ir, r) is None:
                raise ToolError(f"[not_found] Такой вершины нет: '{r}' в ИР '{p}'.")
            title, ontos = await ontologies_for(p)
            found = knowledge.check(ir, title, knowledge.OntologyIndex(ontos))
            if r:
                found = [x for x in found if x["path"] == r or x["path"].startswith(r + "/")]
            places: dict[str, list[str]] = {}
            for x in found:
                places.setdefault(x["path"], []).append(x["problem"])
            items = [{"path": k or "(корень ИР)", "problems": v} for k, v in places.items()]
            kinds = dict(collections.Counter(x["kind"] for x in found))
            hint = ("Нарушений нет: ИР соответствует ограничениям онтологии." if not found else
                    f"Нарушений {len(found)} в {len(items)} вершинах. Это сведения о качестве данных, а не ошибка "
                    "сервера: перечислите их пользователю (пути и что не так).")
            return CheckReport(path=p, ontology=title, ontologies_used=list(ontos), problems_total=len(found),
                               places=len(items), by_kind=kinds, problems=items[:60], truncated=len(items) > 60,
                               hint=hint)

    @mcp.tool(title="Экспорт JSON", annotations=READ)

    @compact
    @pieced(pieces.export)
    async def export_json(
        path: IrPath,
        ctx: Context,
        json_type: JsonType = "universal",
        root: ConceptPath = "",
        depth: Annotated[int, Field(ge=-1, le=max_depth, description="-1 – весь подграф")] = -1,
    ) -> ExportResult:
        """JSON ИР или фрагмента (root) в формате universal / meta / simple. simple – компактное вложенное
        дерево меток без служебных полей: удобно читать карточку элемента (например, газа) или небольшой фрагмент целиком.
        Для обзора структуры и путей вершин удобнее get_subgraph. Слишком большой ответ отклоняется."""
        p, r = ir_path(path), concept(root)
        async with traced("export_json", ctx, {"path": p, "json_type": json_type, "root": r, "depth": depth}, path=p):
            data = await gw.export(p, json_type=json_type, start=r or None, depth=None if depth < 0 else depth)
            size = len(json.dumps(data, ensure_ascii=False))
            if size > settings.max_result_chars:
                parts = ""                          # что выгрузить вместо – части с размерами
                if isinstance(data, dict) and data.get("successors"):
                    sized = [(graph.label(c), len(json.dumps(c, ensure_ascii=False)))
                             for c in data["successors"] if "link" not in c]
                    parts = (" Части первого уровня (символов JSON): " +
                             "; ".join(f"«{lab}» – {n}" + ("" if n <= settings.max_result_chars else " (тоже велик)")
                                       for lab, n in sized[:12]) + ".")
                raise ToolError(f"[too_large] JSON занимает {size} символов (> {settings.max_result_chars}) – "
                                f"целиком не выгрузить.{parts} Задайте root части (и/или depth) или используйте "
                                "get_subgraph.")
            if isinstance(data, dict):           # пути ИР в шапке экспорта – с e-mail владельца; модели он не нужен
                for k in ("path", "ontology"):
                    if isinstance(data.get(k), str):
                        data[k] = short_ir(data[k])
            return ExportResult(path=p, json_type=json_type, chars=size, data=data)

    @mcp.tool(title="Проверка JSON ИР", annotations=LOCAL)

    @compact
    async def validate_resource(
        ir_json: Annotated[dict[str, Any], Field(description="ИР в универсальном JSON-формате")], ctx: Context,
    ) -> ValidationReport:
        """Проверяет структуру ИР (universal) до импорта: обязательные поля, типы вершин, valtype, пути.
        Платформу не вызывает."""
        async with traced("validate_resource", ctx, {"title": ir_json.get("title")}):
            if len(json.dumps(ir_json, ensure_ascii=False)) > MAX_IMPORT_CHARS:
                raise ToolError(f"[too_large] JSON больше {MAX_IMPORT_CHARS} символов")
            errors = graph.validate(ir_json)
            return ValidationReport(valid=not errors, errors=errors, counts=graph.count(ir_json) if not errors else {})

    @mcp.tool(title="Статус сервиса", annotations=READ)

    @compact
    async def service_status(
        service_id: Annotated[str, Field(min_length=1, max_length=40, description="Код ИР сервиса (меню «инфо»)")],
        ctx: Context,
    ) -> dict[str, Any]:
        """Запущен ли сервис платформы."""
        async with traced("service_status", ctx, {"service_id": service_id}):
            return await gw.service_running(service_id)

    # ================================================================ WRITE
    def check_target(target: str) -> None:
        """Запись – только внутри разрешённых папок (IACPAAS_WRITE_PATHS); проверка до предпросмотра и токена."""
        if not settings.write_paths:
            return
        t = target.strip("/")
        if not any(t == w or t.startswith(w + "/") for w in settings.write_paths):
            raise ToolError("[read_only] Изменять можно только внутри папок: "
                            + ", ".join(f"«{w}»" for w in settings.write_paths)
                            + f". «{target}» вне их – его можно только читать.")

    async def guarded(op: str, target: str, args: dict, confirm_token: str, preview: dict, run,
                      human: Any = None) -> WriteResult:
        check_target(target)
        said = human.data if isinstance(human, AcceptedElicitation) else human
        if said == "no_dialog" and settings.require_dialog:
            raise ToolError(NO_DIALOG)
        if not confirm_token:
            token = policy.issue(op, args)
            if said == "dialog":                          # второй вызов откроет человеку окно с этим предпросмотром
                asked[token] = (time.time(), preview_text(preview))
                note = ("Это только предпросмотр: ничего ещё не создано и не изменено. Повторите вызов с теми же "
                        "аргументами и этим confirm_token – сервер сам покажет пользователю окно подтверждения, и "
                        "решит он. В чате согласия не спрашивайте; ответ окна придёт в результате.")
            else:
                note = ("Это только предпросмотр: ничего ещё не создано и не изменено. Покажите его пользователю "
                        "словами «будет создано / будет удалено» и спросите, выполнить ли. Только после явного "
                        "согласия повторите вызов с теми же аргументами и этим confirm_token.")
            return WriteResult(status="pending_confirmation", operation=op, target=target, preview=preview,
                               confirm_token=token, expires_in_s=policy.ttl_s, message=note)
        asked.pop(confirm_token.strip(), None)            # окно по этому токену – один раз
        if said == "dialog":                              # клиент с окном, а окна не было: токен не из его предпросмотра
            raise ToolError("[confirmation] Этот токен не из предпросмотра в этом окне – запросите предпросмотр заново "
                            "(вызов без confirm_token); подтверждение будет в окне у пользователя.")
        if isinstance(human, (AcceptedElicitation, DeclinedElicitation, CancelledElicitation)):
            log.info("human_dialog", extra={"operation": op, "target": target, "answer": human.action,
                                            "approve": getattr(said, "approve", None)})
        if isinstance(human, CancelledElicitation):
            # Claude Code в VS Code ответил «cancel» через 2 с, а окна пользователь не видел – это не отказ человека
            policy.check(op, args, confirm_token)
            return WriteResult(status="cancelled", operation=op, target=target, preview=preview,
                               message="[cancelled] Окно подтверждения закрыто без ответа или клиент его не показал – "
                                       "ничего не изменено. Не повторяй вызов. Скажи пользователю: если окна «Разрешить?» "
                                       "не было, этот клиент окна сервера не показывает – сохранить можно в веб-чате или "
                                       "консоли агента (там кнопка «Разрешить»).")
        if isinstance(human, DeclinedElicitation) or (isinstance(said, HumanDecision) and not said.approve):
            policy.check(op, args, confirm_token)         # токен сгорает: повтор того же вызова снова спросит
            log.info("write_rejected_by_human", extra={"operation": op, "target": target})
            return WriteResult(status="rejected", operation=op, target=target, preview=preview,
                               message="[rejected] Пользователь отклонил изменение в окне подтверждения – ничего не "
                                       "изменено. Не повторяй вызов; коротко скажи, что изменение отменено.")
        if reason := policy.check(op, args, confirm_token):
            raise ToolError(f"[confirmation] {reason}")
        return await execute(op, target, preview, run)

    async def execute(op: str, target: str, preview: dict, run) -> WriteResult:
        result = await gw.write(target, run)
        log.info("audit_write", extra={"operation": op, "target": target, "where": mode, "preview": preview})
        return WriteResult(status="done", operation=op, target=target, preview=preview,
                           message=f"Выполнено ({'песочница' if mode == 'sandbox' else 'платформа'}). "
                                   f"Ответ платформы: {json.dumps(result, ensure_ascii=False, default=str)[:300]}")

    def empty_names(ir: dict, paths: list[str]) -> list[str]:
        """Элементы (или элементы выбранных групп), у которых в базе только название."""
        out = []
        for p in paths:
            n = graph.find_node(ir, p) or {}
            items = [k for k in n.get("successors") or [] if "link" not in k] if knowledge.is_group(n) else [n]
            out += [graph.label(x) for x in items if not x.get("successors")]
        return out

    async def peek(p: str, c: str) -> dict:
        try:
            ir = await gw.export(p, start=c or None, depth=0)
            node = graph.find_node(ir, c) if c else ir
            return {"label": graph.label(node), "comment": node.get("comment", ""), "meta": node.get("meta")} if node else {}
        except BackendError as e:
            return {"error": str(e)}

    async def free_name(folder: str, title: str) -> str:
        """Свободное имя рядом с занятым – «Имя 2», «Имя 3»…: модель на [conflict] придумывала имена с «≥» и «%»."""
        for n in range(2, 10):
            if "error" in await peek(f"{folder}/{title} {n}", ""):
                return f"{title} {n}"
        return f"{title} {time.strftime('%d.%m %H-%M')}"

    async def check_concept(p: str, c: str, m: str) -> None:
        """Проверка аргументов до предпросмотра – токен не выдаётся на заведомо неверный вызов."""
        node, actual = await gw.export(p), []
        for lab in split_labels(c):
            node = next((x for x in node.get("successors") or [] if graph.label(x) == lab), None)
            if node is None:
                raise ToolError(f"[not_found] Вершины '{c}' нет в ИР '{p}'. Путь возьмите из query_graph.")
            actual.append(node.get("meta") or "")
        if split_labels(m) != actual:
            raise ToolError(f"[invalid_request] metaconcept_path '{m}' не совпадает с метавершинами пути – нужно "
                            f"'{join_labels(actual)}'. Предпросмотр не создан: повторите вызов с этим значением.")
        if "link" in node:
            raise ToolError("[invalid_request] Это ссылка на вершину; меняйте саму вершину по её пути.")

    asked: dict[str, tuple[float, str]] = {}            # токен -> (когда, предпросмотр) для окна подтверждения

    def can_ask(ctx: Context) -> bool:
        caps = ctx.client_capabilities
        el = caps.elicitation if caps is not None else None
        return el is not None and (el.form is not None or el.url is None)

    async def ask_human(ctx: Context, confirm_token: str) -> Any:
        """Подтверждение человеком в окне клиента (MCP elicitation: VS Code, Claude Code, Copilot). Без токена –
        только отметка, что окно доступно (будет предпросмотр); с токеном – окно с предпросмотром: модель в нём не
        участвует и подтвердить запись сама не может. Клиент без окна – «no_dialog»: там решает наш агент кнопкой
        (токен модели не показывается) или запись выключена (IACPAAS_REQUIRE_DIALOG)."""
        if settings.confirm_mode == "chat":
            # VS Code / Claude Code: подтверждение в чате – модель показывает предпросмотр и спрашивает «да / нет»,
            # клиент спрашивает разрешение на вызов. Окна сервера не используются: панель Claude Code их не
            # показывает и сама отвечает «отклонить»
            return "no_dialog"
        if not can_ask(ctx):
            return "no_dialog"
        token = (confirm_token or "").strip()
        if not token:
            return "dialog"
        for t, (ts, _) in list(asked.items()):
            if time.time() - ts > policy.ttl_s:
                asked.pop(t, None)
        if token not in asked:                            # чужой или просроченный токен – окно не нужно, откажет проверка
            return "dialog"
        return Elicit(f"IACPaaS: агент хочет изменить данные.\n\n{asked[token][1]}\n\nРазрешить?", HumanDecision)

    Human = Annotated[ElicitationResult[HumanDecision], Resolve(ask_human)]

    if mode != "off":
        tok = Annotated[str, Field(max_length=100, description="Пусто – получить предпросмотр. Токен из предпросмотра – "
                                                               "выполнить (только после согласия пользователя)")]

        @mcp.tool(title="Изменить вершину", annotations=WRITE)

        @compact
        async def update_concept(
            path: IrPath, concept_path: ConceptPath,
            metaconcept_path: Annotated[str, Field(min_length=1, max_length=1000, description=(
                "Метки метавершин по пути к вершине, по одной на каждую вершину concept_path, через '/' "
                "(значения поля meta из get_subgraph), например 'Группа/Наблюдение'"))],
            ctx: Context,
            mark: Annotated[str, Field(max_length=2000, description="Новая метка (имя или значение); пусто – не менять")] = "",
            comment: Annotated[str, Field(max_length=2000, description="Новый комментарий; пусто – не менять")] = "",
            confirm_token: tok = "", human: Human = None,
        ) -> WriteResult:
            """ИЗМЕНЯЕТ метку и/или комментарий вершины. Двухшаговый вызов через confirm_token."""
            p, c, m = ir_path(path), concept(concept_path), concept(metaconcept_path)
            if not c:
                raise ToolError("[invalid_request] concept_path не может быть пустым")
            if not mark and not comment:
                raise ToolError("[invalid_request] Укажите mark и/или comment")
            args = {"path": p, "concept_path": c, "metaconcept_path": m, "mark": mark, "comment": comment}
            check_target(p)                               # папка – до любых чтений и токена
            async with traced("update_concept", ctx, {**args, "confirm_token": confirm_token}, path=p):
                if not confirm_token:                     # выполнение по токену проверяют токен и платформа
                    await check_concept(p, c, m)
                before = await peek(p, c)
                preview = {"ИР": p, "вершина": c, "было": before,
                           "станет": {k: v for k, v in {"label": mark, "comment": comment}.items() if v}}
                return await guarded("update_concept", p, args, confirm_token, preview,
                                     lambda: backend.update_concept(p, c, m, mark, comment or None), human=human)

        @mcp.tool(title="Импорт ИР", annotations=WRITE)

        @compact
        async def import_inforesource(
            folder_path: Annotated[str, Field(min_length=1, max_length=500, description="Папка в «Мой Фонд» без имени ИР")],
            ir_json: Annotated[dict[str, Any], Field(description="ИР в универсальном JSON-формате")],
            ctx: Context,
            clear_if_exists: Annotated[bool, Field(description="Перезаписать существующий ИР (очистить перед импортом)")] = False,
            confirm_token: tok = "", human: Human = None,
        ) -> WriteResult:
            """СОЗДАЁТ или ПЕРЕЗАПИСЫВАЕТ ИР из JSON. Сначала проверяет структуру. Двухшаговый вызов."""
            folder = ir_path(folder_path)
            size = len(json.dumps(ir_json, ensure_ascii=False))
            if size > MAX_IMPORT_CHARS:
                raise ToolError(f"[too_large] JSON ИР {size} символов > {MAX_IMPORT_CHARS}: импорт такого размера "
                                "через агента не поддерживается")
            errors = graph.validate(ir_json)
            if errors:
                raise ToolError("[invalid_json] JSON не прошёл проверку: " + "; ".join(errors[:10]))
            check_name(str(ir_json["title"]))
            target = f"{folder}/{ir_json['title']}"
            check_target(target)
            args = {"folder_path": folder, "ir_json": ir_json, "clear_if_exists": clear_if_exists}
            async with traced("import_inforesource", ctx, {"folder_path": folder, "title": ir_json.get("title"),
                                                           "clear_if_exists": clear_if_exists,
                                                           "confirm_token": confirm_token}, path=target):
                exists = "error" not in await peek(target, "")
                preview = {"ИР": target, "уже существует": exists, "перезапись": clear_if_exists,
                           "размер": size_text(graph.count(ir_json))}
                res = await guarded("import_inforesource", target, args, confirm_token, preview,
                                    lambda: backend.import_ir(folder, ir_json, clear_if_exists), human=human)
                if res.status == "done":
                    remember_own(target, str(ir_json["title"]), str(ir_json.get("comment") or "импортирован агентом"))
                return res

        @mcp.tool(title="Удаление из фонда", annotations=WRITE)

        @compact
        async def delete_from_fund(
            path: IrPath, mode: DeleteMode, ctx: Context,
            concept_path: ConceptPath = "",
            metaconcept_path: Annotated[str, Field(max_length=1000, description="Для режимов подграфа: метки метавершин пути")] = "",
            confirm_token: tok = "", human: Human = None,
        ) -> WriteResult:
            """УДАЛЯЕТ: deleteIR – ИР целиком, trashIR – в корзину, clearIR – всё кроме корня,
            deleteSubGraph / clearSubGraph – подграф от concept_path. Двухшаговый вызов."""
            p, c, m = ir_path(path), concept(concept_path), concept(metaconcept_path)
            if mode in ("deleteSubGraph", "clearSubGraph") and not c:
                raise ToolError("[invalid_request] Для режимов подграфа нужен concept_path")
            args = {"path": p, "mode": mode, "concept_path": c, "metaconcept_path": m}
            check_target(p)
            async with traced("delete_from_fund", ctx, {**args, "confirm_token": confirm_token}, path=p):
                try:
                    full = await gw.export(p)
                    scope = graph.count(graph.find_node(full, c) or {}) if c else graph.count(full)
                except BackendError as e:
                    if e.kind == "not_found" and forget_own(p):
                        raise ToolError(f"[not_found] ИР «{p}» на платформе уже нет (удалён раньше, например в другом "
                                        "окне) – убран из списка своих ИР. Удалять нечего.") from e
                    raise ToolError(f"[{e.kind}] {e}.{suggest(p)}") from e
                preview = {"ИР": p, "режим": mode, "вершина": c or "(весь ИР)", "затронет": size_text(scope)}
                # «удали подборку газов высокой чистоты» при трёх похожих – модель выбрала одну сама, не спросив
                head = " ".join(p.rsplit("/", 1)[-1].split()[:3]).lower()
                similar = [e.title for e in read_own() if e.path != p and e.title.lower().startswith(head)]
                if similar and mode in ("deleteIR", "trashIR", "clearIR"):
                    preview["похожие свои ИР – тот ли выбран?"] = similar
                res = await guarded("delete_from_fund", p, args, confirm_token, preview,
                                    lambda: backend.delete(p, mode, c or None, m or None), human=human)
                if res.status == "done" and mode in ("deleteIR", "trashIR"):
                    forget_own(p)
                return res

        @mcp.tool(title="Сохранить подборку", annotations=WRITE)

        @compact
        async def save_selection(
            source_path: IrPath,
            title: Annotated[str, Field(min_length=1, max_length=120, description=(
                "Имя нового ИР: только буквы, цифры, пробел, точка и дефис, например «Газы высокой чистоты не менее 99.999»"))],
            elements: Annotated[GasList, Field(min_length=1, max_length=20000, description=(
                "Газы или целые классы одной строкой через «;» – названия или пути (из gas_query, query_graph), "
                "например «Аргон 4.8; Аргон 6.0» или «Моногазы/Газы на основе Гелия»"))],
            ctx: Context,
            folder_path: Annotated[str, Field(max_length=500, description=(
                "Папка для нового ИР; пусто – первая папка, где разрешена запись"))] = "",
            comment: Annotated[str, Field(max_length=2000, description="Комментарий к ИР: зачем подборка")] = "",
            replace: Annotated[bool, Field(description="Перезаписать ИР с таким именем, если он уже есть")] = False,
            notes: Annotated[list[Note], Field(max_length=100, description=(
                "Пометки к элементам – пишутся в комментарий вершины (результат проверки, «нет стандарта»)"))] = [],
            confirm_token: tok = "", human: Human = None,
        ) -> WriteResult:
            """СОЗДАЁТ в своей папке новый ИР – подборку выбранных элементов исходной базы на той же онтологии
            (структура и значения копируются как есть; элементы, на которые ссылаются выбранные, например газы
            из состава смеси, добавляются сами). Пометки (notes) становятся комментариями вершин; чтобы пометить
            уже сохранённую подборку – тот же вызов с replace=true. Исходный ИР не меняется. Двухшаговый вызов."""
            src = ir_path(source_path)
            if any(ch in title for ch in "/$;‴"):
                raise ToolError("[invalid_request] В имени ИР нельзя использовать символы / $ ; ‴")
            title = check_name(title)
            if folder_path:
                folder = ir_path(folder_path)
            elif settings.write_paths:
                folder = settings.write_paths[0]
            elif mode == "sandbox":
                folder = "Загрузки/Песочница ИИ"
            else:
                raise ToolError("[invalid_request] Укажите folder_path – папку для нового ИР.")
            target = f"{folder}/{title.strip()}"
            check_target(target)
            items = knowledge.split_items(elements)
            async with traced("save_selection", ctx, {"source_path": src, "title": title.strip(), "folder_path": folder,
                                                      "elements": len(items), "confirm_token": confirm_token},
                              path=src):
                ir = await gw.export(src)
                # путь любой вершины – как есть; названия газов – путями базы (неоднозначно – спросить человека)
                def exists(e: str) -> bool:                    # название газа – не путь, ошибки пути нет
                    try:
                        return graph.find_node(ir, normalize_concept_path(e)) is not None
                    except PathError:
                        return False
                exact = [concept(e) for e in items if exists(e)]
                rest = [e for e in items if not exists(e)]
                named = resolve_gases(ir, "; ".join(rest), ("Газ", "Газовая смесь"), groups=True, write=True)[0] \
                    if rest else []
                elems = list(dict.fromkeys(exact + named))
                marks = {}
                for n in notes:                                # путь пометки тоже может быть названием
                    np = concept(n.path) if exists(n.path) else next(
                        iter(resolve_gases(ir, n.path, ("Газ", "Газовая смесь"), write=True)[0]), concept(n.path))
                    marks[np] = n.text
                if stray := [p for p in marks if not any(p == e or p.startswith(e + "/") for e in elems)]:
                    raise ToolError(f"[invalid_request] Пометки к элементам вне подборки: {stray[:3]}. Путь пометки – "
                                    "один из elements (или вершина внутри него).")
                args = {"source_path": src, "title": title.strip(), "elements": elems, "folder_path": folder,
                        "comment": comment, "replace": replace, "notes": marks}
                sel, added, missing = knowledge.build_selection(ir, elems, title.strip(), comment, marks,
                                                                     knowledge.full_ir_path(ir, folder, title.strip()))
                if missing:
                    # модель сочиняла пути по названию («Гелий газообразный марка 5.0») – подсказываем настоящие
                    known = [x["path"] for x in knowledge.element_rows(ir, "Газ")]
                    near = {m: knowledge.closest_element(m, known) for m in missing[:8]}
                    hint = "; ".join(f"«{m}» → «{c}»" for m, c in near.items() if c)
                    raise ToolError(f"[not_found] В '{src}' нет вершин: {missing[:5]}. "
                                    + (f"Похожие настоящие пути: {hint}. Проверьте и повторите вызов с ними. "
                                       if hint else "")
                                    + "Названия возьмите из gas_query(op=\"list\"). Предпросмотр не создан.")
                errors = graph.validate(sel)
                if errors:
                    raise ToolError("[invalid_json] Подборка не прошла проверку: " + "; ".join(errors[:5]))
                exists = "error" not in await peek(target, "")
                if exists and not replace:
                    free = await free_name(folder, title.strip())
                    raise ToolError(f"[conflict] ИР «{target}» уже есть. Свободное имя: «{free}» – повторите вызов "
                                    "с ним; или replace=true, чтобы перезаписать (перезапись – тоже с подтверждением).")
                names = [split_labels(e)[-1] for e in elems]
                preview = {"новый ИР": target, "онтология": short_ir(sel.get("ontology")), "из ИР": src,
                           "элементы": names[:30] + ([f"… ещё {len(names) - 30}"] if len(names) > 30 else []),
                           "добавлены, т.к. на них ссылаются": [split_labels(a)[-1] for a in added],
                           "пометки": {split_labels(p)[-1]: v for p, v in marks.items()},
                           "размер": size_text(graph.count(sel)), "перезапись существующего": exists}
                preview = {k: v for k, v in preview.items() if v or v is False}
                if empty := empty_names(ir, elems):
                    preview["внимание"] = ("в базе только название, без состава и стандарта: "
                                           + ", ".join(f"«{x}»" for x in empty[:10]))
                res = await guarded("save_selection", target, args, confirm_token, preview,
                                    lambda: backend.import_ir(folder, sel, replace), human=human)
                if res.status == "done":
                    remember_own(target, title.strip(), (comment or f"Подборка из «{src.split('/')[-1]}»")
                                 + f": {len(elems)} элем.")
                return res

        @mcp.tool(title="Черновик газовой смеси", annotations=WRITE)

        @compact
        async def save_mixture(
            source_path: IrPath,
            name: Annotated[str, Field(min_length=1, max_length=200, description=(
                "Название смеси, например «Аргон-Двуокись углерода Ar-82% CO2-18%»"))],
            components: Annotated[list[MixPart], Field(min_length=2, max_length=10, description=(
                "Компоненты: газ из базы (название из gas_query) и его доля, %; сумма долей – 100"))],
            ctx: Context,
            title: Annotated[str, Field(max_length=120, description=(
                "Имя нового ИР: только буквы, цифры, пробел, точка и дефис; пусто – «Черновик смеси <название>»"))] = "",
            symbol: Annotated[str, Field(max_length=100, description="Химическое обозначение смеси, например «Ar + CO₂»")] = "",
            group: Annotated[str, Field(max_length=200, description="Класс смесей; пусто – «Черновики смесей»")] = "",
            standard: Annotated[str, Field(max_length=200, description="Стандарт (ГОСТ/ТУ), если известен")] = "",
            comment: Annotated[str, Field(max_length=2000, description="Комментарий: зачем смесь")] = "",
            replace: Annotated[bool, Field(description="Перезаписать ИР с таким именем")] = False,
            confirm_token: tok = "", human: Human = None,
        ) -> WriteResult:
            """СОЗДАЁТ в своей папке новый ИР с черновиком газовой смеси по онтологии базы газов: компоненты –
            ссылки на газы базы (их карточки копируются) с концентрацией в %. Исходная база не меняется.
            Двухшаговый вызов через confirm_token."""
            src = ir_path(source_path)
            title = check_name(title.strip() or re.sub(r"[^А-Яа-яЁёA-Za-z0-9 .\-]+", " ",
                                                       f"Черновик смеси {name}").replace("  ", " ")[:120])
            if any(ch in title + name for ch in "/$;‴"):
                raise ToolError("[invalid_request] В имени ИР и смеси нельзя использовать символы / $ ; ‴")
            parts = [(concept(c.gas), c.percent) for c in components]
            if len({p for p, _ in parts}) != len(parts):
                raise ToolError("[invalid_request] Один и тот же газ указан дважды.")
            total = sum(pct for _, pct in parts)
            if abs(total - 100) > 0.001:
                raise ToolError(f"[invalid_request] Сумма долей {total:g} %, а должна быть 100 %. Предпросмотр не "
                                "создан: уточните доли у пользователя.")
            folder = settings.write_paths[0] if settings.write_paths else "Загрузки/Песочница ИИ"
            target = f"{folder}/{title.strip()}"
            check_target(target)
            args = {"source_path": src, "title": title.strip(), "name": name.strip(), "components": parts,
                    "symbol": symbol.strip(), "group": group.strip() or "Черновики смесей", "standard": standard.strip(),
                    "comment": comment, "replace": replace}
            async with traced("save_mixture", ctx, {**{k: v for k, v in args.items() if k != "components"},
                                                    "components": len(parts), "confirm_token": confirm_token},
                              path=src):
                ir = await gw.export(src)
                # газ можно назвать как в ответе gas_query («Аргон газообразный первый сорт», «Гелий 5.0») – путь
                # находит код; несколько подходящих – спросить человека (resolve_gases с write=True)
                parts = [(p if graph.find_node(ir, p) is not None else resolve_gases(ir, p, ("Газ",), write=True)[0][0],
                          pct) for p, pct in parts]
                if len({p for p, _ in parts}) != len(parts):
                    raise ToolError("[invalid_request] Один и тот же газ указан дважды.")
                args["components"] = parts
                sel, missing = knowledge.build_mixture(ir, args["title"], args["name"], args["group"], parts,
                                                       args["standard"], comment, args["symbol"],
                                                       knowledge.full_ir_path(ir, folder, args["title"]))
                if missing:
                    raise ToolError(f"[not_found] Это не газы базы '{src}': {missing[:5]}. Пути газов возьмите из "
                                    "gas_query(op=\"list\"). Предпросмотр не создан.")
                if errors := graph.validate(sel):
                    raise ToolError("[invalid_json] Смесь не прошла проверку: " + "; ".join(errors[:5]))
                exists = "error" not in await peek(target, "")
                if exists and not replace:
                    free = await free_name(folder, title.strip())
                    raise ToolError(f"[conflict] ИР «{target}» уже есть. Свободное имя: «{free}» – повторите вызов "
                                    "с ним; или replace=true.")
                preview = {"новый ИР": target, "смесь": args["name"], "обозначение": args["symbol"] or "не указано",
                           "класс": args["group"],
                           "состав": [f"{split_labels(p)[-1]} – {pct:g} %" for p, pct in parts],
                           "стандарт": args["standard"] or "не указан (по онтологии обязателен – проверка это отметит)",
                           "размер": size_text(graph.count(sel)), "перезапись существующего": exists}
                notes = []
                for p, _ in parts:
                    if not empty_names(ir, [p]):
                        continue
                    group = graph.find_node(ir, join_labels(split_labels(p)[:-1])) or {}
                    filled = [graph.label(k) for k in group.get("successors") or [] if k.get("successors")]
                    notes.append(f"«{split_labels(p)[-1]}» в базе записан только названием (нет состава, чистоты и "
                                 "стандарта)" + (f"; заполненные газы того же класса: {', '.join(filled[:4])}"
                                                if filled else "; заполненных газов этого вещества в базе нет"))
                if notes:
                    preview["внимание"] = "; ".join(notes)
                res = await guarded("save_mixture", target, args, confirm_token, preview,
                                    lambda: backend.import_ir(folder, sel, replace), human=human)
                if res.status == "done":
                    remember_own(target, args["title"], comment or f"Черновик смеси «{args['name']}»")
                return res

        # PUT /api/data/update на платформе не работает – пометки через save_selection(notes)
        if mode != "sandbox":
            mcp.remove_tool("update_concept")

        # сервис платформы может изменить что угодно – ограничить папкой нельзя; на реальной платформе их нет
        if mode == "sandbox":
            @mcp.tool(title="Запуск сервиса", annotations=WRITE)

            @compact
            async def run_service(
                service_id: Annotated[str, Field(min_length=1, max_length=40, description="Код ИР сервиса (меню «инфо»)")],
                ctx: Context,
                imports: Annotated[list[ImportItem], Field(max_length=10, description="ИР для импорта перед запуском")] = [],
                exports: Annotated[list[ExportItem], Field(max_length=10, description="ИР для экспорта после работы")] = [],
                confirm_token: tok = "", human: Human = None,
            ) -> WriteResult:
                """Запускает безынтерфейсный сервис платформы (import → run → export). Импорт меняет данные,
                поэтому это изменяющая операция. Сервис должен уложиться в 30 с. Двухшаговый вызов."""
                imp = [{"path": ir_path(i.path), "json": i.ir_json, "clear_if_exists": i.clear_if_exists} for i in imports]
                exp = [{"path": ir_path(e.path), "json_type": e.json_type, "start": concept(e.root) or None,
                        "depth": None if e.depth < 0 else e.depth} for e in exports]
                args = {"service_id": service_id, "imports": imp, "exports": exp}
                async with traced("run_service", ctx, {"service_id": service_id, "imports": [i["path"] for i in imp],
                                                       "exports": [e["path"] for e in exp], "confirm_token": confirm_token}):
                    preview = {"сервис": service_id, "импорт": [i["path"] for i in imp], "экспорт": [e["path"] for e in exp]}
                    return await guarded("run_service", service_id, args, confirm_token, preview,
                                         lambda: backend.run_service(service_id, imp, exp, True), human=human)

            @mcp.tool(title="Остановка сервиса", annotations=WRITE)

            @compact
            async def stop_service(
                service_id: Annotated[str, Field(min_length=1, max_length=40)], ctx: Context, confirm_token: tok = "", human: Human = None,
            ) -> WriteResult:
                """Останавливает запущенный сервис платформы. Двухшаговый вызов."""
                args = {"service_id": service_id}
                async with traced("stop_service", ctx, {**args, "confirm_token": confirm_token}):
                    return await guarded("stop_service", service_id, args, confirm_token, {"сервис": service_id},
                                         lambda: backend.stop_service(service_id), human=human)

        # ================================================================ PROMPTS / RESOURCE

    @mcp.prompt(title="Разобрать инфоресурс")
    def analyze_inforesource(path: str) -> str:
        """Пошаговый разбор ИР: обзор, структура, ключевые понятия."""
        return (f"Разбери инфоресурс IACPaaS «{path}». Шаги: найди его через list_inforesources (если путь неточный), "
                "вызови summarize_inforesource, затем get_subgraph с depth=1 и при необходимости глубже по самым "
                "крупным группам. Итог: на какой онтологии построен ИР, какие группы понятий, сколько вершин и дуг, "
                "3–5 примеров вершин с путями. Ничего не изменяй.")

    @mcp.resource("iacpaas://guide", title="Как пользоваться сервером IACPaaS", mime_type="text/markdown")
    def guide() -> str:
        """Краткая справка по инструментам и ограничениям."""
        return (
            "# IACPaaS MCP\n"
            f"- Режим: {backend.name}; запись: {mode}.\n"
            "- Пути ИР – внутри «Мой Фонд»: 'Папка/Имя ИР'. Пути вершин – метки от корня через '/'.\n"
            f"- Лимиты: depth ≤ {max_depth}, вершин в ответе ≤ {max_nodes}, JSON ≤ {settings.max_result_chars} символов.\n"
            "- Платформа отвечает до 30 с; "
            + (f"результаты чтения кэшируются на {int(settings.cache_ttl_s)} с.\n" if settings.cache_ttl_s
               else "данные читаются с платформы при каждом запросе (без кэша).\n") +
            "- Запись: вызов без confirm_token → предпросмотр; с токеном → выполнение.\n")

    for t in mcp._tool_manager.list_tools():
        t.parameters = _strip_titles(t.parameters)
        if t.output_schema:
            t.output_schema = _strip_titles(t.output_schema)

    mcp._iacpaas = {"gateway": gw, "policy": policy, "mode": mode, "registry": entries}   # для тестов и статуса
    return mcp
