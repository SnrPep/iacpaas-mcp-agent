"""Ядро standalone-агента – общее для CLI и веб-чата.

- MCP: тот же сервер, что у VS Code; подключение stdio (сервер запускается подпроцессом), http (к запущенному
  серверу с Bearer-токеном) или inproc (сервер в этом же процессе – для тестов и демо).
- Модель: любой профиль из config/models.toml, переключается на лету; контекст сессии общий для всех моделей.
- Память: LangGraph AsyncSqliteSaver (data/agent.sqlite) – диалоги и незавершённые подтверждения переживают
  перезапуск; история хранится целиком, а модели прошлые вопросы уходят без результатов инструментов
  (ContextMiddleware).
- Безопасность: лимиты вызовов, подтверждение человеком перед реальным выполнением изменяющих инструментов.
- Дописывание на ходу (как в Claude Code): сообщение, присланное во время работы агента (steer), попадает в диалог
  перед ближайшим шагом модели; не успело (модель уже писала итоговый ответ) – возвращается клиенту в leftover.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

from fastmcp.client import Client                                    # noqa: E402
from fastmcp.client.transports import StdioTransport                 # noqa: E402
from langchain.agents import create_agent                            # noqa: E402
from langchain.agents.middleware import (AgentMiddleware, HumanInTheLoopMiddleware,  # noqa: E402
                                         ModelCallLimitMiddleware, ModelRetryMiddleware,
                                         ToolCallLimitMiddleware, ToolErrorMiddleware, hook_config)
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig  # noqa: E402
from langchain.agents.middleware.model_retry import default_retry_on  # noqa: E402
from langchain.mcp import MCPAdapter                                 # noqa: E402
from langchain_core.messages import (AIMessage, AIMessageChunk, HumanMessage, RemoveMessage,  # noqa: E402
                                     ToolMessage)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver         # noqa: E402
from langgraph.types import Command, interrupt                       # noqa: E402

from iacpaas_mcp.config import PROJECT_ROOT                          # noqa: E402
from iacpaas_mcp.knowledge import split_items                       # noqa: E402

from . import check                                                  # noqa: E402

from .locks import SharedLock                                        # noqa: E402
from .models import (context_budget, context_window, chars_per_token, make_model, parallel_limit,  # noqa: E402
                     resolve_name)
from .sessions import SessionStore                                   # noqa: E402
from .settings import AgentSettings                                  # noqa: E402
from .trace import JsonlTrace, usage                                        # noqa: E402

DEFAULT_PROMPT = "Ты ассистент по базам знаний IACPaaS. Работай только через инструменты MCP."
STOPPED = "⏹ Запрос остановлен пользователем."
LIMIT_NOTE = ("Не удалось ответить: модель слишком много раз вызывала инструменты и не пришла к ответу (сработал "
              "предел шагов). Переформулируйте вопрос или задайте его частями.")
LIMIT_PIECES = ("⚠ **Модель не уложилась в предел шагов** – ниже всё, что по этому вопросу собрал код из базы (у каждой "
                "части – как код понял запрос).\n\n")


# уточнение к прошлому ответу: «а из них», «его», «эти», «тот же», «а какие жидкие»
FOLLOW = re.compile(r"\b(?:он|она|оно|они|его|её|ее|их|него|неё|нее|них|ним|ней|нему|этот|эта|это|эти|этих|этого|"
                    r"этой|тот|та|те|того|той|тех|такой|такие|там|тогда|ещё|еще|выше|предыдущ\w*|прошл\w*)\b|"
                    r"^\s*а\s|^\s*и\s|из\s+них", re.I)
FOLLOW_NOTE = ("\n\n[Справка агента: это уточнение к прошлому. Что код посчитал по прошлым вопросам (последний – "
               "внизу):\n{items}\nОпредели, к какому вопросу уточнение (обычно к последнему, где есть такие газы), и в "
               "запросе укажи gases явными названиями этих газов через «;». Неясно, к какому, – спроси пользователя, "
               "перечислив варианты.]")


def follow_items(msgs: list, last: int = 20) -> list[str]:
    """Прошлые вопросы и что по ним сделал код (кодом, без пересказа): у gas_query – «понял так» и газы, у
    инструментов платформы – инструмент, ИР и вершина (уточнение «сколько там связей» – к вершине последнего)."""
    blocks: list[list] = []
    calls: dict[str, tuple[str, dict]] = {}
    for m in msgs:
        if is_question(m):
            blocks.append([_shown(m), []])
            continue
        if isinstance(m, AIMessage):
            calls.update({tc.get("id"): (tc["name"], tc.get("args") or {}) for tc in m.tool_calls or []})
        if not (blocks and isinstance(m, ToolMessage) and m.status != "error"):
            continue
        try:
            d = json.loads(_text(m.content))
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if m.name == check.GAS_TOOL and d.get("understood"):
            names = [r["name"] for r in d.get("rows") or [] if isinstance(r, dict) and r.get("name")]
            blocks[-1][1].append(f"понял так: {d['understood']}" + (
                f"; газы ({len(names)}): " + "; ".join(names[:15]) + (" …" if len(names) > 15 else "")
                if names else ""))
        elif m.name != check.GAS_TOOL:
            args = calls.get(m.tool_call_id, ("", {}))[1]
            where = ", ".join(f"{k}=«{args[k]}»" for k in ("path", "root", "concept_path", "filter")
                              if str(args.get(k) or "").strip())
            blocks[-1][1].append(f"{m.name}" + (f"({where})" if where else ""))
    out = []
    for n, (q, got) in enumerate(blocks, 1):
        if got and not DIALOG_ASK.search(q):
            out.append(f"{n}) «{q[:150]}» → " + " | ".join(got))
    return out[-last:]


def _row_names(d: dict) -> list[str]:
    return [r["name"] for r in d.get("rows") or [] if isinstance(r, dict) and r.get("name")]


def follow_prev(msgs: list) -> list[tuple[str, dict, list[str]]]:
    """Все прошлые вопросы для уточнения: [(вопрос, аргументы запроса о газах или None, газы ответа)]. Запрос – тот, чей
    кусок стоит в ответе (код пишет список таких запросов в метаданные ответа, исправленный кодом – тоже; пустой
    список – ответ без кусков о газах: запись, по памяти), а в старых ответах – последний удачный gas_query хода; расчёт
    (calc) газы не отбирает – не берётся; вопрос о самом диалоге – без запроса."""
    blocks: list[list] = []                          # [вопрос, аргументы, газы]
    calls: dict[str, dict] = {}
    for m in msgs:
        if is_question(m):
            blocks.append([_shown(m), None, []])
            continue
        if not blocks:
            continue
        if isinstance(m, AIMessage):
            calls.update({tc.get("id"): tc.get("args") or {} for tc in m.tool_calls or []})
            if not m.tool_calls and not _meta(m).get("draft") and "queries" in _meta(m):
                qs = [q for q in _meta(m)["queries"] if (q.get("args") or {}).get("op") != "calc"]
                blocks[-1][1:] = [qs[-1]["args"], qs[-1].get("gases") or []] if qs else [None, []]
        elif isinstance(m, ToolMessage) and m.name == check.GAS_TOOL and m.status != "error":
            args = calls.get(m.tool_call_id) or {}
            try:
                d = json.loads(_text(m.content))
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("block") and args.get("op") != "calc":
                blocks[-1][1:] = [args, _row_names(d)]
    return [(q, None if DIALOG_ASK.search(q) else a, g) for q, a, g in blocks]


FAILED = re.compile(r"^Model call failed after (\d+) attempts? with (\w+)")
FAILED_NOTE = ("⚠ **Модель не ответила:** {err} после {n} попыток (сбой на стороне модели, не в данных). "
               "Повторите вопрос{tail}")


def _ru(text: str) -> str:
    """Служебные английские сообщения LangChain – по-русски."""
    if m := FAILED.match(text.strip()):
        return FAILED_NOTE.format(err="таймаут" if "Timeout" in m[2] else m[2], n=m[1], tail=".")
    return LIMIT_NOTE if text.startswith("Model call limits exceeded") else text
# Пометка для модели у сообщения, дописанного во время работы: не всё дописанное – про текущую задачу
# Как в Claude Code: сообщение, присланное во время работы, получает тот же агент на ближайшем шаге
STEER_NOTE = ("[Пользователь написал это, пока ты работал над задачей. Одной короткой фразой подтверди, что учтёшь "
              "(без вопросов пользователю и без названий инструментов), и продолжай. Поправка или уточнение – учти в "
              "задаче; новый вопрос – выполни тоже и ответь на всё в итоговом ответе.]\n")
STEER_HEAD = re.compile(r"^\[Пользователь написал это, пока ты работал[^\]]*\]\n")
# Просят код – напоминание рядом с вопросом: правило в инструкции GigaChat не выполнял и «показывал результат»
# (выдуманную таблицу с ошибкой: N = 10 вместо 11). В истории диалога пользователю не показывается.
CODE_ASK = re.compile(r"\bкод|python|питон|скрипт|программ", re.IGNORECASE)
# Вопрос о самом диалоге – справка с прошлыми вопросами по порядку: GigaChat называл «первым вопросом» пример
# из своей инструкции, хотя настоящий вопрос был у него в истории
DIALOG_ASK = re.compile(r"(перв|последн|прошл|предыдущ|раньш)\w*(\s+\w+){0,2}\s+(вопрос|ответ)"
                        r"|мо[йи]\w*(\s+\w+){0,2}\s+вопрос|что я (спрашивал|писал|просил)|как\w* вопрос\w* я"
                        r"|ты (отвечал|писал|говорил|ответил)|в (твоём|твоем|своём|своем) ответе", re.IGNORECASE)
ANSWER_ASK = re.compile(r"ответ|отвечал|писал|говорил", re.IGNORECASE)
DIALOG_NOTE = ("\n\n[Справка агента: вопросы пользователя в этом диалоге по порядку: {qs}. Отвечай по ней, а не по "
               "примерам из инструкции.]")
# Ответ на первый вопрос через 8 вопросов уже вне окна памяти – модель выдумала суммы примесей
ANSWERS_NOTE = ("\n\n[Справка агента: твои ответы на вопросы этого диалога (начало каждого): {ans}. Числа о прошлых "
                "ответах бери только отсюда, даже если раньше в диалоге ты называл другие – верна справка; нужного "
                "здесь нет – так и скажи.]")
CODE_NOTE = ("\n\n[Напоминание: код ты не запускаешь. Результаты его работы – таблицу, числа, вывод print – не пиши "
             "и не придумывай, даже если просят; после кода одна строка: «Запустите код – он выведет результат». "
             "Только стандартная библиотека Python.]")

# Переменные, которые нужны подпроцессу MCP-сервера (ключи моделей ему не передаются)
CHILD_ENV_PREFIXES = ("IACPAAS_", "MCP_", "LOG_", "MOCK_", "PYTHON", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                      "no_proxy", "http_proxy", "https_proxy")
CHILD_ENV_KEYS = {"PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
                  "APPDATA", "LOCALAPPDATA", "COMSPEC", "PATHEXT", "LANG", "LC_ALL"}


def tool_failure(exc: Exception, request) -> str:
    """Исключение при вызове инструмента (сервер упал, обрыв связи, таймаут) -> ToolMessage(status="error").
    История диалога остаётся корректной, модель видит понятную причину. Детали – только в трассе/логах."""
    name = (getattr(request, "tool_call", None) or {}).get("name", "инструмент")
    return (f"[tool_unavailable] Вызов {name} не выполнен: {type(exc).__name__}. MCP-сервер или платформа "
            "недоступны. Не повторяй вызов в этом ответе; сообщи пользователю, что данные сейчас недоступны.")


@dataclass
class Approval:
    index: int
    tool: str
    args: dict[str, Any]
    description: str = ""


@dataclass
class TurnResult:
    session_id: str
    model: str
    status: str                                  # done | needs_approval | error
    answer: str = ""
    verify: dict[str, Any] = field(default_factory=dict)     # проверка ответа кодом (check.py)
    approvals: list[Approval] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def tool_annotations(tool) -> dict[str, Any]:
    return (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {}) or {}


def is_write_tool(tool) -> bool:
    ann = tool_annotations(tool)
    return not ann.get("read_only_hint", False)



def _text(content: Any) -> str:
    """Текст сообщения модели: строка или список блоков (у Anthropic и др.)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text")


def _turn_dict(r: "TurnResult") -> dict[str, Any]:
    d = asdict(r)
    d.pop("events", None)                        # события уже ушли потоком
    return d


def _messages(state) -> list:
    return state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])


def _meta(m) -> dict:
    return (m.response_metadata or {}).get("iacpaas", {})


def _shown(m) -> str:
    """Текст сообщения пользователя, как он его писал: без пометок и справок, добавленных для модели."""
    meta, text = _meta(m), _text(m.content)
    if meta.get("shown") is not None:
        return meta["shown"]
    if meta.get("steer"):                                # пометка к дописанному (и её прежние формулировки)
        text = STEER_HEAD.sub("", text, count=1)
    if meta.get("note") and text.endswith(CODE_NOTE):
        text = text[:-len(CODE_NOTE)]
    return text


def is_question(m) -> bool:
    """Вопрос пользователя – начало хода (дописанное во время работы и замечание проверки – часть хода)."""
    return isinstance(m, HumanMessage) and not _meta(m).get("steer") and not _meta(m).get("check")


KEEP_MIN, KEEP_MAX = 10, 100                     # границы длины памяти (ползунок в веб-чате)
CHARS_PER_TOKEN = 2                              # оценка с запасом: у GigaChat по трассе ~2 символа на токен
STUB = ("[Результат «{name}» из более раннего вопроса здесь не повторяется, чтобы не тратить токены ({n} симв.). "
        "Нужны эти данные – вызови инструмент снова.]")
EARLIER = ("[Начало этого диалога сжато кодом (это не другой сеанс – ты его видел). Ранние вопросы пользователя по "
           "порядку и выписка из твоих ответов – дословно, без пересказа: {qs}. Чего в выписке нет – о тех ответах не "
           "называй: предложи получить заново.]\n\n")


def _visible(m) -> bool:
    """Сообщение, которое видно в чате: вопрос или ответ модели (не вызов инструмента и не его результат)."""
    return isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)


def _size(msgs: list) -> int:
    """Сколько символов уйдёт модели: текст и аргументы вызовов."""
    return sum(len(_text(m.content)) + sum(len(json.dumps(tc.get("args"), ensure_ascii=False))
                                           for tc in getattr(m, "tool_calls", None) or []) for m in msgs)


def extract(answer: str, limit: int = 700) -> str:
    """Выписка из ответа – кодом, дословно: строки с числами (и строки таблиц) без разметки; чисел нет – начало."""
    lines = [re.sub(r"[*_`#>]+", "", x).strip(" |-•\t") for x in answer.splitlines()]
    lines = [re.sub(r"\s*\|\s*", " | ", x) for x in lines if x and not re.fullmatch(r"[\s|:\-]+", x)]
    # длинная строка режется по границе слова: «ещё 32» не должно стать «ещё 3»
    keep = [x if len(x) <= 160 else x[:160].rsplit(" ", 1)[0] + " …" for x in lines
            if re.search(r"\d", x) and not x.startswith("⚠ Проверка кодом")]
    out = "; ".join(keep) if keep else " ".join(lines)[:200]
    return out if len(out) <= limit else out[:limit].rsplit(";", 1)[0] + "; …"


def digest_item(n: int, block: list, short: bool = False) -> str:
    """Строка выжимки о прошлом вопросе: вопрос, выписка из ответа, пути газов из вызовов, пометка проверки."""
    q = _shown(block[0]).strip() if is_question(block[0]) else ""
    s = f"{n}) «{q[:120 if short else 300]}»"
    if short:
        return s
    ans = next((m for m in reversed(block) if isinstance(m, AIMessage) and not m.tool_calls
                and _text(m.content).strip() and not _meta(m).get("draft")), None)
    if ans is not None:
        s += f" → ответ: {extract(_text(ans.content))}"
        if (_meta(ans).get("verify") or {}).get("problems"):
            s += " (проверка кодом: в этом ответе были расхождения с данными)"
    paths: list[str] = []
    for m in block:
        for tc in getattr(m, "tool_calls", None) or []:
            for k in ("gases", "root", "elements", "source_path"):
                v = (tc.get("args") or {}).get(k)
                for x in (split_items(v) if k in ("gases", "elements") else [v]):   # газы – строкой «газ 1; газ 2»
                    if isinstance(x, str) and x and x not in paths:
                        paths.append(x)
    if paths:
        s += "; пути: " + ", ".join(f"«{p[:120]}»" for p in paths[:6]) + (" …" if len(paths) > 6 else "")
    return s


def _blocks(past: list) -> list[list]:
    """Прошлое по вопросам: [вопрос, вызовы, результаты, ответ], …"""
    out: list[list] = []
    for m in past:
        if is_question(m) or not out:
            out.append([m])
        else:
            out[-1].append(m)
    return out


def digest(blocks: list[list], short_upto: int = 0) -> str:
    items = [digest_item(n, b, short=n <= short_upto) for n, b in enumerate(blocks, 1)]
    if short_upto >= len(blocks) and len(items) > 12:       # первые вопросы важнее середины («какой был первый»)
        items = items[:6] + ["…"] + items[-5:]
    return " ".join(items)


def no_call(m: AIMessage) -> dict:
    """additional_kwargs сообщения, вызов которого отменил код: без function_call и functions_state_id GigaChat –
    по ним GigaChat ищет результат отменённого вызова и отвечает 422 на весь диалог."""
    return {k: v for k, v in (m.additional_kwargs or {}).items() if k not in ("function_call", "functions_state_id")}


def compact_messages(msgs: list, keep: int, budget: int | None = None, compact_id: str | None = None,
                     info: dict | None = None) -> list:
    """Что уходит модели (история в БД не меняется). Текущий вопрос – всегда целиком. Прошлое сжимается кодом,
    всегда в одном порядке (предсказуемо, без пересказа моделью):
    1) черновики и замечания проверки – убираются; результаты инструментов старше предыдущего вопроса – заглушкой;
    2) сжато вручную («сжать весь чат», compact_id) – всё до отметки уходит в выжимку;
    3) окно памяти keep: в нём только сообщения чата (вопросы и ответы), лишнее – рывками кратно трети окна
       (реже ломается кэш GigaChat) – в выжимку; предыдущий вопрос – всегда целиком;
    4) бюджет (токены прошлого из профиля модели): не влезло – старые вопросы в выжимку, затем данные
       предыдущего вопроса – заглушкой, затем выжимка короче (у старых вопросов только сам вопрос).
    Выжимка – вопросы по порядку, числа и строки таблиц из ответов дословно, пути газов из вызовов."""
    qs = [i for i, m in enumerate(msgs) if is_question(m)]
    last = qs[-1] if qs else -1
    if last <= 0:
        return msgs
    full_from = qs[-2] if len(qs) > 1 else last
    past = []
    for i, m in enumerate(msgs[:last]):
        if _meta(m).get("draft") or _meta(m).get("check"):
            continue
        # ответ без вызова, а function_call GigaChat остался (история, записанная до правки) – вызов без результата
        if isinstance(m, AIMessage) and not m.tool_calls and "function_call" in (m.additional_kwargs or {}):
            m = m.model_copy(update={"additional_kwargs": no_call(m)})
        # данные предыдущего вопроса – целиком: на них ссылаются «сохрани эти газы», «отсортируй их»
        if i < full_from and isinstance(m, ToolMessage) and len(text := _text(m.content)) > 200:
            m = ToolMessage(content=STUB.format(name=m.name or "инструмент", n=len(text)), id=m.id,
                            tool_call_id=m.tool_call_id, name=m.name, status=m.status)
        past.append(m)
    blocks = _blocks(past)
    cut = 0                                              # сколько первых вопросов – в выжимку
    if compact_id:
        cut = next((i + 1 for i, b in enumerate(blocks) if any(m.id == compact_id for m in b)), 0)
    vis = [sum(_visible(m) for m in b) for b in blocks]
    if keep and sum(vis[cut:]) > keep:
        step = max(keep // 3, 1)
        need = -(-(sum(vis[cut:]) - keep) // step) * step
        while cut < len(blocks) - 1 and need > 0:
            need -= vis[cut]
            cut += 1
    chars = budget * CHARS_PER_TOKEN if budget else 0
    short = 0
    if chars:
        def total() -> int:
            return _size([m for b in blocks[cut:] for m in b]) + len(digest(blocks[:cut], short))
        while cut < len(blocks) - 1 and total() > chars:
            cut += 1
        if total() > chars and cut < len(blocks):        # данные предыдущего вопроса – тоже заглушкой
            blocks[-1] = [ToolMessage(content=STUB.format(name=m.name or "инструмент", n=len(_text(m.content))),
                                      id=m.id, tool_call_id=m.tool_call_id, name=m.name, status=m.status)
                          if isinstance(m, ToolMessage) and len(_text(m.content)) > 200 else m for m in blocks[-1]]
        while short < cut and total() > chars:          # выжимка короче: у старых вопросов – только вопрос
            short += 1
    kept = [m for b in blocks[cut:] for m in b]
    if info is not None:
        info.update({"questions": len(blocks), "digest": cut, "full": len(blocks) - cut, "short": short,
                     "tokens": (_size(kept) + len(digest(blocks[:cut], short))) // CHARS_PER_TOKEN,
                     "budget": budget or 0, "manual": bool(compact_id)})
    if not cut:
        return past + list(msgs[last:])
    note = EARLIER.format(qs=digest(blocks[:cut], short))
    rest = kept + list(msgs[last:])
    first = rest[0]
    return [first.model_copy(update={"content": note + _text(first.content)}), *rest[1:]]


class ContextMiddleware(AgentMiddleware):
    """Экономия токенов на каждом шаге модели (compact_messages): длина памяти – с ползунка (keep_messages),
    бюджет – из профиля модели (context_tokens), отметка «сжать весь чат» – из списка сессий (compact_id).
    Сводка – событием context: веб-чат показывает, сколько вопросов целиком и сколько выжимкой."""

    def __init__(self, keep: int, budget: int | None = None, runtime: "AgentRuntime | None" = None):
        super().__init__()
        self.keep, self.budget, self.rt = keep, budget, runtime

    async def awrap_model_call(self, request, handler):
        from langgraph.config import get_config
        try:                                             # длина памяти из веб-чата (ползунок) – на этот ход
            conf = get_config().get("configurable", {})
        except RuntimeError:                             # вне графа (тесты middleware)
            conf = {}
        keep = conf.get("keep_messages") or self.keep
        info: dict[str, Any] = {}
        msgs = compact_messages(request.messages, keep, self.budget, conf.get("compact_id"), info)
        sid = conf.get("thread_id")
        if info and self.rt and (emit := self.rt._emit.get(sid)):
            emit({"event": "context", "session": sid, **info})
        return await handler(request.override(messages=msgs))


MISSING_ARG = re.compile(r"validation error.*?\n(\w+)\n\s+Field required", re.S)
REPEAT = ("[repeat] Этот вызов с теми же аргументами уже вернул ошибку – повторять его бесполезно. Исправь "
          "аргументы по подсказке из прошлой ошибки или ответь пользователю, что не получилось и почему.")
REPEAT_OK = ("[repeat] Этот вызов с теми же аргументами в этом ответе уже выполнен – его результат выше, данные "
             "те же. Не повторяй: возьми тот результат и отвечай (или вызови другой инструмент).")
REPEAT_STOP = ("[stop] Ты снова повторяешь тот же вызов. Инструменты в этом ответе больше не вызывай: ответь "
               "пользователю по уже полученным данным, а чего не получилось – так и скажи.")


class RepeatGuardMiddleware(AgentMiddleware):
    """Ошибка проверки аргументов – по-русски; тот же вызов в одном ответе второй раз не выполняется
    (GigaChat иначе повторяет его до предела шагов, а удачный – ещё и с полным результатом каждый раз)."""

    @staticmethod
    def _before(request) -> str:
        """Был ли уже в этом ходе такой же вызов: 'error', 'success' или ''. Ход – с вопроса: после замечания проверки
        модель вызывала все инструменты заново и упиралась в лимит вызовов."""
        tc, msgs = request.tool_call, _messages(request.state)
        start = max((i for i, m in enumerate(msgs) if is_question(m)), default=-1)
        done = {m.tool_call_id: m.status for m in msgs[start + 1:] if isinstance(m, ToolMessage)}
        same = [done[c["id"]] for m in msgs[start + 1:] if isinstance(m, AIMessage) for c in m.tool_calls
                if c["id"] != tc["id"] and c["id"] in done and c["name"] == tc["name"] and c["args"] == tc["args"]]
        return "error" if "error" in same else "success" if same else ""

    async def awrap_tool_call(self, request, handler):
        tc = request.tool_call
        if before := self._before(request):
            msgs = _messages(request.state)
            start = max((i for i, m in enumerate(msgs) if is_question(m)), default=-1)
            repeats = sum(1 for m in msgs[start + 1:] if isinstance(m, ToolMessage) and _text(m.content).startswith(
                ("[repeat]", "[stop]")))
            # GigaChat повторял тот же неверный вызов по 10 раз – до предела шагов и без ответа
            text = REPEAT_STOP if repeats >= 2 else REPEAT if before == "error" else REPEAT_OK
            return ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"], status=before)
        res = await handler(request)
        if isinstance(res, ToolMessage) and res.status == "error":
            if m := MISSING_ARG.search(_text(res.content)):
                res.content = (f"[invalid_request] Не хватает обязательного параметра «{m.group(1)}». "
                               "Повтори вызов, добавив его.")
            elif "validation error" in (text := _text(res.content)):
                res.content = schema_error(text)
        return res


def schema_error(text: str) -> str:
    """Ошибка проверки аргументов (pydantic, по-английски) – по-русски: модель её не понимала и повторяла вызов."""
    out = []
    for field_name, msg in re.findall(r"\n(\w+)\n\s+(.+?)(?: \[type=|\n|$)", text):
        if m := re.search(r"at least (\d+) items?.*?not (\d+)", msg):
            msg = f"в списке нужно не меньше {m.group(1)}, передано {m.group(2)} – все значения одним списком"
        elif m := re.search(r"at most (\d+) items?", msg):
            msg = f"в списке не больше {m.group(1)}"
        elif "Input should be" in msg:
            msg = "недопустимое значение: " + msg.replace("Input should be", "допустимо")
        elif "valid" in msg:
            msg = "неверный тип значения"
        out.append(f"«{field_name}» – {msg}")
    return ("[invalid_request] Неверные аргументы: " + ("; ".join(out) or text[:200]) +
            ". Исправь и вызови один раз; не повторяй тот же вызов.")


REJECTED = ("[rejected] Пользователь нажал «Отклонить» – ничего не изменено. Не повторяй вызов; коротко скажи, "
            "что изменение отменено.")
NOT_ASKED = ("[not_requested] Пользователь не просил ничего сохранять, создавать или удалять – изменяющие "
             "инструменты вызывай только по его прямой просьбе. Ничего не создано. Ответь на вопрос; сохранить "
             "можно предложить одной строкой в конце ответа.")
# просьба что-то записать: в вопросе (и в дописанном по ходу) или «да» на предложение модели
WRITE_ASK = re.compile(r"сохран|созда|запиш|удал|убер|сотри|стер|очист|помет|отмет|переимен|измен|поменя|исправ|"
                       r"добав|подборк|черновик|импорт|скопир|копи", re.IGNORECASE)
YES = re.compile(r"^\s*(да|давай|ок|окей|хорошо|конечно|согласен|сделай|делай|выполняй|можно|ага|угу|\+)\b",
                 re.IGNORECASE)
REPEAT_ASK = re.compile(r"^\s*(повтори\w*|ещё раз|еще раз|снова|заново|попробуй\w*(\s+\w+)?\s+(ещё|еще|снова)|"
                        r"давай ещё|давай еще)\b", re.IGNORECASE)
COMPACT_ASK = re.compile(r"^\s*(сожми|сжать|сократи|ужми|компакт)\w*\s+(весь\s+)?(контекст|историю|диалог|чат|память)"
                         r"\s*[.!]*\s*$", re.IGNORECASE)


def write_asked(msgs: list) -> bool:
    """Просил ли пользователь в текущем ходе что-то изменить. Короткое «да» – если модель перед этим
    предлагала изменение."""
    last = max((i for i, m in enumerate(msgs) if is_question(m)), default=-1)
    if last < 0:
        return False
    said = [_shown(m) for m in msgs[last:] if isinstance(m, HumanMessage) and not _meta(m).get("check")]
    if any(WRITE_ASK.search(t) for t in said):
        return True
    q = said[0].strip()
    # «повтори» после неудачной записи («ИР занят сервисом») – та же просьба, что в прошлом вопросе
    if len(q) < 40 and REPEAT_ASK.match(q):
        before = next((m for m in reversed(msgs[:last]) if is_question(m)), None)
        if before is not None and WRITE_ASK.search(_shown(before)):
            return True
    prev = next((m for m in reversed(msgs[:last]) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return len(q) < 40 and bool(YES.match(q)) and prev is not None and bool(WRITE_ASK.search(_text(prev.content)))


CLAIM = re.compile(r"сохран[её]н|создан[аоы]?\b|удал[её]н|записан|выполнен[аоы]?\b", re.IGNORECASE)
HONEST = re.compile(r"не (удалось|получилось|выполн|сохран|создан|удал)|отмен|отклон|ошибк", re.IGNORECASE)
NOT_DONE = "\n\n⚠ **Проверка агента: изменение не выполнено** – {why}. Ничего не сохранено и не удалено."


class WriteClaimMiddleware(AgentMiddleware):
    """Модель написала «подборка сохранена», хотя запись вернула ошибку (выдуманные пути – «Предпросмотр не
    создан»). Такое не доверяем модели: если в ходе была попытка записи и ни одна не удалась, а ответ говорит
    «сохранено / создано / удалено», к ответу дописывается предупреждение (и в историю тоже)."""

    def __init__(self, write_tools: list[str]):
        super().__init__()
        self.write = set(write_tools)

    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        msgs = _messages(state)
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage) or last.tool_calls or not CLAIM.search(text := _text(last.content)) \
                or HONEST.search(text):                  # модель сама сказала, что не вышло
            return None
        start = max((i for i, m in enumerate(msgs) if is_question(m)), default=0)
        tries = [m for m in msgs[start:] if isinstance(m, ToolMessage) and m.name in self.write]
        if not tries or any(m.status != "error" and '"status":"done"' in _text(m.content) for m in tries):
            return None
        why = re.sub(r"^Error executing tool \w+:\s*", "", _text(tries[-1].content)).split(". Пути")[0][:200]
        fixed = last.model_copy(update={"content": text + NOT_DONE.format(why=why.rstrip("."))})
        return {"messages": [fixed]}                     # тот же id – сообщение заменяется


VERIFY_NOTE = ("[Проверка агента (код): {items}. Перепиши ответ целиком, все разделы: газы и числа – только "
               "кусками gas_query (метки block); не хватает данных – вызови gas_query ещё раз. О проверке не "
               "пиши.]")
# переспрос («укажите, какая смесь») на вопрос о базе, когда в базу ещё не смотрели
CLARIFY = re.compile(r"уточните|укажите|какую именно|какой именно|какая именно|о какой|о каком|какую смесь|"
                     r"какой газ вы|нужно знать|не могли бы вы", re.IGNORECASE)
BASE_ASK = re.compile(r"смес|газ|марк|сорт|примес|аргон|гели|азот|кислород|водород|неон|ксенон|криптон|метан|"
                      r"двуокис|углекисл|ацетилен|пропан|чистот", re.IGNORECASE)
ASK_FIRST = ("[Проверка агента (код): ты переспросил пользователя, не посмотрев базу. Сначала вызови инструмент "
             "(смеси – gas_query(op=\"mixtures\"), газы – gas_query(op=\"list\", gases=…)). Подходит один "
             "вариант – отвечай по нему; переспрашивай, только если в базе их несколько, и перечисли их. О проверке "
             "не пиши.]")


class VerifyMiddleware(AgentMiddleware):
    """Итоговый ответ проверяет код (check.py): куски gas_query подставляются по меткам, в тексте модели вне кусков –
    ни газов базы, ни чисел; вопрос о газах – только через запрос; запрос сверяется с вопросом по явным признакам.
    Ошибка – модель до двух раз переписывает ответ по списку замечаний; не помогло – показываются только куски кода
    (текст модели – черновиком), красная строка. Итог – в метаданных ответа (iacpaas.verify)."""

    def __init__(self, runtime: "AgentRuntime", retries: int = 2):
        super().__init__()
        self.rt, self.retries = runtime, retries

    def _emit(self, event: dict[str, Any]) -> None:
        from .trace import _thread
        sid = _thread(None)
        if emit := self.rt._emit.get(sid):
            emit({"event": "verify", "session": sid, **event})

    async def _follow(self, question: str, target: tuple | None, got: dict, results_turn: list, args_of: dict,
                      fixes: list[dict]) -> list[dict]:
        """Уточнение к прошлому вопросу target (check.follow_target): газы кусков ответа не из газов того вопроса –
        замечание (в got["remarks"]) и исправленный запрос – отбор того вопроса + своё действие (check.follow_fix). Тот
        вопрос был по всей базе – проверять нечего."""
        if not target or not (scope := check.follow_fix({}, target[1])):
            return fixes
        allowed = set(_row_names(await self.rt.call_gas({"op": "list", **scope})))
        if not allowed:
            return fixes
        rows = {d["block"]: set(_row_names(d)) for n, d in results_turn
                if n == check.GAS_TOOL and isinstance(d.get("block"), str)}
        bad = [k for k in got["used"] if (args_of.get(k) or {}).get("op") != "calc" and rows.get(k)
               and not rows[k] <= allowed]
        if not bad:
            return fixes
        out = [dict(f) for f in fixes]
        for k in bad:
            same = next((f for f in out if f.get("old") == k and not f.get("drop")), None)
            if not same:
                same = {"old": k, "args": args_of[k], "part": question}
                out.append(same)
            same.pop("use", None)                    # запрос и так чинится – в рамках газов того вопроса
            same["args"] = check.follow_fix(same["args"], target[1])
            # запасной запрос – только действие и отбор того вопроса (если в полном что-то не примет сервер)
            same["bare"] = check.follow_fix({x: v for x, v in same["args"].items()
                                             if x in ("op", "by", "order", "top", "show", "unit")}, target[1])
        got["remarks"].append((f"это уточнение к вопросу «{target[0][:120]}» – отвечай только среди его газов",
                               f"уточнение к вопросу «{target[0][:120]}», а газы ответа не из него"))
        return out

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        msgs = _messages(state)
        last = msgs[-1] if msgs else None
        start = max((i for i, m in enumerate(msgs) if is_question(m)), default=0)
        turn = msgs[start:]
        results_all, results_turn = check.tool_results(_ok_tools(msgs)), check.tool_results(_ok_tools(turn))
        # «сохрани эти газы» – модель передала метку куска {{#id}} вместо газов: газы этого куска подставляет код прямо
        # в вызов (до выполнения) – в истории и трассе видно, что на самом деле сохранено
        if isinstance(last, AIMessage) and last.tool_calls:
            fixed_calls = [{**tc, "args": f} if (f := marks_to_names(tc.get("args") or {}, msgs)) else tc
                           for tc in last.tool_calls]
            if fixed_calls != last.tool_calls:
                return {"messages": [last.model_copy(update={"tool_calls": fixed_calls})]}
        # модель по кругу повторяет тот же запрос (после «[stop]») – ход кончается: ответ из полученных кусков, запрос
        # дальше исправляет код (как после последнего повтора)
        looped = isinstance(last, AIMessage) and bool(last.tool_calls) and all(
            tc["name"] == check.GAS_TOOL for tc in last.tool_calls) and any(
            isinstance(m, ToolMessage) and _text(m.content).startswith("[stop]") for m in turn)
        if looped:
            got_blocks = list(dict.fromkeys(d["block"] for n, d in results_turn if n == check.GAS_TOOL
                                            and isinstance(d.get("block"), str)))
            # кусков нет (все вызовы с ошибкой) – тоже конец хода: дальше – как ответ без куска (строгий итог)
            # вызов убирается целиком: и tool_calls, и function_call GigaChat – иначе в истории остаётся вызов без
            # результата, и GigaChat отвечает 422 на каждый следующий вопрос диалога
            last = last.model_copy(update={"content": "\n\n".join(got_blocks) or "Так по базе посчитать не могу.",
                                           "tool_calls": [], "additional_kwargs": no_call(last)})
        if not isinstance(last, AIMessage) or last.tool_calls or not (text := _text(last.content)).strip() \
                or text in (MODEL_BUSY, MODEL_429):
            return None
        # сбой модели («Model call failed after 3 attempts with ReadTimeout») – не ответ; куски кода хода – человеку
        if m := FAILED.match(text.strip()):
            got = "\n\n".join(d["answer"] for n, d in results_turn if n == check.GAS_TOOL and d.get("answer"))
            note = FAILED_NOTE.format(err="таймаут" if "Timeout" in m[2] else m[2], n=m[1],
                                      tail=". Ниже – ответ, собранный кодом из базы без ИИ:\n\n" + got if got else ".")
            self._emit({"status": "done", "skipped": True})
            return {"messages": [last.model_copy(update={"content": note, "response_metadata": {
                **(last.response_metadata or {}), "iacpaas": {**_meta(last), "verify": {"skipped": True}}}})]}
        tries = sum(1 for m in turn if isinstance(m, HumanMessage) and _meta(m).get("check"))
        if looped:
            tries = self.retries                     # круг повторов: больше не спрашивать модель – итог сейчас
        question = " ".join(_shown(m) for m in turn if isinstance(m, HumanMessage) and not _meta(m).get("check"))
        # «Какая доля аргона в смеси?» – GigaChat переспрашивал «какая смесь?», не заглянув в базу, а смесь там одна
        if tries < self.retries and not any(isinstance(m, ToolMessage) for m in turn) \
                and CLARIFY.search(text) and BASE_ASK.search(question):
            self._emit({"status": "retry", "problems": ["переспрос без просмотра базы"]})
            note = HumanMessage(content=ASK_FIRST, response_metadata={"iacpaas": {"ts": time.time(), "check": True}})
            meta = {**(last.response_metadata or {})}
            meta["iacpaas"] = {**meta.get("iacpaas", {}), "draft": True}
            return {"messages": [last.model_copy(update={"response_metadata": meta}), note], "jump_to": "model"}
        # GigaChat пишет ответ кусками: «### 1. …» – вызов инструмента – «### 2. …»; куски этой попытки – в начало
        pieces = [m for m in _pieces(turn) if _text(m.content).strip() not in text]
        text = join_pieces(pieces, text)
        moved = [m.model_copy(update={"content": ""}) for m in pieces]
        # повтор по замечанию: модель переписала только ошибочный раздел – остальные из прошлого черновика
        if tries and (prev := next((_text(m.content) for m in reversed(turn)
                                    if isinstance(m, AIMessage) and _meta(m).get("draft")), "")):
            text = merge_sections(prev, text)
        calls = [(tc["name"], tc.get("args") or {}) for m in turn if isinstance(m, AIMessage)
                 for tc in m.tool_calls or []]
        names, grades = await self.rt.gas_names()
        before = msgs[:start]
        asked = " ".join(_shown(m) for m in before if isinstance(m, HumanMessage) and not _meta(m).get("check"))
        said = " ".join(_text(m.content) for m in before if isinstance(m, AIMessage) and not m.tool_calls
                        and not _meta(m).get("draft"))
        dialog, args_of = bool(DIALOG_ASK.search(question)), block_args(turn)
        # уточнение к прошлому вопросу: смысл запроса сверяется без ссылки «вернёмся к …» – отбор газов того вопроса
        # проверяет _follow
        target = check.follow_target(question, follow_prev(before))
        plan_q = check.follow_part(question) if target else question
        got = check.check(text, question, results_all, results_turn, calls, names, grades, wrote(turn),
                          asked, said, dialog, args_of, plan_q)
        # исправить запрос можно, если ошибок связки нет: запрос не по вопросу или запрос упал и куска нет
        failed = all("не удался" in p for p in got["problems"])
        fixes = check.corrections(plan_q, [(k, args_of.get(k, {})) for k in got["used"]], list(args_of.items())) \
            if (got["remarks"] or got["problems"]) and failed else []
        # уточнение («а какой из них…», «вернёмся к …») – газы ответа должны быть из газов того прошлого вопроса; и при
        # ошибках связки: иначе в строгом итоге (только куски кода) остался бы кусок не тех газов
        fixes = await self._follow(question, target, got, results_turn, args_of, fixes)
        fixed: list[str] = []
        if fixes and tries:
            # модель не исправила запрос по замечанию – исправленный запрос выполняет код и ставит его кусок
            draft = text
            if not check.MARK.search(text):
                text = ""                                # «не могу» без куска – ответом станут куски кода
            have = {d["block"]: d for n, d in results_turn if n == check.GAS_TOOL and isinstance(d.get("block"), str)}
            for f in fixes:
                if f.get("drop"):                        # лишний кусок (число калькулятором) – убрать
                    if f["old"] in text:
                        text = text.replace(f["old"], "")
                        fixed.append(f"«{f['part']}» – лишний кусок калькулятора убран")
                    continue
                # верный запрос модель уже делала – его кусок; иначе исправленный запрос выполняет код
                d = have.get(f.get("use") or "") or await self.rt.call_gas(f["args"])
                if not d.get("block") and f.get("bare"):
                    f = {**f, "args": f["bare"]}
                    d = await self.rt.call_gas(f["args"])
                if not d.get("block"):
                    continue
                text = text.replace(f["old"], d["block"]) if f["old"] and f["old"] in text \
                    else text + "\n\n" + d["block"]
                results_all, results_turn = results_all + [(check.GAS_TOOL, d)], results_turn + [(check.GAS_TOOL, d)]
                calls, args_of = calls + [(check.GAS_TOOL, f["args"])], {**args_of, d["block"]: f["args"]}
                fixed.append(f"«{f['part']}» – {d.get('understood', '')}")
            if fixed:
                got = check.check(text, question, results_all, results_turn, calls, names, grades, wrote(turn),
                                  asked, said, dialog, args_of, plan_q)
            else:
                text = draft
            fixes = []
        # несоответствие запроса вопросу – одна просьба (с готовым исправленным запросом); осталось – код исправил
        # запрос сам, а что исправить нельзя – «возможно, понял вопрос не так»
        items = got["problems"] + ([m for m, _ in got["remarks"]] + [fix_note(f) for f in fixes] if not tries else [])
        meta = {**(last.response_metadata or {})}
        meta["iacpaas"] = {**meta.get("iacpaas", {}), "raw": text}
        rep: dict[str, Any] = {"source": got["source"], "pieces": got["pieces"], "understood": got["understood"],
                               "blocks": got["blocks"], "checked": got["checked"], "problems": got["problems"],
                               "unverified": [h for _, h in got["remarks"]], "retried": tries,
                               "ok": [f"«{blocks_title(p)}» – газы и числа вставил код из базы, ИИ их не писал"
                                      for p in got["pieces"]] + got.get("matched", [])}
        if fixed:
            rep["fixed"] = fixed
        # версия данных (дата изменения ИР на платформе) – по каким данным посчитан ответ
        rep["versions"] = sorted({str(d["data_version"]) for n, d in results_turn
                                  if n == check.GAS_TOOL and d.get("data_version")})
        if items and tries < self.retries:
            self._emit({"status": "retry", "problems": items})
            meta["iacpaas"] = {**meta["iacpaas"], "draft": True, "verify": rep}
            note = HumanMessage(content=VERIFY_NOTE.format(items="; ".join(items[:8])),
                                response_metadata={"iacpaas": {"ts": time.time(), "check": True}})
            return {"messages": [*moved, last.model_copy(update={"content": text, "response_metadata": meta}), note],
                    "jump_to": "model"}
        final = got["text"]
        if got["problems"]:
            # ошибка осталась: человеку – только то, что собрал код; текст ИИ – черновиком под кнопкой
            rep["strict"], rep["draft"] = True, got["text"]
            final = "\n\n".join(got["pieces"])
            # меток модель так и не вставила – куски запросов этого хода (их собрал код) вместо пустого ответа
            own = list(dict.fromkeys(d["block"] for n, d in results_turn if check.is_piece(d)))
            if not final and own:
                alt = check.check("\n\n".join(own), question, results_all, results_turn, calls, names, grades,
                                  wrote(turn), asked, said, dialog, args_of, plan_q)
                final = alt["text"]
                rep.update(pieces=alt["pieces"], understood=alt["understood"], blocks=alt["blocks"],
                           checked=alt["checked"], unverified=[h for _, h in alt["remarks"]])
            final = final or STRICT_EMPTY
        elif got["source"] == "model":
            rep["skipped"] = True                    # написал ИИ, данных базы в ответе нет
        # человек отклонил запись, а модель об этом не сказала – говорит код
        if rejected(turn) and not re.search(r"отмен|отклон|не сохран|не выполн|не изменен", final, re.I):
            final = "**Изменение отклонено – ничего не сохранено.**\n\n" + final
        # запись выполнена, а модель об этом не сказала (показала только газы) – подтверждение пишет код
        elif (done := done_writes(turn)) and not re.search(r"сохран|создан|удал|записан|готово", final, re.I):
            final = "**Готово: " + "; ".join(f"{WRITE_DONE.get(op, 'изменено')} – «{t}»" for op, t in done) \
                + ".**\n\n" + final
        # запросы, чьи куски в ответе (и исправленные кодом) – по ним следующее уточнение «а из них …»
        rows = {d["block"]: _row_names(d) for n, d in results_turn
                if n == check.GAS_TOOL and isinstance(d.get("block"), str)}
        queries = [{"args": args_of[k], "gases": rows.get(k, [])} for k in got["used"] if k in args_of]
        meta["iacpaas"] = {**meta["iacpaas"], "verify": rep, "queries": queries}
        self._emit({"status": "done", **rep})
        return {"messages": [*moved, last.model_copy(update={"content": final, "response_metadata": meta})]}


def block_args(turn: list) -> dict[str, dict]:
    """Метка куска gas_query → аргументы вызова, который его дал (по tool_call_id)."""
    args = {tc.get("id"): tc.get("args") or {} for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls or []}
    out = {}
    for name, d in check.tool_results([(m.tool_call_id, _text(m.content)) for m in turn
                                       if isinstance(m, ToolMessage) and m.name == check.GAS_TOOL
                                       and m.status != "error"]):
        if isinstance(d.get("block"), str) and name in args:
            out[d["block"]] = args[name]
    return out


def fix_note(f: dict) -> str:
    """Исправление запроса – просьбой модели (первый заход)."""
    if f.get("drop"):
        return f"убери метку {f['old']}: число из куска не печатай калькулятором"
    if f.get("use"):
        return f"на «{f['part']}» вставь метку {f['use']} – этот запрос ты уже сделал верно"
    return f"вызови ровно {check.call_text(f['args'])}"


def _ok_tools(ms: list) -> list[tuple[str, str]]:
    """Удачные результаты инструментов: (инструмент, текст)."""
    return [(m.name or "", _text(m.content)) for m in ms if isinstance(m, ToolMessage) and m.status != "error"]


STRICT_EMPTY = ("**Ответ по базе не собран:** ИИ не сделал запрос к базе или написал данные сам – такой ответ не "
                "показывается. Спросите ещё раз, точнее (что именно и о каких газах).")


def blocks_title(piece: str) -> str:
    """Заголовок куска для человека: первая строка без разметки."""
    first = next((ln for ln in piece.splitlines() if ln.strip()), "")
    t = re.sub(r"[*_#|]", "", first).strip().rstrip(":").strip()
    return t[:90] + ("…" if len(t) > 90 else "")


PIECE = re.compile(r"^#{2,4}\s|\d", re.M)                 # кусок ответа – раздел или числа, а не «сейчас посмотрю»


HEAD = re.compile(r"^\s*#{1,6}\s*")


SECTION = re.compile(r"^\s*#{1,4}\s*(?:\S+\s+)??(\d{1,2})[.)]\s", re.M)


def _sections(text: str) -> tuple[str, dict[str, str]]:
    """Текст до первого раздела и разделы «### N. …» по номеру; повтор заголовка («### 3.» и «#### 3.») – тот же
    раздел."""
    heads = list(SECTION.finditer(text))
    out: dict[str, str] = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[h[1]] = (out.get(h[1], "") + "\n\n" + text[h.start():end].strip()).strip()
    return (text[:heads[0].start()].strip() if heads else text.strip()), out


def merge_sections(prev: str, new: str) -> str:
    """Повтор ответа разделами: разделы нового текста заменяют разделы прошлого с тем же номером, остальные – из
    прошлого, по порядку номеров. Новый ответ без разделов или со всеми разделами – как есть."""
    _, old = _sections(prev)
    head, fresh = _sections(new)
    if len(old) < 2 or not fresh or set(old) <= set(fresh):
        return new
    nums = sorted(set(old) | set(fresh), key=int)
    return "\n\n".join(([head] if head else []) + [fresh.get(n) or old[n] for n in nums])


def _lead(line: str) -> tuple[bool, set[str]] | None:
    """Строка-подзаголовок («### 2. …» или «Лучшие по водороду:») – (заголовок ли с #, основы слов), иначе None.
    Жирный «**По водороду лучшие:**» – тоже."""
    bold = bool(re.fullmatch(r"\*\*[^*\n]{1,80}\*\*:?", line.strip()))     # «**Лучше всего по водороду**» без «:»
    line = line.strip().strip("*_` ")
    if not (HEAD.match(line) or line.endswith(":") or bold):
        return None
    return bool(HEAD.match(line)), {w[:4] for w in re.findall(r"[^\W\d_]+", HEAD.sub("", line).lower())}


def join_pieces(pieces: list, text: str) -> str:
    """Куски ответа – в начало итогового; заголовок, который модель повторила в итоге («### 4. …» в куске и «#### 4.
    …» в ответе), – один раз. Подзаголовок в конце куска, а следующий
    кусок начинается с похожего («Лучшие по водороду:» – вызов – «Лучше всего по водороду:»), – тоже один раз."""
    heads = {HEAD.sub("", ln).strip() for ln in text.splitlines() if HEAD.match(ln)}
    chunks = [kept for m in pieces
              if (kept := "\n".join(ln for ln in _text(m.content).splitlines()
                                    if not (HEAD.match(ln) and HEAD.sub("", ln).strip() in heads)).strip())]
    if not chunks:
        return text
    chunks.append(text.strip())
    for i in range(len(chunks) - 1):
        lines = chunks[i].splitlines()
        tail, nxt = _lead(lines[-1]), _lead(chunks[i + 1].lstrip().split("\n", 1)[0])
        # «### 4. … по метану» и следующее «Метан:» – разные уровни, не повтор
        if tail and nxt and tail[0] == nxt[0] and tail[1] and nxt[1] \
                and len(tail[1] & nxt[1]) >= 0.6 * min(len(tail[1]), len(nxt[1])):
            chunks[i] = "\n".join(lines[:-1]).strip()
    return "\n\n".join(c for c in chunks if c)


def _pieces(turn: list) -> list[AIMessage]:
    """Тексты рядом с вызовами инструментов в последней попытке хода (после вопроса или замечания проверки), кроме
    ответов на сообщение, дописанное по ходу работы (steer), – это части итогового ответа."""
    start = max((i for i, m in enumerate(turn) if isinstance(m, HumanMessage) and not _meta(m).get("steer")),
                default=0)
    out, steered = [], False
    for m in turn[start + 1:-1]:
        if isinstance(m, HumanMessage):
            steered = steered or bool(_meta(m).get("steer"))
        elif isinstance(m, AIMessage):
            if m.tool_calls and not steered and PIECE.search(t := _text(m.content)) and t.strip():
                out.append(m)
            steered = False
    return out


def wrote(turn: list) -> bool:
    """В ходе запись выполнена или отклонена человеком (WriteResult): ответ – подтверждение «сохранено / отменено»."""
    for m in turn:
        if isinstance(m, ToolMessage) and m.status != "error":
            try:
                r = json.loads(_text(m.content))
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("operation") and r.get("status") in ("done", "rejected"):
                return True
    return False


WRITE_DONE = {"save_selection": "подборка сохранена", "save_mixture": "черновик смеси создан",
              "delete_from_fund": "удалено", "import_inforesource": "ИР создан", "update_concept": "вершина изменена"}


def done_writes(turn: list) -> list[tuple[str, str]]:
    """Выполненные в ходе записи: (операция, куда)."""
    out = []
    for m in turn:
        if isinstance(m, ToolMessage) and m.status != "error":
            try:
                r = json.loads(_text(m.content))
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("operation") and r.get("status") == "done":
                out.append((str(r["operation"]), str(r.get("target") or "")))
    return out


def rejected(turn: list) -> bool:
    """В ходе человек нажал «Отклонить»."""
    for m in turn:
        if isinstance(m, ToolMessage):
            text = _text(m.content)
            if text.startswith("[rejected]") or '"status":"rejected"' in text.replace(" ", ""):
                return True
    return False


COUNT = {"два": 2, "две": 2, "оба": 2, "обе": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
         "восемь": 8, "девять": 9, "десять": 10}
SEPARATE = re.compile(r"разн\w*|отдельн\w*|кажд\w*|по одному|по одной|поодиночке|порознь", re.IGNORECASE)
COUNT_ASK = re.compile(r"\b(\d{1,2}|" + "|".join(COUNT) + r")\s+(?:\w+\s+)?газ", re.IGNORECASE)


def too_few(tc: dict, msgs: list) -> str:
    """«Сохрани эти два газа»: GigaChat сохранял по одному газу в разные подборки. Сколько газов просил человек,
    столько путей должно быть в одном вызове – иначе понятная ошибка до записи."""
    els = split_items((tc.get("args") or {}).get("elements"))
    last = max((i for i, m in enumerate(msgs) if is_question(m)), default=-1)
    if not els or last < 0 or not (m := COUNT_ASK.search(q := _shown(msgs[last]))):
        return ""
    if SEPARATE.search(q):                               # «в разные подборки», «каждый отдельно» – по одному можно
        return ""
    n = int(m.group(1)) if m.group(1).isdigit() else COUNT[m.group(1).lower()]
    if len(els) >= n:
        return ""
    return (f"[invalid_request] Пользователь просил {n} газа(ов), а в elements {len(els)}. Передай ВСЕ {n} газа "
            "одной строкой через «;» в одном вызове, например elements=\"Аргон 4.8; Аргон 6.0\" – получится одна "
            "подборка. Ничего не создано.")


FROM_BASE = re.compile(r"из\s+(?:выданн\w+\s+|основн\w+\s+)?баз\w*(?:\s+газ\w*)?", re.I)
OWN_PLACE = re.compile(r"песочниц|мо[ейюя]\s+папк|сво[ейюя]\s+(?:папк|подборк)|подборк", re.I)
BASE_READ_ONLY = ("[read_only] Пользователь просит удалить из выданной базы газов – она только для чтения, менять её "
                  "нельзя. Ничего не удаляй (и своё в «Песочнице» тоже) – так и скажи; можно предложить сохранить "
                  "подборку без этого газа в свою папку.")


def marks_to_names(args: dict, msgs: list) -> dict | None:
    """Метки кусков gas_query в аргументах записи («elements": "{{#5377}}") → названия газов этого куска."""
    rows = {d["block"]: [r["name"] for r in d.get("rows") or [] if r.get("name")]
            for _, d in check.tool_results(_ok_tools(msgs)) if isinstance(d.get("block"), str)}
    out, changed = {}, False
    for k, v in args.items():
        if isinstance(v, str) and check.MARK.search(v):
            v = check.MARK.sub(lambda m: "; ".join(rows.get("{{" + m[1] + "}}", [])), v).strip("; ")
            changed = True
        out[k] = v
    return out if changed else None


class ConfirmButtonMiddleware(AgentMiddleware):
    """Запись – кнопками: предпросмотр сервера сразу уходит человеку карточкой «Разрешить / Отклонить»,
    после «Разрешить» тот же вызов выполняется с токеном сервера. Спрашивать «да» текстом модели не нужно.
    Без просьбы пользователя (write_asked) изменяющий вызов не выполняется: модель сама сохраняла подборки
    посреди анализа. При продолжении после кнопки узел инструментов выполняется заново: предпросмотр
    запрашивается ещё раз (только чтение), interrupt() сразу возвращает решение."""

    def __init__(self, write_tools: list[str]):
        super().__init__()
        self.write = set(write_tools)

    async def awrap_tool_call(self, request, handler):
        tc = request.tool_call
        if tc["name"] not in self.write or (tc.get("args") or {}).get("confirm_token"):
            return await handler(request)
        msgs = _messages(request.state)
        if not write_asked(msgs):
            return ToolMessage(content=NOT_ASKED, tool_call_id=tc["id"], name=tc["name"], status="error")
        # «удали аргон 6.0 из базы газов» – выданная база только для чтения; своё в «Песочнице» не трогать
        ask = next((_shown(m) for m in reversed(msgs) if isinstance(m, HumanMessage) and not _meta(m).get("check")), "")
        if tc["name"] == "delete_from_fund" and FROM_BASE.search(ask) and not OWN_PLACE.search(ask):
            return ToolMessage(content=BASE_READ_ONLY, tool_call_id=tc["id"], name=tc["name"], status="error")
        if short := too_few(tc, _messages(request.state)):
            return ToolMessage(content=short, tool_call_id=tc["id"], name=tc["name"], status="error")
        res = await handler(request)
        try:
            data = json.loads(_text(res.content)) if isinstance(res, ToolMessage) else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict) or data.get("status") != "pending_confirmation":
            return res
        answer = interrupt({"action_requests": [{"name": tc["name"], "args": data.get("preview") or {},
                                                 "description": "Агент хочет ИЗМЕНИТЬ данные"}]})
        decision = ((answer or {}).get("decisions") or [{}])[0]
        if decision.get("type") != "approve":
            return ToolMessage(content=REJECTED, tool_call_id=tc["id"], name=tc["name"])
        args = {**tc["args"], "confirm_token": data["confirm_token"]}
        return await handler(request.override(tool_call={**tc, "args": args}))


class SteerMiddleware(AgentMiddleware):
    """Перед каждым шагом модели забирает сообщения, дописанные пользователем во время работы агента,
    и добавляет их в диалог – модель учитывает их сразу, не дожидаясь конца ответа."""

    def __init__(self, runtime: "AgentRuntime"):
        super().__init__()
        self.rt = runtime

    async def abefore_model(self, state, runtime) -> dict[str, Any] | None:
        from .trace import _thread
        sid = _thread(None)
        items = self.rt._inbox.pop(sid, None) if sid else None
        if not items:
            return None
        if emit := self.rt._emit.get(sid):
            emit({"event": "steer", "session": sid, "items": items})
        return {"messages": [HumanMessage(content=STEER_NOTE + it["text"], response_metadata={"iacpaas": {
            "ts": time.time(), "steer": True}}) for it in items]}


MODEL_BUSY = ("Модель занята другим окном (консоль или веб-чат на том же ключе): ключ принимает один запрос за "
               "раз, а ответ там ещё не готов. Повторите вопрос, когда другое окно закончит.")
MODEL_429 = ("Модель ответила «занята» (429): ключ принимает один запрос за раз, а сейчас его использует другая "
             "программа. Повторите вопрос через минуту.")


def retry_on(trace: JsonlTrace, e: BaseException) -> bool:
    """Повторять ли шаг модели после ошибки (429 – нет, сразу понятный ответ); каждая ошибка – в трассу."""
    from .trace import _thread
    again = not rate_limited(e) and default_retry_on(e)
    trace.emit({"event": "model_retry", "session": _thread(None), "error": f"{type(e).__name__}: {e}"[:500],
                "again": again})
    # повтор – с новым X-Session-ID (без кэша GigaChat): зависания ReadTimeout / ServerError шли подряд на одном и том
    # же запросе переписывания, а id кэша у всех диалогов общий
    if again:
        try:
            from gigachat.context import session_id_cvar
            session_id_cvar.set(f"iacpaas-retry-{uuid.uuid4().hex[:12]}")
        except ImportError:
            pass
    return again


def rate_limited(e: BaseException | None) -> bool:
    """429 от провайдера (GigaChat RateLimitError, OpenAI-совместимые) – и в цепочке причин."""
    while e is not None:
        if getattr(e, "status_code", None) == 429 or type(e).__name__ == "RateLimitError":
            return True
        e = e.__cause__ or e.__context__
    return False


class ModelGate(AgentMiddleware):
    """Очередь к модели: не больше N запросов одновременно (parallel в models.toml). Внутри процесса – семафор
    (два диалога в веб-чате), между процессами – файловая блокировка (консоль и веб-чат на одном ключе GigaChat:
    раньше второй получал 429 и минуты повторов). Блокировка – на один шаг модели: окна идут по очереди, пока
    одно ждёт платформу, модель свободна для другого. Ждём не дольше wait_s, затем понятный ответ, а не 429."""

    def __init__(self, runtime: "AgentRuntime", sem: asyncio.Semaphore, shared: SharedLock | None = None,
                 wait_s: float = 150.0):
        super().__init__()
        self.rt, self.sem, self.shared, self.wait_s = runtime, sem, shared, wait_s

    def _notify(self, request, text: str) -> None:
        from .trace import _thread
        sid = _thread(request)
        if emit := self.rt._emit.get(sid):
            emit({"event": "model_wait", "session": sid, "text": text})

    async def awrap_model_call(self, request, handler):
        if self.sem.locked():
            self._notify(request, "Жду очереди к модели (её занял другой диалог)")
        async with self.sem:
            if self.shared and not self.shared.try_acquire():
                self._notify(request, "Модель занята другим окном (консоль или веб-чат) – жду очереди")
                deadline = time.monotonic() + self.wait_s
                while not self.shared.try_acquire(waiting=True):
                    if time.monotonic() > deadline:
                        return AIMessage(content=MODEL_BUSY)
                    await asyncio.sleep(0.2)
            try:
                return await handler(request)
            except Exception as e:
                if rate_limited(e):                       # повторы 429 выключены (retry_on) – сразу понятный ответ
                    return AIMessage(content=MODEL_429)
                raise
            finally:
                if self.shared:
                    self.shared.release()


class SessionCacheMiddleware(AgentMiddleware):
    """Кэш контекста GigaChat: постоянный заголовок X-Session-ID для профиля модели. Начало запроса, совпавшее
    с прошлым запросом этой сессии, берётся из кэша и не оплачивается (precached_prompt_tokens). Инструкция
    и схемы инструментов (~5 тыс. токенов) одинаковы во всех диалогах – с общим id они из кэша даже в первом
    вопросе нового диалога (проба: 21 новый токен, 3 156 из кэша); с id диалога первый вопрос шёл без кэша
    (5 036 платных), без заголовка – кэш через раз.
    https://developers.sber.ru/docs/ru/gigachat/guides/keeping-context"""

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        try:
            from gigachat.context import session_id_cvar
        except ImportError:                       # библиотеки GigaChat нет – прослойка ничего не делает
            session_id_cvar = None
        self.cvar = session_id_cvar

    async def awrap_model_call(self, request, handler):
        if self.cvar is None:
            return await handler(request)
        token = self.cvar.set(self.key)
        try:
            return await handler(request)
        finally:
            self.cvar.reset(token)


async def _agent_confirms(*_: Any) -> None:
    """Не вызывается: сессии он не передаётся (см. AgentRuntime.start), только отмечает клиента для адаптера."""
    raise RuntimeError("окно подтверждения сервера агенту не нужно")


def tool_kind(tool) -> str:
    """read – чтение платформы, write – изменение, local – без обращения к платформе."""
    ann = tool_annotations(tool)
    if not ann.get("read_only_hint", False):
        return "write"
    return "read" if ann.get("open_world_hint", True) else "local"


class AgentRuntime:
    def __init__(self, settings: AgentSettings | None = None, *, mcp_target: Any = None):
        self.s = settings or AgentSettings.from_env()
        self._target_override = mcp_target
        self._stack = AsyncExitStack()
        self.tools: list = []
        self.checkpointer: AsyncSqliteSaver | None = None
        self.sessions = SessionStore(self.s.sessions_file)
        self._agents: dict[str, tuple[Any, JsonlTrace]] = {}
        self._traces: dict[str, JsonlTrace] = {}
        self._gas_names: tuple[set[str], set[str]] | None = None
        self._models: dict[str, tuple[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._inbox: dict[str, list[dict[str, Any]]] = {}       # дописанное на ходу: session -> [{id, text}]
        self._emit: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._gates: dict[str, asyncio.Semaphore] = {}          # группа моделей -> очередь запросов
        self._shared: dict[str, SharedLock | None] = {}         # группа -> очередь между процессами
        self.system_prompt = DEFAULT_PROMPT

    # ------------------------------------------------------------ lifecycle
    def _mcp_target(self) -> Any:
        if self._target_override is not None:
            return self._target_override
        if self.s.mcp == "http":
            if not self.s.mcp_token:
                raise RuntimeError("AGENT_MCP=http: задайте MCP_BEARER_TOKEN (тот же, что у сервера)")
            return Client(self.s.mcp_url, auth=self.s.mcp_token, timeout=self.s.tool_timeout_s)
        if self.s.mcp == "inproc":
            from iacpaas_mcp.config import Settings
            from iacpaas_mcp.server import build_server, create_backend
            st = Settings.from_env()
            st.pieces = "marks"                          # куски по меткам подставляет VerifyMiddleware
            return build_server(st, create_backend(st))
        # stdio: сервер – дочерний процесс; его stderr пишется в файл, а не в консоль
        env = {k: v for k, v in os.environ.items() if k in CHILD_ENV_KEYS or k.startswith(CHILD_ENV_PREFIXES)}
        env["PYTHONIOENCODING"] = "utf-8"
        env["IACPAAS_PIECES"] = "marks"                  # куски по меткам подставляет VerifyMiddleware
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        transport = StdioTransport(sys.executable, ["-m", "iacpaas_mcp", "--transport", "stdio"], env=env,
                                   cwd=str(PROJECT_ROOT), log_file=log_dir / "server.stderr.log")
        return Client(transport, timeout=self.s.tool_timeout_s)

    async def start(self) -> "AgentRuntime":
        if self.s.prompt_file.is_file():
            self.system_prompt = self.s.prompt_file.read_text(encoding="utf-8")
        target = self._mcp_target()
        client = target if isinstance(target, Client) else Client(target, timeout=self.s.tool_timeout_s)
        # Запись подтверждает сам агент (кнопка «Разрешить», токен модели не виден) – окно сервера (MCP elicitation)
        # ему не нужно: LangChain объявил бы его за клиента без своего обработчика, и человека спрашивали бы дважды.
        # Отметка «обработчик есть» без передачи в сессию: адаптер клиент не трогает, сервер видит клиента без окон.
        client._elicitation_callback = _agent_confirms
        adapter = MCPAdapter(client)
        await self._stack.enter_async_context(adapter)
        self.tools = [t for t in await adapter.list_tools() if t.name not in self.s.hide_tools]
        self._tools_tok = {}                                     # схемы для шапки «контекст» – пересчитать
        self.s.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.checkpointer = await self._stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(self.s.db_file)))
        await self.sessions.open()
        self._stack.push_async_callback(self.sessions.close)
        return self

    async def close(self) -> None:
        await self._stack.aclose()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()

    # ---------------------------------------------------------------- agent
    def agent_for(self, model_name: str) -> tuple[Any, JsonlTrace, str]:
        """Агент на модель: инструкция, инструменты MCP, проверка ответа; чекпоинтер общий – диалог продолжается при
        смене модели."""
        name = resolve_name(model_name, self.s.models_file)
        if name not in self._agents:
            if name not in self._models:
                self._models[name] = make_model(name, self.s.models_file)
            name, model = self._models[name]
            trace = self._traces.setdefault(name, JsonlTrace(self.s.trace_file, name))
            self._agents[name] = (self._assemble(name, model, trace), trace)
        agent, trace = self._agents[name]
        return agent, trace, name

    def _assemble(self, name: str, model: Any, trace: JsonlTrace) -> Any:
        """Граф агента: модель, инструменты MCP, прослойки."""
        tools = list(self.tools)
        write_tools = [t.name for t in tools if is_write_tool(t)]
        # Человек подтверждает только реальное выполнение (вызов с confirm_token); предпросмотр безопасен.
        gate = InterruptOnConfig(allowed_decisions=["approve", "reject"],
                                 description="Агент хочет ИЗМЕНИТЬ данные",
                                 when=lambda req: bool((req.tool_call.get("args") or {}).get("confirm_token")))
        group, limit = parallel_limit(name, self.s.models_file)
        # один запрос за раз на ключ – и между процессами (консоль, веб-чат): файл data/model-<группа>.lock
        shared = SharedLock((self.s.lock_dir or self.s.db_file.parent) / f"model-{group}.lock") \
            if limit == 1 else None
        middleware = [ModelGate(self, self._gates.setdefault(group, asyncio.Semaphore(limit)),
                                self._shared.setdefault(group, shared))] if limit else []
        middleware += [
            trace,
            SessionCacheMiddleware(f"iacpaas-agent-{name}"),
            SteerMiddleware(self),
            ToolErrorMiddleware(on_error=tool_failure),
            RepeatGuardMiddleware(),
            ConfirmButtonMiddleware(write_tools),
            VerifyMiddleware(self),
            WriteClaimMiddleware(write_tools),
            # пауза перед повтором – 3 и 6 с; текст каждой ошибки – в трассу (model_retry): раньше повторы его
            # глотали, и причину «ServerError после 3 попыток» по трассе было не узнать
            ModelRetryMiddleware(max_retries=2, initial_delay=3.0, retry_on=partial(retry_on, trace)),
            ModelCallLimitMiddleware(run_limit=self.s.max_model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=self.s.max_tool_calls, exit_behavior="continue"),
            ContextMiddleware(self.s.keep_messages, context_budget(name, self.s.models_file), self),
        ]
        if write_tools:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on={n: gate for n in write_tools}))
        return create_agent(model, tools, system_prompt=self.system_prompt,
                            middleware=middleware, checkpointer=self.checkpointer, name="iacpaas_agent")

    async def gas_names(self) -> tuple[set[str], set[str]]:
        """Названия газов базы и номера марок в них («6.0», «1») – для проверки текста модели; один раз за запуск.
        Нет инструмента или базы – пусто (тогда проверяются только числа)."""
        if self._gas_names is None:
            d = await self.call_gas({"op": "card"})
            if not d:                                   # база недоступна – проверим позже ещё раз
                return set(), set()
            names = {r["name"] for r in d.get("rows") or [] if r.get("name")}
            grades = {x.replace(",", ".") for n in names for x in re.findall(r"\d+(?:[.,]\d+)?", n)}
            self._gas_names = (names, grades)
        return self._gas_names

    async def call_gas(self, args: dict[str, Any]) -> dict[str, Any]:
        """Вызов gas_query кодом агента (имена газов для проверки, исправленный запрос); ошибка – пустой ответ."""
        tool = next((t for t in self.tools if t.name == check.GAS_TOOL), None)
        if tool is None:
            return {}
        try:
            res = await tool.ainvoke(dict(args))
        except Exception:
            return {}
        text = res if isinstance(res, str) else " ".join(
            x.get("text", "") if isinstance(x, dict) else str(x) for x in (res if isinstance(res, list) else [res]))
        try:
            d = json.loads(text)
        except ValueError:
            return {}
        return d if isinstance(d, dict) else {}

    def running(self) -> set[str]:
        return {sid for sid, t in self._running.items() if not t.done()}

    def reset_models(self) -> None:
        """После смены ключей в .env: агенты пересоздаются с новыми ключами при следующем вопросе."""
        self._agents.clear()
        self._traces.clear()
        self._models.clear()

    @staticmethod
    def _config(session_id: str) -> dict:
        # шагов графа на ход: каждая итерация – ~8 узлов (прослойки до и после модели, инструменты); предел должен
        # быть выше лимита вызовов модели (25), иначе ход обрывается ошибкой, а не понятным ответом
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 400}

    async def _run(self, session_id: str, model_name: str, payload: Any,
                   emit: Callable[[dict[str, Any]], None] | None = None, keep: int | None = None) -> TurnResult:
        """Один ход агента. emit получает события по мере выполнения: шаги модели и инструментов из трассы
        и куски текста ответа ({"event": "token"}) – веб-чат показывает их вживую."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            sess = await self.sessions.get(session_id) or await self.sessions.create(session_id=session_id)
            try:
                agent, trace, name = self.agent_for(model_name)
            except Exception as e:                       # нет ключа, нет Ollama, неверный профиль
                return TurnResult(session_id, model_name, "error", error=str(e))
            events: list[dict[str, Any]] = []

            def listener(e: dict[str, Any]) -> None:
                if e.get("session") in (session_id, None):
                    events.append(e)
                    if emit:
                        emit(e)
            trace.listeners.append(listener)
            if emit:
                self._emit[session_id] = emit
            cfg = self._config(session_id)
            if keep:
                cfg["configurable"]["keep_messages"] = min(max(int(keep), KEEP_MIN), KEEP_MAX)
            if sess.get("compact_id"):                   # «сжать весь чат»: до отметки модель видит выжимку
                cfg["configurable"]["compact_id"] = sess["compact_id"]
            try:
                # stream_mode="messages": текст ответа приходит кусками, пока модель пишет
                async for chunk, meta in agent.astream(payload, cfg, stream_mode="messages"):
                    if emit and meta.get("langgraph_node") == "model" and isinstance(chunk, AIMessageChunk):
                        if text := _text(chunk.content):
                            emit({"event": "token", "text": text})
            except asyncio.CancelledError:
                await self._close_cancelled(agent, cfg)
                raise
            except Exception as e:
                why = LIMIT_NOTE if type(e).__name__ == "GraphRecursionError" else f"{type(e).__name__}: {e}"
                return TurnResult(session_id, name, "error", events=events, error=why)
            finally:
                trace.listeners.remove(listener)
                self._emit.pop(session_id, None)
            await self.sessions.touch(session_id, model=name)
            state = await agent.aget_state(cfg)
            if state.interrupts:
                return TurnResult(session_id, name, "needs_approval", events=events,
                                  approvals=self._approvals(state.interrupts))
            answer, verify = self._last_answer(state.values), self._last_verify(state.values)
            # предел шагов – ход кончился до проверки ответа; куски, которые код уже собрал, – человеку
            if answer.startswith("Model call limits exceeded") and (alt := await self._limit_answer(agent, cfg, state)):
                answer, verify = alt
            return TurnResult(session_id, name, "done", answer=_ru(answer), events=events, verify=verify)

    async def _limit_answer(self, agent, cfg: dict, state) -> tuple[str, dict] | None:
        """Модель не уложилась в предел шагов: ответ – куски этого хода, собранные кодом (каждый со строкой «понял
        вопрос так»), с пометкой; в истории – то же. Кусков нет – None (остаётся сообщение о пределе)."""
        msgs = _messages(state.values)
        start = max((i for i, m in enumerate(msgs) if is_question(m)), default=0)
        turn = msgs[start:]
        results_all, results_turn = check.tool_results(_ok_tools(msgs)), check.tool_results(_ok_tools(turn))
        own = list(dict.fromkeys(d["block"] for n, d in results_turn if check.is_piece(d)))
        if not own:
            return None
        question = " ".join(_shown(m) for m in turn if isinstance(m, HumanMessage) and not _meta(m).get("check"))
        calls = [(tc["name"], tc.get("args") or {}) for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls or []]
        names, grades = await self.gas_names()
        got = check.check("\n\n".join(own), question, results_all, results_turn, calls, names, grades,
                          block_args=block_args(turn))
        text = LIMIT_PIECES + got["text"]
        verify = {"source": "code", "pieces": got["pieces"], "understood": got["understood"], "blocks": got["blocks"],
                  "checked": got["checked"], "problems": [], "unverified": [h for _, h in got["remarks"]],
                  "strict": True, "draft": "", "limit": True}
        last = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not m.tool_calls), None)
        if last is not None:
            try:
                meta = {**(last.response_metadata or {})}
                meta["iacpaas"] = {**meta.get("iacpaas", {}), "verify": verify}
                await agent.aupdate_state(cfg, {"messages": [last.model_copy(update={"content": text,
                                                                                     "response_metadata": meta})]},
                                          as_node="model")
            except Exception:                            # история останется с сообщением о пределе – не критично
                pass
        return text, verify

    async def _close_cancelled(self, agent, cfg: dict) -> None:
        """Пользователь остановил запрос. Если модель успела запросить инструменты, а ответов нет – история
        «сломана» (провайдеры требуют ответ на каждый tool call). Дописываем ответы-заглушки и пометку."""
        try:
            state = await agent.aget_state(cfg)
            msgs = state.values.get("messages", []) if state else []
            answered = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
            last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
            dangling = [tc for tc in (last_ai.tool_calls if last_ai else []) if tc["id"] not in answered]
            if dangling:
                await agent.aupdate_state(cfg, {"messages": [
                    ToolMessage(content="[cancelled] Остановлено пользователем", tool_call_id=tc["id"],
                                name=tc["name"], status="error") for tc in dangling]}, as_node="tools")
            await agent.aupdate_state(cfg, {"messages": [AIMessage(
                content=STOPPED, response_metadata={"iacpaas": {"ts": time.time(), "stopped": True}})]},
                as_node="model")
        except Exception:                                # история останется как есть – не критично
            pass

    async def _prepare_ask(self, session_id: str, text: str, model: str | None) -> tuple[str, Any] | TurnResult:
        try:
            waiting = await self.pending(session_id, model)
        except Exception as e:                           # нет ключа модели, неверный профиль и т. п.
            return TurnResult(session_id, model or self.s.model, "error", error=str(e))
        if waiting:
            return TurnResult(session_id, model or self.s.model, "error",
                              error="Сначала ответьте на запрос подтверждения (approve/reject)")
        sess = await self.sessions.get(session_id) or await self.sessions.create(session_id=session_id)
        if sess["title"].startswith("Диалог "):
            await self.sessions.touch(session_id, title=text[:60])
        code = bool(CODE_ASK.search(text))
        extra = CODE_NOTE if code else ""
        if DIALOG_ASK.search(text):
            saved = await self.checkpointer.aget_tuple(self._config(session_id))
            old = saved.checkpoint.get("channel_values", {}).get("messages", []) if saved else []
            qs = [(_shown(m), _meta(m).get("steer")) for m in old if isinstance(m, HumanMessage)
                  and (m.additional_kwargs or {}).get("lc_source") != "summarization" and not _meta(m).get("check")]
            if qs:
                extra += DIALOG_NOTE.format(qs=" ".join(
                    f"{n}) «{q[:200]}»" + (" (дописано во время работы)" if st else "")
                    for n, (q, st) in enumerate(qs, 1)))
            if ANSWER_ASK.search(text):
                # «вопрос N → ответ»: последний текст модели до следующего вопроса; ответы на вопросы о самом
                # диалоге не берём – модель повторяла свой прошлый неверный ответ, а не первый
                pairs: list[list] = []
                for m in old:
                    if is_question(m) and (m.additional_kwargs or {}).get("lc_source") != "summarization":
                        pairs.append([_shown(m), ""])
                    elif isinstance(m, AIMessage) and pairs and not m.tool_calls and _text(m.content).strip() \
                            and not _meta(m).get("draft"):
                        pairs[-1][1] = _text(m.content)
                ans = [f"на вопрос {n} – «{a[:1200]}»" for n, (q, a) in enumerate(pairs, 1)
                       if a and not DIALOG_ASK.search(q)]
                if ans:
                    extra += ANSWERS_NOTE.format(ans="; ".join(ans))
        # уточнение к прошлому («а из них жидкие?», «сравни его с 4.8»): что код посчитал по каждому прошлому вопросу
        # и какие газы вышли – даже если тот вопрос давно ушёл из окна памяти в выжимку
        if FOLLOW.search(text) and not DIALOG_ASK.search(text):
            saved = await self.checkpointer.aget_tuple(self._config(session_id))
            old = saved.checkpoint.get("channel_values", {}).get("messages", []) if saved else []
            if items := follow_items(old):
                extra += FOLLOW_NOTE.format(items="\n".join(items))
        self._gas_names = None                           # названия газов для проверки – по свежим данным
        # shown – что видит пользователь в истории (без служебных справок для модели)
        msg = HumanMessage(content=text + extra, response_metadata={"iacpaas": {
            "ts": time.time(), **({"note": True} if code else {}), **({"shown": text} if extra else {})}})
        return model or sess.get("model") or self.s.model, {"messages": [msg]}

    async def _prepare_decide(self, session_id: str, approve: list[bool] | bool, model: str | None,
                              reason: str) -> tuple[str, Any] | TurnResult:
        try:
            pending = await self.pending(session_id, model)
        except Exception as e:
            return TurnResult(session_id, model or self.s.model, "error", error=str(e))
        if not pending:
            return TurnResult(session_id, model or self.s.model, "error", error="Нет ожидающих подтверждений")
        flags = approve if isinstance(approve, list) else [approve] * len(pending)
        decisions = [{"type": "approve"} if ok else
                     {"type": "reject", "message": reason or "Пользователь отклонил изменение. Не повторяй вызов, "
                                                            "сообщи пользователю, что изменение не выполнено."}
                     for ok in flags]
        sess = await self.sessions.get(session_id)
        return model or (sess or {}).get("model") or self.s.model, Command(resume={"decisions": decisions})

    async def ask(self, session_id: str, text: str, model: str | None = None,
                  emit: Callable[[dict[str, Any]], None] | None = None, keep: int | None = None,
                  edit: int | None = None) -> TurnResult:
        """Вопрос. edit – номер исправленного вопроса (с 0): он и всё после него удаляются, ход – заново."""
        if edit is not None and not await self.truncate(session_id, edit, model):
            return TurnResult(session_id, model or self.s.model, "error", error="Нет такого вопроса для правки")
        prep = await self._prepare_ask(session_id, text, model)
        if isinstance(prep, TurnResult):
            return prep
        # каждый вопрос разбирает модель заново (кэша ответов нет), данные – свежие с платформы
        return await self._run(session_id, *prep, emit=emit, keep=keep)

    async def decide(self, session_id: str, approve: list[bool] | bool, model: str | None = None,
                     reason: str = "", emit: Callable[[dict[str, Any]], None] | None = None,
                     keep: int | None = None) -> TurnResult:
        prep = await self._prepare_decide(session_id, approve, model, reason)
        return prep if isinstance(prep, TurnResult) else await self._run(session_id, *prep, emit=emit, keep=keep)

    async def stream(self, session_id: str, *, text: str | None = None, approve: list[bool] | bool | None = None,
                     model: str | None = None, keep: int | None = None,
                     edit: int | None = None) -> AsyncIterator[dict[str, Any]]:
        """Ход агента как поток событий для веб-чата: шаги, текст ответа кусками, итог (done / needs_approval /
        error / cancelled). Остановка – cancel(session_id) или разрыв соединения клиентом."""
        queue: asyncio.Queue = asyncio.Queue()
        started = time.perf_counter()

        async def runner() -> None:
            try:
                if text is not None:
                    r = await self.ask(session_id, text, model, emit=queue.put_nowait, keep=keep, edit=edit)
                else:
                    r = await self.decide(session_id, approve if approve is not None else False, model,
                                          emit=queue.put_nowait, keep=keep)
                queue.put_nowait({"event": "result", **_turn_dict(r), "leftover": self._inbox.pop(session_id, []),
                                  "ms": round((time.perf_counter() - started) * 1000)})
            except asyncio.CancelledError:
                queue.put_nowait({"event": "cancelled", "leftover": self._inbox.pop(session_id, []),
                                  "ms": round((time.perf_counter() - started) * 1000)})
            except Exception as e:                       # не должно случаться: _run сам ловит ошибки
                queue.put_nowait({"event": "result", "status": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(runner())
        self._running[session_id] = task
        try:
            while (item := await queue.get()) is not None:
                yield item
        finally:
            if not task.done():                          # клиент ушёл – останавливаем агента и модель
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if self._running.get(session_id) is task:
                self._running.pop(session_id, None)

    def steer(self, session_id: str, text: str, item_id: str = "") -> bool:
        """Дописать сообщение в идущий ход агента. False – агент сейчас не работает (отправьте обычным вопросом)."""
        task = self._running.get(session_id)
        if not text.strip() or not task or task.done():
            return False
        self._inbox.setdefault(session_id, []).append({"id": item_id, "text": text.strip()})
        return True

    def unsteer(self, session_id: str, item_id: str) -> bool:
        """Забрать дописанное, пока модель его не прочитала. False – уже прочитано (или такого нет)."""
        items = self._inbox.get(session_id) or []
        left = [it for it in items if it["id"] != item_id]
        if len(left) == len(items):
            return False
        self._inbox[session_id] = left
        return True

    def cancel(self, session_id: str) -> bool:
        task = self._running.get(session_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def pending(self, session_id: str, model: str | None = None) -> list[Approval]:
        saved = await self.checkpointer.aget_tuple(self._config(session_id))
        if not saved or not any(w[1] == "__interrupt__" for w in saved.pending_writes or []):
            return []                     # подтверждения не ждём – агента (модель, MCP) ради истории не поднимаем
        agent, _, _ = self.agent_for(model or self.s.model)
        state = await agent.aget_state(self._config(session_id))
        return self._approvals(state.interrupts) if state and state.interrupts else []

    @staticmethod
    def _approvals(interrupts) -> list[Approval]:
        out: list[Approval] = []
        for intr in interrupts:
            value = intr.value if hasattr(intr, "value") else intr
            for i, req in enumerate((value or {}).get("action_requests", [])):
                args = {k: ("***" if k == "confirm_token" else v) for k, v in (req.get("args") or {}).items()}
                out.append(Approval(i, req.get("name", "?"), args, req.get("description", "")))
        return out

    @staticmethod
    def _last_verify(value: dict) -> dict[str, Any]:
        last = next((m for m in reversed(value.get("messages", [])) if isinstance(m, AIMessage) and not m.tool_calls),
                    None)
        return _meta(last).get("verify") or {} if last is not None else {}

    @staticmethod
    def _last_answer(value: dict) -> str:
        for m in reversed(value.get("messages", [])):
            if isinstance(m, AIMessage) and not m.tool_calls:
                c = m.content
                return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return ""

    async def _saved(self, session_id: str) -> list:
        """Сообщения диалога прямо из чекпоинта: модель не нужна (её может уже не быть)."""
        saved = await self.checkpointer.aget_tuple(self._config(session_id))
        return saved.checkpoint.get("channel_values", {}).get("messages", []) if saved else []

    async def truncate(self, session_id: str, index: int, model: str | None = None) -> bool:
        """Правка вопроса (как в основных чатах): вопрос номер index (с 0) и всё после него – вопросы, ответы,
        вызовы, ожидающее подтверждение – удаляются из истории и для модели, и в чате."""
        task = self._running.get(session_id)
        if task and not task.done() and task is not asyncio.current_task():   # идёт другой ход этого диалога
            return False
        msgs = await self._saved(session_id)
        qs = [i for i, m in enumerate(msgs) if is_question(m)]
        if not 0 <= index < len(qs):
            return False
        tail = msgs[qs[index]:]
        agent, _, _ = self.agent_for(model or (await self.sessions.get(session_id) or {}).get("model") or self.s.model)
        await agent.aupdate_state(self._config(session_id), {"messages": [RemoveMessage(id=m.id) for m in tail]},
                                  as_node="model")
        sess = await self.sessions.get(session_id) or {}
        if sess.get("compact_id") in {m.id for m in tail}:   # сжатое вручную удалено – отметка не нужна
            await self.sessions.set_compact(session_id, None)
        return True

    async def compact(self, session_id: str) -> dict[str, Any]:
        """«Сжать весь чат» (как /compact в Claude Code): всё, что было до этой минуты, модель дальше видит
        выжимкой кодом (вопросы, числа и строки таблиц из ответов, пути газов); история в БД и в чате – целиком.
        Возвращает выжимку – человек видит, что именно осталось модели."""
        if session_id in self.running():
            return {"ok": False, "text": "Агент сейчас отвечает – сожмите после ответа."}
        msgs = await self._saved(session_id)
        blocks = _blocks([m for m in msgs if not (_meta(m).get("draft") or _meta(m).get("check"))])
        if not any(is_question(b[0]) for b in blocks):
            return {"ok": False, "text": "Сжимать нечего: в диалоге ещё нет вопросов."}
        await self.sessions.set_compact(session_id, msgs[-1].id)
        text = digest(blocks)
        return {"ok": True, "questions": sum(is_question(b[0]) for b in blocks), "before": _size(msgs) // CHARS_PER_TOKEN,
                "after": len(text) // CHARS_PER_TOKEN, "digest": text}

    async def context(self, session_id: str, keep: int | None = None, model: str | None = None) -> dict[str, Any]:
        """Сколько прошлого диалога уйдёт модели со следующим вопросом (токены после сжатия кодом) при этой длине
        памяти и бюджете модели – то же compact_messages, что в ContextMiddleware, без вызова модели."""
        msgs = await self._saved(session_id)
        sess = await self.sessions.get(session_id) or {}
        name = model or sess.get("model") or self.s.model
        budget = context_budget(name, self.s.models_file)
        keep = min(100, max(10, keep or self.s.keep_messages))
        info: dict[str, Any] = {}
        compact_messages([*msgs, HumanMessage(content="…", id="next-question")], keep, budget, sess.get("compact_id"),
                         info)
        # как /context в Claude Code: весь следующий запрос по частям – инструкция, схемы инструментов, история – и
        # какую долю окна модели он займёт. Символы на токен: замер по последнему шагу модели этого чата (сколько
        # символов ушло – сколько токенов прочитала, по её данным), иначе из профиля, иначе – «примерно» (~)
        fixed = len(self.system_prompt) + self._tools_chars()
        cpt, how, last = chars_per_token(name, self.s.models_file), "profile", None
        # замер – по последним шагам вместе (символы всех / токены всех): по одному шагу делитель скакал от текста
        # шага (русский, JSON, числа), и не меняющиеся инструкция и инструменты «прыгали» после каждого вопроса
        steps = [i for i in range(len(msgs)) if isinstance(msgs[i], AIMessage) and usage(msgs[i]).get("tokens_in")]
        meas = []                                         # (символы истории шага, токены шага)
        for at in steps[-40:]:
            t = usage(msgs[at])["tokens_in"]
            h = _size(compact_messages(list(msgs[:at]), keep, budget, sess.get("compact_id")))
            if 1.2 <= (fixed + h) / t <= 8:               # вне – замер не тот (сбой, чужая модель)
                meas.append((h, t))
        if steps:
            last = usage(msgs[steps[-1]])["tokens_in"]
        n = len(meas)
        cpt_fixed = cpt_hist = None
        if meas:
            cpt, how = sum(fixed + h for h, _ in meas) / sum(t for _, t in meas), "measured"
            # инструкция и схемы инструментов не меняются – их делитель по шагам почти без истории (иначе данные
            # инструментов – JSON, числа – тянули его и неизменная часть «плыла»); история – по остатку токенов
            small = [(h, t) for h, t in meas if h <= fixed / 3]
            cpt_fixed = sum(fixed + h for h, _ in small) / sum(t for _, t in small) if small else cpt
            rest = sum(t - fixed / cpt_fixed for _, t in meas)
            cpt_hist = sum(h for h, _ in meas) / rest if rest > 0 else cpt
            if not 1.2 <= cpt_hist <= 8:
                cpt_hist = cpt
        if not cpt:
            cpt, how = 3.0, "approx"
        hist = info.get("tokens", _size(msgs) // CHARS_PER_TOKEN) * CHARS_PER_TOKEN
        ch, cf = cpt_hist or cpt, cpt_fixed or cpt
        return {"tokens": int(hist / ch), "budget": int(budget * CHARS_PER_TOKEN / ch), "keep": keep,
                "questions": info.get("questions", 0), "full": info.get("full", 0), "digest": info.get("digest", 0),
                "manual": bool(sess.get("compact_id")), "window": context_window(name, self.s.models_file),
                "system": int(len(self.system_prompt) / cf), "tools": int(self._tools_chars() / cf),
                "tools_n": len(getattr(self, "tools", None) or []), "last_in": last,
                "cpt": round(cpt, 2), "cpt_from": how, "steps": n, "model": self._model_name(name)}

    def _model_name(self, name: str) -> str:
        """«auto» – какой профиль на самом деле (для подписи в окне контекста)."""
        try:
            return resolve_name(name, self.s.models_file)
        except Exception:
            return name

    def _tools_chars(self) -> int:
        """Схемы инструментов, которые уходят модели с каждым шагом (название, описание, параметры) – в символах."""
        cache = self.__dict__.setdefault("_tools_tok", {})
        if "all" not in cache:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            tools = list(getattr(self, "tools", None) or [])
            try:
                text = json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False)
            except Exception:                           # схема не собралась – оценка по описаниям
                text = " ".join(f"{t.name} {t.description}" for t in tools)
            cache["all"] = len(text)
        return cache["all"]

    async def history(self, session_id: str, model: str | None = None) -> list[dict[str, Any]]:
        messages = await self._saved(session_id)
        mark = (await self.sessions.get(session_id) or {}).get("compact_id")
        out = []
        steered = False                                  # пользователь дописал сообщение после прошлого шага модели
        for m in messages:
            meta = (m.response_metadata or {}).get("iacpaas", {})
            if isinstance(m, HumanMessage):
                if (m.additional_kwargs or {}).get("lc_source") == "summarization" or meta.get("check"):
                    continue                             # резюме прежнего сжатия, замечание проверки – не вопросы
                text = _shown(m)
                steered = bool(meta.get("steer"))
                out.append({"role": "user", "text": text, "ts": meta.get("ts"),
                            **({"steer": True} if meta.get("steer") else {})})
            elif isinstance(m, AIMessage):
                if not meta.get("stopped"):              # пометка «остановлено» – не шаг модели
                    out.append({"role": "model", "ts": meta.get("ts"), "ms": meta.get("ms"),
                                **usage(m),
                                "tool_calls": [tc["name"] for tc in m.tool_calls]})
                for tc in m.tool_calls:
                    out.append({"role": "tool_call", "tool": tc["name"], "id": tc.get("id"),
                                "args": {k: ("***" if k == "confirm_token" else v) for k, v in tc["args"].items()}})
                if (text := _text(m.content)) and m.tool_calls:   # сказал по ходу работы (до вызова инструмента)
                    # ответ на дописанное – в ленту; иначе это часть работы, страница кладёт его внутрь шага
                    out.append({"role": "interim", "text": text, "ts": meta.get("ts"),
                                **({"steer": True} if steered else {})})
                elif text and meta.get("draft"):         # черновик, не прошедший проверку кодом
                    out.append({"role": "draft", "text": text, "ts": meta.get("ts"),
                                "problems": (meta.get("verify") or {}).get("problems", [])})
                elif text:
                    v = meta.get("verify")
                    out.append({"role": "assistant", "text": _ru(text), "ts": meta.get("ts"),
                                "stopped": bool(meta.get("stopped")), **({"verify": v} if v else {})})
                steered = False
            elif isinstance(m, ToolMessage):
                out.append({"role": "tool_result", "tool": m.name, "id": m.tool_call_id, "status": m.status,
                            "text": _text(m.content)[:1500]})
            if mark and m.id == mark:                    # здесь пользователь сжал чат: что осталось модели
                upto = messages[:messages.index(m) + 1]
                blocks = _blocks([x for x in upto if not (_meta(x).get("draft") or _meta(x).get("check"))])
                out.append({"role": "compact", "text": digest(blocks), "questions": sum(is_question(b[0]) for b in blocks),
                            "before": _size(upto) // CHARS_PER_TOKEN})
        return out

    def trace_for(self, session_id: str) -> list[dict[str, Any]]:
        """Трасса одного диалога из logs/agent_trace.jsonl – для скачивания из веб-чата."""
        if not self.s.trace_file.is_file():
            return []
        rows = []
        with self.s.trace_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("session") == session_id:
                    rows.append(e)
        return rows
