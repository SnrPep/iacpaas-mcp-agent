"""Трасса агента: JSONL-файл + подписчики (CLI и веб-чат показывают шаги вживую).
События: model_start / model_end (время, токены, какие инструменты выбрала модель), tool_call / tool_result."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt

SECRET_ARGS = {"confirm_token"}


def _thread(request) -> str | None:
    """id сессии (thread_id LangGraph) – из runtime инструмента или из контекста выполнения графа."""
    cfg = getattr(getattr(request, "runtime", None), "config", None)
    if not cfg:
        try:
            from langgraph.config import get_config
            cfg = get_config()
        except Exception:                        # вне графа (тесты middleware) – без сессии
            cfg = None
    return (cfg or {}).get("configurable", {}).get("thread_id")


def usage(ai: AIMessage | None) -> dict[str, Any]:
    """Токены шага: tokens_in – всё, что прочитала модель (вместе с кэшем), tokens_cached – из них из кэша.
    GigaChat кладёт кэшированную часть отдельно (precached_prompt_tokens) и в input_tokens её не включает –
    без поправки «↑» был бы в десятки раз меньше настоящего (23 вместо 3 177)."""
    u = (ai.usage_metadata or {}) if ai is not None else {}
    tin = u.get("input_tokens")
    cached = (u.get("input_token_details") or {}).get("cache_read") or 0
    tu = (ai.response_metadata or {}).get("token_usage") or {} if ai is not None else {}
    if tin is not None and "precached_prompt_tokens" in tu:
        tin += tu.get("precached_prompt_tokens") or 0
    return {"tokens_in": tin, "tokens_out": u.get("output_tokens"), "tokens_cached": cached}


def _say(ai: AIMessage) -> str:
    c = ai.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


class JsonlTrace(AgentMiddleware):
    def __init__(self, file: Path, model_name: str):
        super().__init__()
        self.file = file
        self.model_name = model_name
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    def emit(self, entry: dict[str, Any]) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model_name, **entry}
        with self.file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        for fn in list(self.listeners):
            fn(entry)

    async def awrap_model_call(self, request, handler):
        thread = _thread(request)
        start = time.perf_counter()
        self.emit({"event": "model_start", "session": thread})
        try:
            response = await handler(request)
        except Exception as e:
            self.emit({"event": "model_error", "session": thread, "error": f"{type(e).__name__}: {e}"[:300],
                       "ms": round((time.perf_counter() - start) * 1000)})
            raise
        ms = round((time.perf_counter() - start) * 1000)
        msgs = getattr(response, "result", None) or ([response] if isinstance(response, AIMessage) else [])
        ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        if ai is not None:                       # время ответа – в response_metadata: провайдеру он не отправляется
            ai.response_metadata = {**(ai.response_metadata or {}), "iacpaas": {"ts": time.time(), "ms": ms}}
        self.emit({"event": "model_end", "session": thread, "ms": ms,
                   **usage(ai),
                   "tool_calls": [tc["name"] for tc in (ai.tool_calls if ai else [])],
                   # текст рядом с вызовом инструмента – ответ по ходу работы (например, на дописанное сообщение)
                   **({"say": _say(ai)} if ai is not None and ai.tool_calls and _say(ai) else {})})
        return response

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        thread = _thread(request)
        args = {k: ("***" if k in SECRET_ARGS and v else v) for k, v in (call.get("args") or {}).items()}
        start = time.perf_counter()
        self.emit({"event": "tool_call", "session": thread, "tool": call["name"], "call_id": call.get("id"),
                   "args": args})
        try:
            result = await handler(request)
        except GraphInterrupt:                   # остановка на подтверждение кнопкой – не ошибка
            self.emit({"event": "tool_result", "session": thread, "tool": call["name"], "call_id": call.get("id"),
                       "status": "success", "preview": "Предпросмотр готов – ждёт вашего подтверждения",
                       "ms": round((time.perf_counter() - start) * 1000)})
            raise
        except Exception as e:
            self.emit({"event": "tool_exception", "session": thread, "tool": call["name"],
                       "call_id": call.get("id"), "error": repr(e)})
            raise
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = " ".join(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in content)
        self.emit({"event": "tool_result", "session": thread, "tool": call["name"], "call_id": call.get("id"),
                   "status": getattr(result, "status", "success"),
                   "ms": round((time.perf_counter() - start) * 1000), "preview": str(content)[:1500]})
        return result
