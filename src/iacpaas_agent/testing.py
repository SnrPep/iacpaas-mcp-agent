"""Сценарная модель: заранее заданные шаги (вызовы инструментов и ответы). Проверяет всю цепочку
агент -> MCP -> IACPaaS без LLM и без сети. С cycle=True сценарий повторяется на каждый вопрос."""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import uuid
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage


def _messages(script: list[dict[str, Any]], cycle: bool):
    # Каждый раз НОВОЕ сообщение с уникальными id: LangGraph заменяет сообщения с одинаковым id,
    # поэтому повторно отдавать те же объекты нельзя.
    for _ in itertools.count() if cycle else range(1):
        for step in script:
            uid = uuid.uuid4().hex[:8]
            # tokens_in – «сколько прочитала модель» (как в ответе настоящей модели) – для замера символов на токен
            use = {"usage_metadata": {"input_tokens": step["tokens_in"], "output_tokens": 1,
                                      "total_tokens": step["tokens_in"] + 1}} if "tokens_in" in step else {}
            if "tools" in step:                 # несколько вызовов одним шагом (как у настоящей модели)
                yield AIMessage(id=f"ai_{uid}", content=step.get("say", ""), tool_calls=[{
                    "name": t["tool"], "args": dict(t.get("args", {})), "id": f"call_{uid}_{i}", "type": "tool_call"}
                    for i, t in enumerate(step["tools"])], **use)
            elif "tool" in step:
                # как у GigaChat: вызов – и в tool_calls, и в additional_kwargs.function_call
                call = {"name": step["tool"], "arguments": dict(step.get("args", {}))}
                yield AIMessage(id=f"ai_{uid}", content=step.get("say", ""), tool_calls=[{   # say – текст по ходу работы
                    "name": step["tool"], "args": dict(step.get("args", {})), "id": f"call_{uid}", "type": "tool_call"}],
                    additional_kwargs={"function_call": call, "functions_state_id": f"fs_{uid}"}, **use)
            else:
                yield AIMessage(id=f"ai_{uid}", content=step["say"], **use)


class ScriptedModel(GenericFakeChatModel):
    delay: float = 0.0                          # «думает» столько секунд – для проверки остановки запроса

    def __init__(self, script: list[dict[str, Any]], cycle: bool = False, **kw: Any):
        # без потоковой выдачи: GenericFakeChatModel при стриминге теряет tool_calls
        super().__init__(messages=_messages(list(script), cycle), disable_streaming=True, **kw)

    async def _agenerate(self, *args: Any, **kwargs: Any):
        if self.delay:
            await asyncio.sleep(self.delay)
        res = self._generate(*args, **kwargs)
        # «{{@count_graph}}» в тексте сценария – метка куска из последнего результата этого инструмента: заранее
        # метку не узнать (она зависит от аргументов вызова)
        msgs = args[0] if args else kwargs.get("messages") or []
        marks: dict[str, str] = {}
        for m in msgs:
            if isinstance(m, ToolMessage):
                c = m.content if isinstance(m.content, str) else "".join(
                    x.get("text", "") if isinstance(x, dict) else str(x) for x in m.content or [])
                try:
                    d = json.loads(c)
                except ValueError:
                    continue
                if isinstance(d, dict) and isinstance(d.get("block"), str):
                    marks[m.name or ""] = d["block"]
        for g in res.generations:
            if isinstance(g.message.content, str) and "{{@" in g.message.content:
                g.message.content = re.sub(r"\{\{@(\w+)\}\}", lambda x: marks.get(x[1], x[0]), g.message.content)
        return res

    def bind_tools(self, tools, **kwargs):      # инструменты «знает» сценарий, а не модель
        return self
