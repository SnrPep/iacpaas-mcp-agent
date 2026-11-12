"""Память диалога: сжатие кодом под бюджет модели, «сжать весь чат», правка вопроса, контекст в консоли и вебе."""

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import IR, make_server
from iacpaas_agent import core
from iacpaas_agent.core import AgentRuntime, compact_messages
from test_agent import UPDATE, settings, write_models

GAS = "Моногазы/Газы на основе Аргона/Аргон газообразный высокой чистоты марка "


def turn(i, big=500):
    return [HumanMessage(f"вопрос {i}", id=f"q{i}"),
            AIMessage("", id=f"a{i}", tool_calls=[{"name": "get_subgraph", "args": {"root": GAS + f"{i}.0"},
                                                   "id": f"c{i}", "type": "tool_call"}]),
            ToolMessage("x" * big, tool_call_id=f"c{i}", name="get_subgraph", id=f"t{i}"),
            AIMessage(f"**Аргон {i}.0**\n| Компонент | Доля, % |\n|---|---|\n| O | 0.000{i} |\nЕщё текст без чисел.",
                      id=f"r{i}")]


def test_extract_does_not_cut_numbers():
    """Строка выжимки длиннее предела режется по слову: число не обрезается («ещё 32» → не «ещё 3»)."""
    from iacpaas_agent.core import extract
    line = "y" + "x " * 76 + "и ещё 32)."
    assert line[159:161] == "32"                     # предел 160 – ровно посреди числа
    out = extract(line)
    assert out.endswith(" …") and not re.search(r"ещё 3(?!\d)", out)


def test_digest_keeps_gases_of_gas_query():
    """Газы из запроса gas_query (gases="газ 1; газ 2") попадают в выжимку – «сравни их» после сжатия знает, какие."""
    from iacpaas_agent.core import digest_item
    block = [HumanMessage("сравни аргон 4.8 и 6.0", id="q1"),
             AIMessage("", id="a1", tool_calls=[{"name": "gas_query", "id": "c1", "type": "tool_call",
                                                 "args": {"op": "compare", "gases": "Аргон 4.8; Аргон 6.0"}}]),
             ToolMessage("{}", tool_call_id="c1", name="gas_query", id="t1"),
             AIMessage("{{#ab12}}", id="r1")]
    assert "пути: «Аргон 4.8», «Аргон 6.0»" in digest_item(1, block)


def test_budget_moves_old_questions_to_digest_with_numbers_and_paths():
    """Мало бюджета (локальная модель) – старые вопросы уходят в выжимку кодом: вопрос, числа из ответа и путь
    газа дословно; предыдущий вопрос – целиком; тот же вход – тот же выход (предсказуемо)."""
    msgs = [m for i in range(1, 7) for m in turn(i, big=3000)] + [HumanMessage("вопрос 7", id="q7")]
    info = {}
    out = compact_messages(msgs, keep=100, budget=2000, info=info)
    note = out[0].content
    assert note.startswith("[Начало этого диалога сжато кодом")
    assert "1) «вопрос 1» → ответ: Аргон 1.0; O | 0.0001; пути: «" + GAS + "1.0»" in note
    assert "Ещё текст без чисел" not in note                     # выписка – только строки с числами
    assert out[-1].content == "вопрос 7"
    assert any(m.content == "x" * 3000 for m in out)              # данные предыдущего вопроса – целиком
    assert info["digest"] >= 3 and info["full"] >= 1 and info["tokens"] <= 2000
    assert compact_messages(msgs, keep=100, budget=2000) == out   # предсказуемо
    big = compact_messages(msgs, keep=100, budget=100_000)        # бюджет GigaChat – всё целиком
    assert not big[0].content.startswith("[Начало") and len(big) == len(msgs)


def test_tiny_budget_stubs_previous_data_then_shortens_digest():
    msgs = [m for i in range(1, 30) for m in turn(i, big=5000)] + [HumanMessage("последний", id="qz")]
    info = {}
    out = compact_messages(msgs, keep=100, budget=500, info=info)
    note = out[0].content
    assert "1) «вопрос 1»" in note and "28) «вопрос 28»" in note  # вопросы не теряются
    assert note.endswith("вопрос 29")                             # предыдущий вопрос – целиком, после выжимки
    assert not any(m.content == "x" * 5000 for m in out)          # данные предыдущего – заглушкой
    assert info["short"] > 0 and "→ ответ" in note                # у старых – только вопрос, у новых – и числа


def test_drafts_and_check_notes_are_not_sent_later():
    draft = AIMessage("черновик 999", id="d1", response_metadata={"iacpaas": {"draft": True}})
    check = HumanMessage("[Проверка агента…]", id="k1", response_metadata={"iacpaas": {"check": True}})
    msgs = turn(1) + [draft, check, AIMessage("ответ 11", id="f1")] + [HumanMessage("вопрос 2", id="q2")]
    out = compact_messages(msgs, keep=100)
    assert all(m.id not in ("d1", "k1") for m in out) and out[-1].content == "вопрос 2"


def test_manual_compact_marker_digests_everything_before():
    msgs = turn(1) + turn(2) + [HumanMessage("вопрос 3", id="q3")]
    out = compact_messages(msgs, keep=100, compact_id="r2")
    assert out[0].content.startswith("[Начало этого диалога сжато кодом") and out[0].content.endswith("вопрос 3")
    assert "2) «вопрос 2» → ответ: Аргон 2.0; O | 0.0002" in out[0].content and len(out) == 1


def test_profile_budget():
    from iacpaas_agent.models import DEFAULT_CONTEXT, context_budget
    from iacpaas_agent.settings import AgentSettings
    f = AgentSettings().models_file
    assert context_budget("gigachat", f) == 24000 and context_budget("ollama-small", f) == 3000
    assert context_budget("openai", f) == DEFAULT_CONTEXT


def answer_steps(n):
    return [{"tool": "count_graph", "args": {"path": IR}}] + [{"say": f"В базе 11 вершин (ответ {k})."}
                                                             for k in range(1, n + 1)]


@pytest.mark.anyio
async def test_runtime_compact_history_divider_and_model_sees_digest(tmp_path, monkeypatch):
    used = []
    orig = core.compact_messages
    monkeypatch.setattr(core, "compact_messages",
                        lambda msgs, keep, budget=None, cid=None, info=None: used.append(cid) or
                        orig(msgs, keep, budget, cid, info))
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [
        {"tool": "count_graph", "args": {"path": IR}}, {"say": "По базе: {{@count_graph}}"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.compact("c1"))["ok"] is False             # пусто – сжимать нечего
        await rt.ask("c1", "Сколько вершин?")
        await rt.ask("c1", "А дуг?")
        res = await rt.compact("c1")
        assert res["ok"] and res["questions"] == 2 and "1) «Сколько вершин?» → ответ: Весь ИР: вершин 11, дуг 11" in res["digest"]
        assert res["after"] < res["before"]
        events = []
        await rt.ask("c1", "ещё раз", emit=events.append)
        ctx = [e for e in events if e["event"] == "context"]
        assert ctx and ctx[0]["manual"] and ctx[0]["digest"] == 2
        hist = await rt.history("c1")
    roles = [h["role"] for h in hist]
    assert roles.count("compact") == 1 and roles.index("compact") < [i for i, h in enumerate(hist)
                                                                      if h["role"] == "user"][2]
    assert used[-1] is not None


@pytest.mark.anyio
async def test_edit_question_drops_everything_below(tmp_path):
    """Правка вопроса: он и всё ниже (вопросы, ответы, вызовы, ожидающее подтверждение) удаляются – и для модели,
    и в истории; ход идёт заново от исправленного вопроса."""
    server, _ = make_server(allow_write=True)
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "cycle": True, "script": [{"say": "ответ"}]},
        "w": {"provider": "fake", "model": "s", "script": [{"tool": "update_concept", "args": UPDATE}, {"say": "ok"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        for q in ("первый", "второй", "третий"):
            await rt.ask("e1", q)
        r = await rt.ask("e1", "второй исправленный", edit=1)
        assert r.status == "done"
        users = [h["text"] for h in await rt.history("e1") if h["role"] == "user"]
        assert users == ["первый", "второй исправленный"]
        assert (await rt.ask("e1", "x", edit=5)).status == "error"
        # правка вопроса, после которого ждёт подтверждение изменения: ожидание снимается
        p = await rt.ask("e2", "Пометь светобоязнь", model="w")
        assert p.status == "needs_approval" and await rt.pending("e2", "w")
        r2 = await rt.ask("e2", "Сколько вершин?", edit=0)
        assert r2.status == "done" and not await rt.pending("e2")
        assert [h["text"] for h in await rt.history("e2") if h["role"] == "user"] == ["Сколько вершин?"]


@pytest.mark.anyio
async def test_console_continues_last_console_dialog(tmp_path):
    """Консоль: без --session продолжается последний диалог консоли (раньше каждый запуск – новый, и после
    перезапуска агент «не помнил» разговор); --new – новый; диалоги веб-чата консоль не подхватывает."""
    from argparse import Namespace

    from iacpaas_agent.cli import pick_session
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [{"say": "ок"}]}})
    args = Namespace(session=None, new=False, question=None)
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        first, note = await pick_session(rt, args)
        assert note == ""
        await rt.ask(first, "мой первый вопрос")
        await rt.sessions.create(source="web")                    # веб-диалог свежее – но он не консольный
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:   # перезапуск консоли
        again, note = await pick_session(rt, args)
        assert again == first and "Продолжаю диалог «мой первый вопрос» (1 вопр.)" in note
        fresh, _ = await pick_session(rt, Namespace(session=None, new=True, question=None))
        assert fresh != first


def test_repeat_request_keeps_write_permission():
    """«повтори» после «удали…» (ИР был занят сервисом) – та же просьба; без прошлой просьбы – нет."""
    from iacpaas_agent.core import write_asked
    ask = HumanMessage("удали ресурс с моей папки в песочнице")
    fail = AIMessage("Не получилось удалить: ИР используется сервисом.")
    assert write_asked([ask, fail, HumanMessage("повтори")])
    assert write_asked([ask, fail, HumanMessage("попробуй ещё раз")])
    assert not write_asked([HumanMessage("сколько газов?"), AIMessage("48"), HumanMessage("повтори")])


@pytest.mark.anyio
@pytest.mark.parametrize("keep", [None, 10])                    # консоль (без ползунка, 15) и веб с памятью 10
async def test_twelfth_question_still_knows_the_first(tmp_path, monkeypatch, keep):
    """12-й вопрос при памяти 10: первые вопросы ушли из окна, но модель видит их в выжимке кодом (вопрос и
    числа ответа) – так задумано, это не ошибка; ответы старше окна целиком не видны."""
    seen = []
    orig = core.compact_messages
    monkeypatch.setattr(core, "compact_messages", lambda *a, **k: seen.append(r := orig(*a, **k)) or r)
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [
        {"say": "Ответ: 42 вершины."}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        for i in range(1, 13):
            await rt.ask("k1", f"вопрос номер {i}", keep=keep)
    last = seen[-1]
    assert last[-1].content == "вопрос номер 12"
    assert "1) «вопрос номер 1» → ответ: Ответ: 42 вершины." in last[0].content
    visible = [m for m in last if m.type == "human"]
    assert len(visible) <= (keep or 15) // 2 + 2                  # целиком – только последние вопросы окна
