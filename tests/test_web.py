"""Веб-чат: API поверх того же AgentRuntime (сервер MCP в процессе, сценарная модель)."""
from mcp import Client
from starlette.testclient import TestClient

from conftest import IR, ROOT, make_server
from iacpaas_agent.web.app import create_app
from test_agent import UPDATE, settings, write_models

import anyio
import pytest


def test_web_chat_approval_and_guards(tmp_path):
    server, backend = make_server(allow_write=True)

    async def preview_token():
        async with Client(server) as c:
            return (await c.call_tool("update_concept", UPDATE)).structured_content["confirm_token"]
    token = anyio.run(preview_token)
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "script": [
            {"tool": "count_graph", "args": {"path": IR}}, {"say": "{{@count_graph}}"}]},
        "w": {"provider": "fake", "model": "s", "script": [
            {"tool": "update_concept", "args": {**UPDATE, "confirm_token": token}}, {"say": "Готово"}]}})
    s = settings(tmp_path, models, web_token="sekret")
    app = create_app(s, mcp_target=server)
    with TestClient(app, base_url="http://127.0.0.1:8770") as c:
        assert c.get("/api/health").status_code == 401                       # нужен WEB_TOKEN
        h = {"X-Web-Token": "sekret"}
        health = c.get("/api/health", headers=h).json()
        assert health["server"]["backend"] == "mock" and health["server"]["write_tools"] == "sandbox"
        assert any(m["name"] == "a" for m in c.get("/api/models", headers=h).json()["models"])
        assert {"name": "update_concept", "write": True}.items() <= next(
            t for t in c.get("/api/tools", headers=h).json() if t["name"] == "update_concept").items()

        r = c.post("/api/chat", json={"message": "сколько вершин?", "model": "a"}, headers=h).json()
        assert r["status"] == "done" and "вершин 11, дуг 11" in r["answer"] and next(e for e in r["events"] if e["event"] == "tool_call")["tool"] == "count_graph"
        sid = r["session_id"]
        hist = c.get(f"/api/sessions/{sid}?model=a", headers=h).json()
        first = hist["history"][0]
        assert first["role"] == "user" and first["text"] == "сколько вершин?" and first["ts"] > 0   # время сообщения
        # старый диалог читается и без доступной модели (ключ удалили, выбран «auto» и т. п.)
        nomodel = c.get(f"/api/sessions/{sid}?model=нет-такой", headers=h).json()
        last = nomodel["history"][-1]
        assert last["role"] == "assistant" and "вершин 11, дуг 11" in last["text"] and last["ts"] and nomodel["pending"] == []

        w = c.post("/api/chat", json={"session": sid, "message": "переименуй", "model": "w"}, headers=h).json()
        assert w["status"] == "needs_approval" and w["approvals"][0]["args"]["confirm_token"] == "***"
        d = c.post("/api/decide", json={"session": sid, "approve": True, "model": "w"}, headers=h).json()
        assert d["status"] == "done"
        assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Фотофобия"

        # окно «Ключи» – только модели; доступ к платформе задаётся в консоли (iacpaas-keys), не в чате
        k = c.get("/api/keys", headers=h).json()
        assert "iacpaas" not in [p["name"] for p in k["profiles"]] and "iacpaas_modes" not in k
        bad = c.post("/api/keys", json={"name": "iacpaas", "values": {"IACPAAS_USERNAME": "x"}}, headers=h)
        assert bad.status_code == 400 and "iacpaas-keys" in bad.json()["error"]

        evil = c.post("/api/chat", json={"message": "x"}, headers={**h, "Origin": "https://evil.example"})
        assert evil.status_code == 403
        assert c.get("/", headers=h).status_code == 200
        assert c.delete(f"/api/sessions/{sid}", headers=h).json() == {"deleted": sid}


def test_hidden_attribute_wins_over_css_display():
    """Окно «Ключи» (#keysBox, display:flex) не должно показываться при загрузке страницы."""
    html = (ROOT / "src" / "iacpaas_agent" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert "[hidden]{display:none!important}" in html


def test_page_css_classes_do_not_hit_katex():
    """Класс страницы с тем же именем, что внутри формулы KaTeX, ломает вёрстку: .vlist списка «Что сверено» дал
    дроби зелёный фон и отступы – знаменатель срезался. Правила внутри .katex – не в счёт."""
    import re
    static = ROOT / "src" / "iacpaas_agent" / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", "".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.S)), flags=re.S)
    katex = set(re.findall(r"\.([a-zA-Z][\w-]*)", (static / "katex" / "katex.min.css").read_text(encoding="utf-8")))
    page = {c for sel in re.findall(r"([^{}]+)\{", css) for s in sel.split(",")
            for c in re.findall(r"\.([a-zA-Z][\w-]*)", s.split(".katex")[0])}
    assert page & katex == set()


def _events(text: str) -> list[dict]:
    import json
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_stream_steps_trace_and_tool_kinds(tmp_path):
    """Поток событий для веб-чата: шаги модели и инструментов по порядку, итог; трасса диалога скачивается."""
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "count_graph", "args": {"path": IR}}, {"say": "{{@count_graph}}"}]}})
    app = create_app(settings(tmp_path, models), mcp_target=server)
    with TestClient(app, base_url="http://127.0.0.1:8770") as c:
        ev = _events(c.post("/api/stream", json={"message": "сколько вершин?", "model": "a"}).text)
        kinds = [e["event"] for e in ev]
        assert kinds[0] == "session" and kinds[-1] == "result"
        assert kinds.index("model_start") < kinds.index("tool_call") < kinds.index("tool_result")
        assert next(e for e in ev if e["event"] == "model_end")["tool_calls"] == ["count_graph"]
        res = ev[-1]
        assert res["status"] == "done" and "вершин 11, дуг 11" in res["answer"] and res["ms"] >= 0
        sid = ev[0]["session"]
        trace = c.get(f"/api/sessions/{sid}/trace")
        assert "attachment" in trace.headers["content-disposition"]
        assert {e["event"] for e in _events("\n".join("data: " + x for x in trace.text.splitlines()))} >= {
            "model_start", "model_end", "tool_call", "tool_result"}
        tools = {t["name"]: t["kind"] for t in c.get("/api/tools").json()}
        assert tools["count_graph"] == "read" and tools["list_inforesources"] == "local" and "import_inforesource" not in tools
        m = c.get("/api/models").json()
        assert "auto" in m and {"name", "kind", "ready"} <= set(m["models"][0])     # выбор модели на странице
        assert c.post("/api/stream", json={"model": "a"}).status_code == 400          # пустой запрос
        assert c.post("/api/steer", json={"session": "нет", "message": "ещё"}).json() == {"accepted": False}
        assert c.post("/api/steer", json={"message": "ещё"}).status_code == 400
        assert c.post("/api/unsteer", json={"session": "нет", "id": "m1"}).json() == {"removed": False}
        # KaTeX лежит в проекте и раздаётся сервером: формулы работают без интернета
        js = c.get("/static/katex/katex.min.js")
        assert js.status_code == 200 and "katex" in js.text[:2000].lower()
        assert c.get("/static/katex/fonts/KaTeX_Main-Regular.woff2").status_code == 200
        assert 'src="/static/katex/katex.min.js"' in c.get("/").text


@pytest.mark.anyio
async def test_stop_cancels_turn_and_keeps_history_consistent(tmp_path):
    """«Остановить»: агент прерывается, в истории – пометка, следующий вопрос работает."""
    import asyncio
    from iacpaas_agent.core import STOPPED, AgentRuntime
    server, _ = make_server()
    models = write_models(tmp_path, {
        "slow": {"provider": "fake", "model": "s", "delay": 30, "script": [{"say": "не дождётесь"}]},
        "a": {"provider": "fake", "model": "s", "script": [{"say": "готово"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        seen = []
        async for e in rt.stream("s1", text="долгий вопрос", model="slow"):
            seen.append(e["event"])
            if e["event"] == "model_start":
                assert rt.cancel("s1")
        assert seen[-1] == "cancelled"
        hist = await rt.history("s1")
        assert hist[-1]["role"] == "assistant" and hist[-1]["text"] == STOPPED and hist[-1]["stopped"]
        r = await rt.ask("s1", "ещё вопрос", model="a")
        assert r.status == "done" and r.answer == "готово"
        assert not rt.cancel("s1")                                   # нечего останавливать
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_steer_adds_message_before_next_model_step(tmp_path):
    """Дописывание на ходу: сообщение, присланное во время работы, попадает в диалог перед следующим шагом
    модели; присланное, когда модель уже пишет итог, возвращается в leftover; без работы агента – отказ."""
    from iacpaas_agent.core import AgentRuntime
    server, _ = make_server()
    models = write_models(tmp_path, {
        "t": {"provider": "fake", "model": "s", "script": [{"tool": "count_graph", "args": {"path": IR}},
                                                            {"say": "учёл"}]},
        "a": {"provider": "fake", "model": "s", "delay": 0.2, "script": [{"say": "готово"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert not rt.steer("s1", "рано")                            # агент не работает
        seen = []
        async for e in rt.stream("s1", text="сколько вершин?", model="t"):
            seen.append(e)
            if e["event"] == "tool_call":
                assert rt.steer("s1", "и дуг тоже", "m1")
        steer = [e for e in seen if e["event"] == "steer"]
        assert steer and steer[0]["items"] == [{"id": "m1", "text": "и дуг тоже"}]
        order = [e["event"] for e in seen]
        last_model_end = len(order) - 1 - order[::-1].index("model_end")
        assert order.index("tool_result") < order.index("steer") < last_model_end    # учтено следующим шагом модели
        assert seen[-1]["status"] == "done" and seen[-1]["leftover"] == []
        hist = await rt.history("s1")
        roles = [(h["role"], h.get("steer", False)) for h in hist]
        assert roles.index(("user", True)) == roles.index(("tool_result", False)) + 1   # сразу после инструмента
        assert hist[roles.index(("user", True))]["text"] == "и дуг тоже"

        late = []
        async for e in rt.stream("s2", text="привет", model="a"):
            if e["event"] == "model_start":
                assert rt.steer("s2", "опоздал", "m2")              # модель уже пишет итоговый ответ
            late.append(e)
        assert late[-1]["status"] == "done" and late[-1]["leftover"] == [{"id": "m2", "text": "опоздал"}]
        assert not [h for h in await rt.history("s2") if h.get("steer")]


@pytest.mark.anyio
async def test_steered_message_answered_by_same_agent_along_the_way(tmp_path):
    """Как в Claude Code: дописанное получает тот же агент; текст рядом с вызовом инструмента – ответ по ходу работы
    (событие model_end.say, в истории – interim), задача продолжается; непрочитанное можно забрать (unsteer)."""
    from iacpaas_agent.core import AgentRuntime
    server, _ = make_server()
    models = write_models(tmp_path, {
        "p": {"provider": "fake", "model": "s", "script": [
            {"tool": "count_graph", "args": {"path": IR}},
            {"tool": "count_graph", "args": {"path": IR}, "say": "Привет! Продолжаю считать."},
            {"say": "{{@count_graph}}"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        seen = []
        async for e in rt.stream("s1", text="сколько вершин?", model="p"):
            seen.append(e)
            if e["event"] == "tool_call" and len([x for x in seen if x["event"] == "tool_call"]) == 1:
                assert rt.steer("s1", "привет", "m1") and rt.steer("s1", "передумал", "m2")
                assert rt.unsteer("s1", "m2") and not rt.unsteer("s1", "m2")   # забрал, пока не прочитано
        assert [e["items"] for e in seen if e["event"] == "steer"] == [[{"id": "m1", "text": "привет"}]]
        said = [e["say"] for e in seen if e["event"] == "model_end" and e.get("say")]
        assert said == ["Привет! Продолжаю считать."] and "вершин 11, дуг 11" in seen[-1]["answer"]
        hist = await rt.history("s1")
        roles = [h["role"] for h in hist]
        assert roles.count("interim") == 1 and roles[-1] == "assistant"
        assert next(h for h in hist if h["role"] == "interim").get("steer") is True   # ответ на дописанное – в ленту
        assert roles.index("user", 1) < roles.index("interim")                 # сначала ваше сообщение, потом ответ
        assert [x["id"] for x in await rt.sessions.list()] == ["s1"]           # никаких дополнительных диалогов


@pytest.mark.anyio
async def test_model_gate_one_request_at_a_time(tmp_path):
    """parallel = 1 (GigaChat Freemium): два диалога сразу – второй ждёт очереди к модели (model_wait)."""
    import asyncio
    from iacpaas_agent.core import AgentRuntime
    server, _ = make_server()
    models = write_models(tmp_path, {
        "g": {"provider": "fake", "model": "s", "cycle": True, "delay": 0.3, "parallel": 1,
              "script": [{"say": "ок"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        async def run(sid):
            return [e async for e in rt.stream(sid, text="вопрос", model="g")]
        a, b = await asyncio.gather(run("x1"), run("x2"))
        waits = [e for e in a + b if e["event"] == "model_wait"]
        assert len(waits) == 1 and a[-1]["status"] == b[-1]["status"] == "done"


def test_usage_counts_gigachat_cache_in_input():
    """GigaChat не включает кэш (precached_prompt_tokens) в input_tokens – «↑» должен считать всё прочитанное;
    у OpenAI-подобных кэш уже внутри input_tokens – не удваиваем."""
    from langchain_core.messages import AIMessage
    from iacpaas_agent.trace import usage
    giga = AIMessage("", usage_metadata={"input_tokens": 23, "output_tokens": 5, "total_tokens": 28,
                                         "input_token_details": {"cache_read": 3154}},
                     response_metadata={"token_usage": {"prompt_tokens": 23, "precached_prompt_tokens": 3154}})
    assert usage(giga) == {"tokens_in": 3177, "tokens_out": 5, "tokens_cached": 3154}
    oai = AIMessage("", usage_metadata={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107,
                                        "input_token_details": {"cache_read": 80}})
    assert usage(oai) == {"tokens_in": 100, "tokens_out": 7, "tokens_cached": 80}
    assert usage(None) == {"tokens_in": None, "tokens_out": None, "tokens_cached": 0}


@pytest.mark.anyio
async def test_gigachat_session_id_stable_per_model(tmp_path):
    """Кэш GigaChat: на время запроса к модели X-Session-ID (session_id_cvar) – постоянный для профиля модели
    (общая инструкция и инструменты берутся из кэша в любом диалоге; у режима – свой), после запроса – сброшен."""
    from gigachat.context import session_id_cvar
    from iacpaas_agent.core import AgentRuntime
    from iacpaas_agent.testing import ScriptedModel
    seen = []
    orig = ScriptedModel._agenerate

    async def spy(self, *a, **kw):
        seen.append(session_id_cvar.get())
        return await orig(self, *a, **kw)
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [{"say": "ок"}]}})
    ScriptedModel._agenerate = spy
    try:
        async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
            await rt.ask("dlg-1", "вопрос", model="a")
            await rt.ask("dlg-2", "вопрос", model="a")
    finally:
        ScriptedModel._agenerate = orig
    assert seen == ["iacpaas-agent-a", "iacpaas-agent-a"] and session_id_cvar.get() is None


def test_web_edit_question_compact_and_verify(tmp_path):
    """Веб: правка вопроса (edit) – вопрос и всё ниже удаляются, ход заново; «сжать весь чат» – выжимка и отметка
    в истории; итог проверки ответа кодом – в результате и в истории."""
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [
        {"tool": "count_graph", "args": {"path": IR}}, {"say": "По базе: {{@count_graph}}"}]}})
    app = create_app(settings(tmp_path, models), mcp_target=server)
    with TestClient(app, base_url="http://127.0.0.1:8770") as c:
        ev = _events(c.post("/api/stream", json={"message": "сколько вершин?", "model": "a"}).text)
        sid = ev[0]["session"]
        assert ev[-1]["verify"]["checked"] == 4 and not ev[-1]["verify"]["problems"]   # 11, 11, 1, 3 из куска
        for q in ("а дуг?", "а ещё?"):
            c.post("/api/stream", json={"session": sid, "message": q, "model": "a"})
        ev = _events(c.post("/api/stream", json={"session": sid, "message": "а дуг сколько?", "model": "a",
                                                 "edit": 1}).text)
        assert ev[-1]["status"] == "done"
        hist = c.get(f"/api/sessions/{sid}").json()["history"]
        assert [h["text"] for h in hist if h["role"] == "user"] == ["сколько вершин?", "а дуг сколько?"]
        assert hist[-1]["verify"]["checked"] == 4
        assert c.post("/api/stream", json={"session": sid, "message": "x", "edit": "два"}).status_code == 400
        # контекст в шапке: сколько прошлого уйдёт со следующим вопросом и бюджет модели; после «сжать» – выжимкой
        before = c.get(f"/api/context?session={sid}&keep=10&model=a").json()
        assert before["questions"] == 2 and before["full"] == 2 and before["tokens"] > 0 and before["budget"] > 0
        assert c.get("/api/context").status_code == 400 and c.get(f"/api/context?session={sid}&keep=x").status_code == 400
        r = c.post("/api/compact", json={"session": sid}).json()
        after = c.get(f"/api/context?session={sid}&keep=10&model=a").json()
        assert after["manual"] and after["digest"] == 2 and after["full"] == 0
        assert r["ok"] and r["questions"] == 2 and "1) «сколько вершин?» → ответ: Весь ИР: вершин 11, дуг 11" in r["digest"]
        assert [h["role"] for h in c.get(f"/api/sessions/{sid}").json()["history"]][-1] == "compact"
        assert c.post("/api/compact", json={}).status_code == 400
        assert c.get("/api/sessions").json()[0]["source"] == "web"
