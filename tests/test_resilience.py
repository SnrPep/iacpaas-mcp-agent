"""Сбои и атаки: система должна отвечать понятной ошибкой и оставаться рабочей."""
import asyncio
import json
import os

import httpx
import pytest
from langchain_core.tools import StructuredTool
from mcp import Client
from starlette.testclient import TestClient

from conftest import IR, ROOT, make_server, make_settings
from iacpaas_agent.core import AgentRuntime
from iacpaas_agent.web.app import create_app
from iacpaas_mcp import gateway as gw_mod
from iacpaas_mcp.backend.http import HttpBackend
from iacpaas_mcp.config import load_env
from iacpaas_mcp.server import build_server
from test_agent import settings, write_models

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    real = asyncio.sleep
    monkeypatch.setattr(gw_mod.asyncio, "sleep", lambda s: real(0))


def http_server(handler, **kw):
    s = make_settings(backend="http", api_key="KEY", retries=2, **kw)
    b = HttpBackend("https://iacpaas.example", api_key="KEY", transport=httpx.MockTransport(handler))
    return build_server(s, b)


async def test_platform_timeout_is_retried_then_reported():
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("slow", request=req)
    async with Client(http_server(handler)) as c:
        r = await c.call_tool("get_inforesource", {"path": IR})
    assert r.is_error and "[timeout]" in r.content[0].text and "повторить позже" in r.content[0].text
    assert len(calls) == 3                                     # 1 + 2 повтора


async def test_platform_unavailable_flag_and_recovery():
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json={"success": False, "platformUnavailable": True,
                                             "explanationSafe": "Платформа на обслуживании"})
        return httpx.Response(200, json={"title": "База", "type": "КОРЕНЬ", "meta": "О", "successors": []})
    async with Client(http_server(handler)) as c:
        r = await c.call_tool("get_inforesource", {"path": IR})
    assert not r.is_error and r.structured_content["title"] == "База"


async def test_auth_error_not_retried_and_explained():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(401)
    async with Client(http_server(handler)) as c:
        r = await c.call_tool("count_graph", {"path": IR})
    assert r.is_error and "[auth]" in r.content[0].text and "сообщите пользователю" in r.content[0].text
    assert len(calls) == 1


async def test_garbage_response_is_reported():
    async with Client(http_server(lambda req: httpx.Response(200, text="<html>oops</html>"))) as c:
        r = await c.call_tool("get_inforesource", {"path": IR})
    assert r.is_error and "не JSON" in r.content[0].text


async def test_oversized_json_rejected():
    server, _ = make_server(allow_write=True)
    big = {"title": "X", "type": "КОРЕНЬ", "json_type": "universal", "ontology": "o$;", "meta": "m",
           "comment": "x" * 2_100_000}
    async with Client(server) as c:
        r = await c.call_tool("validate_resource", {"ir_json": big})
        assert r.is_error and "[too_large]" in r.content[0].text
        r = await c.call_tool("import_inforesource", {"folder_path": "Учебные", "ir_json": big})
        assert r.is_error and "[too_large]" in r.content[0].text


async def test_tool_crash_keeps_dialog_consistent(tmp_path):
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "count_graph", "args": {"path": IR}}, {"say": "Данные недоступны"},
        {"say": "второй ответ"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        real = next(t for t in rt.tools if t.name == "count_graph")

        async def boom(**_):
            raise ConnectionError("MCP-сервер упал")
        rt.tools[rt.tools.index(real)] = StructuredTool.from_function(
            coroutine=boom, name="count_graph", description=real.description, args_schema=real.args_schema,
            metadata=real.metadata)                         # аннотации MCP: это инструмент чтения
        r = await rt.ask("c1", "посчитай")
        assert r.status == "done" and r.answer == "Данные недоступны"
        res = next(e for e in r.events if e["event"] == "tool_result")
        assert res["status"] == "error" and "[tool_unavailable]" in res["preview"]
        assert "упал" not in res["preview"]                    # внутренние детали модели не показываются
        r2 = await rt.ask("c1", "ещё")                           # история не сломана, диалог продолжается
        assert r2.status == "done" and r2.answer == "второй ответ"


async def test_model_failure_does_not_break_session(tmp_path, monkeypatch):
    from iacpaas_agent import testing
    from langchain.agents.middleware import ModelRetryMiddleware
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [{"say": "работаю"}]}})
    fail = {"left": 3}
    orig = testing.ScriptedModel._agenerate

    async def flaky(self, *a, **kw):
        if fail["left"] > 0:
            fail["left"] -= 1
            raise ConnectionError("провайдер модели недоступен")
        return await orig(self, *a, **kw)
    monkeypatch.setattr(testing.ScriptedModel, "_agenerate", flaky)
    real_init = ModelRetryMiddleware.__init__
    monkeypatch.setattr(ModelRetryMiddleware, "__init__",
                        lambda self, **kw: real_init(self, **{**kw, "initial_delay": 0.01, "jitter": False}))
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("m1", "привет")
        assert r.status in ("done", "error")                   # не исключение
        assert "ConnectionError" in (r.answer + r.error)
        r2 = await rt.ask("m1", "ещё раз")
        assert r2.status == "done" and r2.answer == "работаю"


async def test_concurrent_requests_same_session(tmp_path):
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"say": "1"}, {"say": "2"}, {"say": "3"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        rs = await asyncio.gather(*(rt.ask("same", f"q{i}") for i in range(3)))
        assert sorted(r.answer for r in rs) == ["1", "2", "3"] and all(r.status == "done" for r in rs)
        assert [h["role"] for h in await rt.history("same")].count("user") == 3


def test_web_rejects_foreign_host_and_bad_input(tmp_path):
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [{"say": "ok"}]}})
    app = create_app(settings(tmp_path, models), mcp_target=server)
    with TestClient(app, base_url="http://evil.example:8770") as c:
        assert c.get("/api/sessions").status_code == 421       # DNS-rebinding
    with TestClient(app, base_url="http://127.0.0.1:8770") as c:
        assert c.post("/api/chat", content=b"not json", headers={"content-type": "application/json"}).status_code == 400
        assert c.post("/api/decide", json={}).status_code == 400
        assert c.post("/api/chat", content=b"x" * 1_100_000).status_code == 413
        assert c.post("/api/chat", json={"message": "   "}).status_code == 400


def test_explicit_env_file_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.delenv("ZZ_TEST", raising=False)
    (tmp_path / "a.env").write_text('ZZ_TEST="из файла"  # комментарий\n', encoding="utf-8")
    assert load_env(tmp_path / "a.env") and os.environ["ZZ_TEST"] == "из файла"
    monkeypatch.delenv("ZZ_TEST")
    monkeypatch.setenv("IACPAAS_ENV_FILE", str(tmp_path / "нет.env"))
    assert load_env() is None and "ZZ_TEST" not in os.environ      # не откатывается к .env проекта


def test_check_platform_refuses_path_outside_fixtures(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cp", ROOT / "scripts" / "check_platform.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def handler(req):
        return httpx.Response(200, json={"title": "x", "type": "КОРЕНЬ", "meta": "m", "successors": []})
    b = HttpBackend("https://iacpaas.example", api_key="KEY", transport=httpx.MockTransport(handler))
    with pytest.raises(SystemExit):
        asyncio.run(mod.check("../../../evil", b, save=True))


async def test_repeated_failing_call_is_not_executed_again(tmp_path):
    """Нет обязательного параметра – ошибка по-русски; тот же вызов второй раз не выполняется."""
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "count_graph", "args": {"root": "x"}}, {"tool": "count_graph", "args": {"root": "x"}},
        {"say": "Не получилось"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("g1", "посчитай")
    res = [e for e in r.events if e["event"] == "tool_result"]
    assert "Не хватает обязательного параметра «path»" in res[0]["preview"]
    assert res[1]["preview"].startswith("[repeat]") and r.answer == "Не получилось"


async def test_own_list_shared_between_server_processes(tmp_path):
    """Свои ИР – общий файл у всех процессов сервера (веб-чат, VS Code). Удалённое в одном окне не «воскресает»
    из памяти другого; ИР, которого на платформе уже нет, сам убирается из списка."""
    own = tmp_path / "own.json"
    gone = "Загрузки/Песочница ИИ/Старая подборка"
    own.write_text(json.dumps([{"path": gone, "title": "Старая подборка", "description": "", "kind": "своя подборка"},
                               {"path": "Загрузки/Песочница ИИ/Живая", "title": "Живая", "description": "",
                                "kind": "своя подборка"}], ensure_ascii=False), encoding="utf-8")

    deleted = set()

    def handler(req):
        name = req.url.params.get("path", "")
        if name.endswith("Старая подборка") and deleted:
            return httpx.Response(404)
        return httpx.Response(200, json={"title": name.rsplit("/", 1)[-1], "type": "КОРЕНЬ", "meta": "М"})
    web, vscode = (http_server(handler, own_registry_file=own) for _ in range(2))
    async with Client(web) as a, Client(vscode) as b:
        assert (await b.call_tool("list_inforesources", {"filter": "Песочница"})).structured_content["total"] == 2
        deleted.add(gone)                                                # удалили на сайте или в другом окне
        r = await a.call_tool("get_inforesource", {"path": gone})       # в веб-чате: на платформе его уже нет
        assert r.is_error and "уже нет" in r.content[0].text
        lst = (await b.call_tool("list_inforesources", {"filter": "Песочница"})).structured_content
        assert [x["title"] for x in lst["items"]] == ["Живая"]           # второй процесс видит это сразу
    assert gone not in own.read_text(encoding="utf-8")


async def test_confirm_dialog_in_client_with_elicitation(tmp_path):
    """VS Code / Claude Code / Copilot: второй шаг записи открывает человеку окно сервера (MCP elicitation) –
    модель с токеном сама подтвердить не может. «Отклонить» – ничего не изменено; клиент без окна в строгом
    режиме (IACPAAS_REQUIRE_DIALOG) – запись выключена."""
    from mcp import types
    from test_agent import UPDATE
    asked, answer = [], {"action": "accept"}

    async def human(context, params):
        asked.append(params.message)
        return types.ElicitResult(action=answer["action"], content={"approve": True} if answer["action"] == "accept"
                                  else None)
    server, backend = make_server(allow_write=True)
    node = lambda: backend.store[IR]["successors"][0]["successors"][1]["name"]   # noqa: E731
    async with Client(server, elicitation_callback=human) as c:
        pre = (await c.call_tool("update_concept", UPDATE)).structured_content
        assert pre["status"] == "pending_confirmation" and "окно подтверждения" in pre["message"] and not asked
        answer["action"] = "decline"
        no = (await c.call_tool("update_concept", {**UPDATE, "confirm_token": pre["confirm_token"]})).structured_content
        assert no["status"] == "rejected" and node() == "Светобоязнь"
        assert "Фотофобия" in asked[0] and "Разрешить?" in asked[0]                # в окне – предпросмотр
        answer["action"] = "accept"
        pre = (await c.call_tool("update_concept", UPDATE)).structured_content
        ok = (await c.call_tool("update_concept", {**UPDATE, "confirm_token": pre["confirm_token"]})).structured_content
        assert ok["status"] == "done" and node() == "Фотофобия" and len(asked) == 2
        fake = await c.call_tool("update_concept", {**UPDATE, "mark": "Х", "confirm_token": pre["confirm_token"]})
        assert fake.is_error and "[confirmation]" in fake.content[0].text and len(asked) == 2   # без окна не пройдёт
    strict, _ = make_server(allow_write=True, require_dialog=True)
    async with Client(strict) as c:                                           # клиент без окна
        r = await c.call_tool("update_concept", UPDATE)
        assert r.is_error and "не умеет показывать окно" in r.content[0].text


async def test_dialog_cancel_is_not_a_human_reject():
    """Claude Code в VS Code ответил на окно «cancel» сам, окна человек не видел, а сервер писал «вы отклонили».
    cancel – отдельный итог: ничего не изменено, модели – подсказка, где сохранить с кнопкой."""
    from mcp import types
    from test_agent import UPDATE

    async def no_window(context, params):
        return types.ElicitResult(action="cancel")
    server, backend = make_server(allow_write=True)
    async with Client(server, elicitation_callback=no_window) as c:
        pre = (await c.call_tool("update_concept", UPDATE)).structured_content
        r = (await c.call_tool("update_concept", {**UPDATE, "confirm_token": pre["confirm_token"]})).structured_content
    assert r["status"] == "cancelled" and "не показал" in r["message"] and "веб-чате" in r["message"]
    assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"


async def test_chat_mode_confirms_in_chat_not_in_window():
    """IACPAAS_CONFIRM_MODE=chat (VS Code / Claude Code): окно сервера не открывается, даже если клиент его
    объявляет (панель Claude Code сама отвечает «отклонить»); модель показывает предпросмотр и спрашивает в чате."""
    from test_agent import UPDATE
    asked = []

    async def window(context, params):
        asked.append(params.message)
        return types.ElicitResult(action="decline")
    from mcp import types
    server, backend = make_server(allow_write=True, confirm_mode="chat")
    async with Client(server, elicitation_callback=window) as c:
        pre = (await c.call_tool("update_concept", UPDATE)).structured_content
        assert "спросите, выполнить ли" in pre["message"]
        ok = (await c.call_tool("update_concept", {**UPDATE, "confirm_token": pre["confirm_token"]})).structured_content
    assert ok["status"] == "done" and not asked
    assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Фотофобия"


async def test_own_list_drops_irs_deleted_on_site(tmp_path):
    """Подборки удалили на сайте – «какие ИР» и схема их больше не называют: сервер сверяет свои ИР с платформой."""
    own = tmp_path / "own.json"
    own.write_text(json.dumps([{"path": f"Загрузки/Песочница ИИ/{n}", "title": n, "description": "",
                                "kind": "своя подборка"} for n in ("Удалённая", "Живая")], ensure_ascii=False),
                   encoding="utf-8")

    def handler(req):
        name = req.url.params.get("path", "")
        if name.endswith("Удалённая"):
            return httpx.Response(404)
        return httpx.Response(200, json={"title": name.rsplit("/", 1)[-1], "type": "КОРЕНЬ", "meta": "М"})
    async with Client(http_server(handler, own_registry_file=own)) as c:
        lst = (await c.call_tool("list_inforesources", {"filter": "Песочница"})).structured_content
    assert [x["title"] for x in lst["items"]] == ["Живая"] and "Удалённая" not in own.read_text(encoding="utf-8")


def test_model_retry_error_goes_to_trace(tmp_path):
    """«ServerError после 3 попыток» – текст каждой ошибки шага модели в трассе (раньше повторы его глотали);
    429 не повторяется."""
    from iacpaas_agent.core import retry_on
    from iacpaas_agent.trace import JsonlTrace
    t = JsonlTrace(tmp_path / "t.jsonl", "m")

    class ServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass
    from gigachat.context import session_id_cvar
    token = session_id_cvar.set("iacpaas-agent-gigachat")
    assert retry_on(t, ServerError("500 Internal Server Error")) is True
    assert session_id_cvar.get().startswith("iacpaas-retry-")      # повтор – без общего кэша GigaChat
    session_id_cvar.reset(token)
    assert retry_on(t, RateLimitError("429")) is False
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["model_retry", "model_retry"]
    assert rows[0]["error"] == "ServerError: 500 Internal Server Error" and rows[0]["again"] is True
    assert rows[1]["again"] is False
