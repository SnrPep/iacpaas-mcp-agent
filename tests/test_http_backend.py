"""HttpBackend против подменной «платформы»: запросы строго по RemoteCalls/OpenAPI, запись выключена."""
import json

import httpx
import pytest
from mcp import Client

from conftest import ROOT
from iacpaas_mcp.backend.base import BackendError
from iacpaas_mcp.backend.http import HttpBackend
from iacpaas_mcp.config import Settings
from iacpaas_mcp.gateway import Gateway
from iacpaas_mcp.server import build_server

pytestmark = pytest.mark.anyio

IR = {"title": "База", "json_type": "universal", "ontology": "o$;", "name": "База", "type": "КОРЕНЬ",
      "meta": "О", "successors": [{"name": "A", "type": "НЕТЕРМИНАЛ", "meta": "Группа"}]}


class FakePlatform:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.jwt_valid = "jwt-1"
        self.fail_next = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/signin":
            body = json.loads(request.content)
            if body["password"] != "secret":
                return httpx.Response(401)
            return httpx.Response(200, json={"accessToken": self.jwt_valid, "username": body["username"]})
        auth_ok = request.headers.get("X-API-KEY") == "KEY" or \
            request.headers.get("Authorization") == f"Bearer {self.jwt_valid}"
        if not auth_ok:
            return httpx.Response(401)
        if self.fail_next:
            self.fail_next -= 1
            return httpx.Response(503)
        if request.url.path == "/api/data/export/user-item":
            p = request.url.params["path"]
            if p == "нет":
                return httpx.Response(200, json={"success": False, "error": "not found",
                                                 "explanationSafe": "ИР не найден"})
            if p == "обёртка":
                return httpx.Response(200, json={"success": True, "result": json.dumps(IR, ensure_ascii=False)})
            if p == "Папка/Опечатка":
                return httpx.Response(200, json={"success": False, "error": "Ошибка", "explanation":
                                                 "Указан неверный путь к инфоресурсу: [student@example.ru, Мой Фонд, Папка, Опечатка]"})
            if request.url.params.get("start-target-concept-path") == "/Нет такой":
                return httpx.Response(200, json={"success": False, "error": "Ошибка", "explanation":
                                                 "StorageException: Заданное начальное понятие не найдено в "
                                                 "экспортируемом инфоресурсе по указанному пути: /Нет такой"})
            if p == "файл":
                return httpx.Response(200, json={"code": "46", "filename": "База.universal.json",
                                                 "data": json.dumps(IR, ensure_ascii=False)})
            return httpx.Response(200, json=IR)
        if request.url.path.endswith("/running"):
            return httpx.Response(200, json={"success": True, "running": False})
        return httpx.Response(200, json={"success": True, "code": "1"})


def backend(platform, **kw):
    return HttpBackend("https://iacpaas.example", transport=httpx.MockTransport(platform), **kw)


async def test_export_request_shape_api_key():
    pf = FakePlatform()
    data = await backend(pf, api_key="KEY").export("Учебные/База", start="Группа", depth=2)
    req = pf.requests[-1]
    assert req.method == "GET" and req.headers["X-API-KEY"] == "KEY"
    assert dict(req.url.params) == {"path": "Учебные/База", "json-type": "universal", "no-blob-data": "true",
                                    "start-target-concept-path": "/Группа", "export-depth": "2"}
    assert data["title"] == "База"


async def test_percent_in_paths_is_escaped():
    """Платформа ещё раз декодирует пути: «%» уходит как «%25» – и в запросе чтения, и в теле удаления."""
    pf = FakePlatform()
    b = backend(pf, api_key="KEY", write_enabled=True)
    await b.export("Песочница/Смесь 5%", start="Ar-95% He-5%")
    assert dict(pf.requests[-1].url.params)["path"] == "Песочница/Смесь 5%25"
    assert dict(pf.requests[-1].url.params)["start-target-concept-path"] == "/Ar-95%25 He-5%25"
    await b.delete("Песочница/Смесь 5%", "deleteIR")
    assert json.loads(pf.requests[-1].content)["path"] == "Песочница/Смесь 5%25"
    await b.delete("Песочница/A + B & C", "deleteIR")
    assert json.loads(pf.requests[-1].content)["path"] == "Песочница/A %2B B %26 C"


async def test_simple_and_meta_built_from_universal():
    pf = FakePlatform()
    b = backend(pf, api_key="KEY")
    simple = await b.export("Учебные/База", json_type="simple", start="A")
    assert pf.requests[-1].url.params["json-type"] == "universal"
    assert pf.requests[-1].url.params["start-target-concept-path"] == "/A"
    assert simple == {"База": {"A": "A"}}
    meta = await b.export("Учебные/База", json_type="meta")
    assert meta["$title"] == "База" and meta["Группа"][0]["$name"] == "A"


async def test_wrapper_and_errors():
    pf = FakePlatform()
    b = backend(pf, api_key="KEY")
    assert (await b.export("обёртка"))["title"] == "База"
    assert (await b.export("файл"))["successors"][0]["name"] == "A"
    with pytest.raises(BackendError) as e:
        await b.export("нет")
    assert e.value.kind == "not_found" and "ИР не найден" in str(e.value)
    with pytest.raises(BackendError) as e:
        await backend(pf, api_key="WRONG").export("x")
    assert e.value.kind == "auth"


async def test_signin_and_relogin_on_expired_jwt():
    pf = FakePlatform()
    b = backend(pf, username="u@x.ru", password="secret")
    await b.export("a")
    assert [r.url.path for r in pf.requests] == ["/api/signin", "/api/data/export/user-item"]
    pf.jwt_valid = "jwt-2"                     # токен «истёк»
    await b.export("a")
    assert [r.url.path for r in pf.requests][-3:] == ["/api/data/export/user-item", "/api/signin",
                                                      "/api/data/export/user-item"]
    with pytest.raises(BackendError) as e:
        await backend(FakePlatform(), username="u", password="bad").export("a")
    assert e.value.kind == "auth"


async def test_gateway_retries_reads_and_caches():
    pf = FakePlatform()
    gw = Gateway(backend(pf, api_key="KEY"), cache_ttl_s=60, retries=2)
    gw_sleep = __import__("asyncio").sleep
    pf.fail_next = 1                               # один 503 -> повтор -> успех
    import iacpaas_mcp.gateway as g
    g.asyncio.sleep = lambda s: gw_sleep(0)
    try:
        assert (await gw.export("a"))["title"] == "База"
        assert gw.stats["retries"] == 1
        n = len(pf.requests)
        await gw.export("a")
        assert len(pf.requests) == n and gw.stats["cache_hits"] == 1
    finally:
        g.asyncio.sleep = gw_sleep


async def test_writes_disabled_send_nothing():
    pf = FakePlatform()
    b = backend(pf, api_key="KEY")                 # write_enabled по умолчанию False
    for call in (b.import_ir("Учебные", IR, False),
                 b.update_concept("x", "a", "m", "new"),
                 b.delete("x", "deleteIR"),
                 b.run_service("1", [], []),
                 b.stop_service("1")):
        with pytest.raises(BackendError) as e:
            await call
        assert e.value.kind == "read_only"
    assert pf.requests == []                       # в «сеть» не ушло ни одного запроса


async def test_writes_enabled_request_shapes():
    pf = FakePlatform()
    b = backend(pf, api_key="KEY", write_enabled=True)
    await b.import_ir("Учебные", IR, False)
    req = pf.requests[-1]
    assert (req.method, req.url.path) == ("POST", "/api/data/import")
    body = json.loads(req.content)
    assert body["path"] == "Учебные" and json.loads(body["json"])["title"] == "База" and body["clearIfExists"] is False
    await b.update_concept("Учебные/База", "A", "Группа", "B", "комм")
    assert json.loads(pf.requests[-1].content) == {"path": "Учебные/База", "concept_path": "A",
                                                   "metaconcept_path": "Группа", "mark": "B", "comment": "комм"}
    await b.delete("Учебные/База", "clearSubGraph", "A", "Группа")
    assert pf.requests[-1].method == "DELETE" and json.loads(pf.requests[-1].content)["mode"] == "clearSubGraph"
    await b.run_service("42", [{"path": "Учебные", "json": IR}], [{"path": "Учебные/База", "json_type": "simple"}])
    body = json.loads(pf.requests[-1].content)
    assert body["id"] == "42" and body["export"] == [{"path": "Учебные/База", "jsonType": "simple"}]


async def test_server_over_http_backend_has_no_write_tools_even_if_allow_write():
    pf = FakePlatform()
    s = Settings(backend="http", api_key="KEY", allow_write=True, remote_write=False, log_file=None,
                 registry_file=ROOT / "config" / "inforesources.json", user_email="student@example.ru")
    b = backend(pf, api_key="KEY", write_enabled=s.allow_write and s.remote_write)
    server = build_server(s, b)
    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert "update_concept" not in names and "delete_from_fund" not in names
        info = (await c.call_tool("get_inforesource", {"path": "Учебные/База"})).structured_content
        assert info["title"] == "База"
        cnt = (await c.call_tool("count_graph", {"path": "Учебные/База", "root": "A"})).structured_content
        assert cnt["vertices"] == 1 and "start-target-concept-path" not in pf.requests[-1].url.params
        typo =(await c.call_tool("get_inforesource", {"path": "Папка/Опечатка"})).content[0].text
        assert "[not_found]" in typo and ("Похожие ИР" in typo or "list_inforesources" in typo)
        assert "student@example.ru" not in typo and "[Папка, Опечатка]" in typo      # e-mail не уходит модели
        vertex = (await c.call_tool("get_subgraph", {"path": "Учебные/База", "root": "Нет такой"})).content[0].text
        assert "[not_found]" in vertex and "Такой вершины нет" in vertex and "Похожие ИР" not in vertex
    assert all(r.method == "GET" for r in pf.requests)


async def test_remote_write_tools_only_in_own_folder():
    """Запись на платформу: подборки и удаление в своей папке; update_concept платформа не выполняет – его нет."""
    pf = FakePlatform()
    s = Settings(backend="http", api_key="KEY", allow_write=True, remote_write=True, log_file=None,
                 write_paths=["Загрузки/Песочница ИИ"], registry_file=ROOT / "config" / "inforesources.json")
    server = build_server(s, backend(pf, api_key="KEY", write_enabled=True))
    async with Client(server) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"save_selection", "save_mixture", "import_inforesource", "delete_from_fund"} <= names
    assert not names & {"update_concept", "run_service", "stop_service"}


async def test_check_platform_script_is_read_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_platform", ROOT / "scripts" / "check_platform.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pf = FakePlatform()
    rep = await mod.check("Учебные/База", backend(pf, api_key="KEY"), save=False)
    assert rep["export_depth0"]["title"] == "База" and rep["export_full"]["counts"]["vertices"] == 2
    assert rep["start_path_labels"]["ok"] is True
    assert pf.requests and all(r.method == "GET" for r in pf.requests)
