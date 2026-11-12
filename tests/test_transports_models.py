"""Настоящие транспорты (подпроцесс stdio, HTTP с Bearer) и совместимость схем со всеми провайдерами моделей."""
import os
import socket
import subprocess
import sys
import time
import warnings

import httpx
import pytest
from mcp import Client, StdioServerParameters

from conftest import IR, ROOT, make_server

pytestmark = pytest.mark.anyio


def child_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IACPAAS_", "MCP_", "LOG_"))}
    env.update({"PYTHONIOENCODING": "utf-8", "IACPAAS_ENV_FILE": str(ROOT / "tests" / "no.env"),
                "LOG_FILE": "", "IACPAAS_BACKEND": "mock"}, **extra)
    return env


async def test_stdio_subprocess():
    params = StdioServerParameters(command=sys.executable, args=["-m", "iacpaas_mcp"], env=child_env(),
                                   cwd=str(ROOT))
    async with Client(params) as c:
        r = await c.call_tool("count_graph", {"path": IR})
        assert r.structured_content["vertices"] == 11
        assert c.protocol_version in ("2026-07-28", "2025-11-25")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_http_requires_bearer_token():
    port, token = _free_port(), "test-token-0123456789"
    proc = subprocess.Popen([sys.executable, "-m", "iacpaas_mcp", "--transport", "http", "--port", str(port)],
                            env=child_env(MCP_BEARER_TOKEN=token), cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        for _ in range(100):
            try:
                httpx.get(url, timeout=0.3, trust_env=False)
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        no = httpx.post(url, json={}, timeout=5, trust_env=False)
        assert no.status_code == 401 and "Bearer" in no.headers["www-authenticate"]
        from fastmcp.client import Client as FClient
        async with FClient(url, auth=token) as c:
            res = await c.call_tool("list_inforesources", {})
            assert any(i["path"] == IR for i in res.structured_content["items"])
        with pytest.raises(Exception):
            async with FClient(url, auth="wrong-token-0123456789") as c:
                await c.list_tools()
    finally:
        proc.terminate()
        proc.wait(10)


def test_http_refuses_to_start_without_token():
    r = subprocess.run([sys.executable, "-m", "iacpaas_mcp", "--transport", "http", "--port", "1"],
                       env=child_env(), cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 2 and "MCP_BEARER_TOKEN" in r.stderr


async def test_every_model_profile_accepts_all_tool_schemas(monkeypatch):
    """Все профили из config/models.toml создаются и принимают схемы всех 22 инструментов
    (GigaChat, например, отвергает anyOf – поэтому это проверяется)."""
    warnings.filterwarnings("ignore")
    from langchain.mcp import MCPAdapter
    from iacpaas_agent.models import make_model, profiles
    for var in ("GIGACHAT_CREDENTIALS", "YANDEX_API_KEY", "YANDEX_FOLDER_ID", "DEEPSEEK_API_KEY",
                "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.setenv(var, "dummy")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    server, _ = make_server(allow_write=True)
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
    assert len(tools) == 21
    models_file = ROOT / "config" / "models.toml"
    for name in profiles(models_file):
        _, model = make_model(name, models_file)
        model.bind_tools(tools)                                      # упадёт, если схема несовместима


def test_auto_skips_ollama_without_downloaded_model(monkeypatch, tmp_path):
    """auto берёт Ollama-профиль, только если сервер запущен И модель профиля скачана."""
    from iacpaas_agent import models
    cfg = tmp_path / "models.toml"
    cfg.write_text('auto = ["big", "small"]\n'
                   '[big]\nprovider = "ollama"\nmodel = "qwen3:8b"\n'
                   '[small]\nprovider = "ollama"\nmodel = "qwen3:4b-instruct-2507-q4_K_M"\n', encoding="utf-8")
    tags = {"models": [{"name": "qwen3:4b-instruct-2507-q4_K_M"}]}
    monkeypatch.setattr(models.httpx, "get", lambda *a, **k: httpx.Response(200, json=tags))
    models._TAGS.clear()
    assert models.resolve_name("auto", cfg) == "small"          # 8b не скачана – пропущена
    tags["models"] = []
    models._TAGS.clear()                                         # ответ Ollama помнится 30 с
    with pytest.raises(models.ModelConfigError):
        models.resolve_name("auto", cfg)                         # сервер есть, моделей нет

    def down(*a, **k):
        raise httpx.ConnectError("нет сервера")
    monkeypatch.setattr(models.httpx, "get", down)
    models._TAGS.clear()
    with pytest.raises(models.ModelConfigError):
        models.resolve_name("auto", cfg)                         # Ollama не запущена


def test_gigachat_certificate_path(monkeypatch, tmp_path):
    """Сертификат Минцифры: относительный путь – от корня проекта; нет файла – понятная ошибка, а не FileNotFoundError."""
    from iacpaas_agent import models
    cfg = tmp_path / "models.toml"
    cfg.write_text('[g]\nprovider = "gigachat"\nmodel = "GigaChat-2"\ncredentials = "${GIGACHAT_CREDENTIALS}"\n',
                   encoding="utf-8")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "dummy")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE_FILE", "certs/нет-такого.cer")
    with pytest.raises(models.ModelConfigError, match="сертификата Минцифры"):
        models.make_model("g", cfg)
    cert = tmp_path / "ca.cer"
    cert.write_text("x", encoding="utf-8")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE_FILE", str(cert))
    _, m = models.make_model("g", cfg)
    assert m.ca_bundle_file == str(cert)


async def test_title_parameter_survives_langchain_conversion():
    """Параметр с именем title (подборка, смесь) должен дойти до модели: LangChain убирает «title» из схем."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from mcp import Client
    from conftest import make_server
    server, _ = make_server(allow_write=True)
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    for name in ("save_selection", "save_mixture"):
        fn = convert_to_openai_tool({"name": name, "description": "x", "parameters": tools[name].input_schema})
        assert "title" in fn["function"]["parameters"]["properties"], name
        assert "Path" not in str(tools[name].input_schema)                  # остальные заголовки убраны
