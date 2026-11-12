"""Ключи: запись в .env без потери комментариев, статус без значений, веб-панель."""
import os

from starlette.testclient import TestClient

from conftest import ROOT, make_server
from iacpaas_agent import keys
from iacpaas_agent.web.app import create_app
from test_agent import settings, write_models


def test_write_env_preserves_file_and_hides_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# шапка\nGIGACHAT_CREDENTIALS=   # ключ сюда\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setenv("IACPAAS_ENV_FILE", str(env))
    keys.write_env({"GIGACHAT_CREDENTIALS": "секрет", "NEW_VAR": "x"})
    text = env.read_text(encoding="utf-8")
    assert "# шапка" in text and "GIGACHAT_CREDENTIALS=секрет  # ключ сюда" in text and "OTHER=1" in text
    assert "NEW_VAR=x" in text and os.environ["GIGACHAT_CREDENTIALS"] == "секрет"
    rows = {r["name"]: r for r in keys.status(ROOT / "config" / "models.toml")}
    assert rows["gigachat"]["ready"] and rows["gigachat"]["vars"] == {"GIGACHAT_CREDENTIALS": True}
    assert "секрет" not in str(rows)                                   # значения не возвращаются
    keys.write_env({"GIGACHAT_CREDENTIALS": None, "NEW_VAR": None})
    assert "GIGACHAT_CREDENTIALS" not in os.environ


def test_profile_vars():
    mf = ROOT / "config" / "models.toml"
    assert keys.profile_vars(mf, "yandex") == ["YANDEX_FOLDER_ID", "YANDEX_API_KEY"]
    assert keys.profile_vars(mf, "ollama-qwen") == []
    assert set(keys.profile_vars(mf, "iacpaas")) == {"IACPAAS_API_KEY", "IACPAAS_USERNAME", "IACPAAS_PASSWORD"}


def test_web_keys_panel(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setenv("IACPAAS_ENV_FILE", str(env))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [{"say": "ok"}]}})
    s = settings(tmp_path, models)
    s.models_file = ROOT / "config" / "models.toml"
    with TestClient(create_app(s, mcp_target=server), base_url="http://127.0.0.1:8770") as c:
        info = c.get("/api/keys").json()
        assert any(p["name"] == "deepseek" and not p["ready"] for p in info["profiles"])
        bad = c.post("/api/keys", json={"name": "deepseek", "values": {"EVIL_VAR": "1"}})
        assert bad.status_code == 400                                     # только переменные профиля
        ok = c.post("/api/keys", json={"name": "deepseek", "values": {"DEEPSEEK_API_KEY": "sk-1"},
                                       "make_default": True}).json()
        assert ok["ok"] and "AGENT_MODEL" in ok["saved"]
        assert "sk-1" not in c.get("/api/keys").text
        login = c.post("/api/keys", json={"name": "iacpaas", "values": {"IACPAAS_USERNAME": "u@x.ru",
                                                                          "IACPAAS_PASSWORD": "p"}})
        assert login.status_code == 400                                   # доступ к платформе – только iacpaas-keys
    text = env.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=sk-1" in text and "AGENT_MODEL=deepseek" in text and "IACPAAS_PASSWORD" not in text
    for k in ("DEEPSEEK_API_KEY", "AGENT_MODEL", "IACPAAS_USERNAME", "IACPAAS_PASSWORD"):
        os.environ.pop(k, None)
