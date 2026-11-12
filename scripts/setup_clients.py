"""Готовит проект к подключению клиентов на ЭТОЙ машине. Запуск: .venv\\Scripts\\python scripts\\setup_clients.py

- .env из .env.example (если ещё нет) со случайным MCP_BEARER_TOKEN;
- .mcp.json – конфиг stdio-сервера для Claude Code, Cursor: абсолютный путь к Python из .venv, поэтому
  не зависит от рабочей папки и переменных клиента;
- .vscode/mcp.json – для VS Code (Copilot, режим Agent): основной stdio-сервер `iacpaas` VS Code берёт из .mcp.json
  (на него ссылается custom agent .github/agents/iacpaas.agent.md – tools: iacpaas/*); здесь – только
  HTTP-вариант `iacpaasHttp` (подключение к уже запущенному серверу с токеном).
Повторный запуск безопасен: .env не перезаписывается."""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def venv_python() -> Path:
    for p in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if p.exists():
            return p
    raise SystemExit("Не найдено .venv. Создайте окружение: python -m venv .venv  и установите зависимости (README).")


def main() -> None:
    py = venv_python()
    env = ROOT / ".env"
    if not env.exists():
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        text = text.replace("MCP_BEARER_TOKEN=                    #",
                            f"MCP_BEARER_TOKEN={secrets.token_urlsafe(24)}  #", 1)
        env.write_text(text, encoding="utf-8")
        print(f"Создан {env} (MCP_BEARER_TOKEN сгенерирован)")
    else:
        print(f"{env} уже есть – не трогаю")

    # IACPAAS_CONFIRM_MODE=chat: в VS Code / Claude Code подтверждение в чате – модель показывает предпросмотр и
    # спрашивает «да / нет», клиент спрашивает разрешение на вызов. Окна сервера (MCP elicitation) не используются:
    # панель Claude Code их не показывает и сама отвечает «отклонить». Веб-чат и консоль – кнопка агента
    mcp_json = {"mcpServers": {"iacpaas": {"type": "stdio", "command": str(py),
                                           "args": ["-m", "iacpaas_mcp", "--transport", "stdio"],
                                           "env": {"IACPAAS_CONFIRM_MODE": "chat", "IACPAAS_PIECES": "text"}}}}
    (ROOT / ".mcp.json").write_text(json.dumps(mcp_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записан .mcp.json (python: {py})")

    vscode = {
        "inputs": [{"type": "promptString", "id": "iacpaas-mcp-token", "password": True,
                    "description": "MCP_BEARER_TOKEN из .env (для HTTP-режима сервера)"}],
        # сервер `iacpaas` VS Code берёт из .mcp.json; второй такой же здесь дал бы двойника
        "servers": {"iacpaasHttp": {"type": "http", "url": "http://127.0.0.1:8765/mcp",
                                    "headers": {"Authorization": "Bearer ${input:iacpaas-mcp-token}"}}},
    }
    (ROOT / ".vscode").mkdir(exist_ok=True)
    (ROOT / ".vscode" / "mcp.json").write_text(json.dumps(vscode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Записан .vscode/mcp.json (HTTP-вариант iacpaasHttp; iacpaas – из .mcp.json)")
    print("\nГотово. Дальше: README.md → «Запуск».")


if __name__ == "__main__":
    if sys.platform == "win32":                          # консоль cp1251 не печатает «→»
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
