"""Настройки агента (.env). Пути относительно корня проекта."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from iacpaas_mcp.config import PROJECT_ROOT, load_env, project_path


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if not v else int(v)


@dataclass
class AgentSettings:
    model: str = "auto"                      # профиль из config/models.toml или auto
    models_file: Path = PROJECT_ROOT / "config" / "models.toml"
    prompt_file: Path = PROJECT_ROOT / "config" / "agent_prompt.md"
    mcp: str = "stdio"                       # stdio – агент сам запускает сервер; http – к запущенному; inproc – в процессе
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    mcp_token: str | None = None
    db_file: Path = PROJECT_ROOT / "data" / "agent.sqlite"
    sessions_file: Path = PROJECT_ROOT / "data" / "sessions.sqlite"
    trace_file: Path = PROJECT_ROOT / "logs" / "agent_trace.jsonl"
    max_tool_calls: int = 25                 # вызовов инструментов на один вопрос (вопросы списком – много частей)
    max_model_calls: int = 25                # обращений к модели на один вопрос
    # сколько сообщений прошлых вопросов (их результаты инструментов – заглушками) модель видит на каждом шаге;
    # текущий вопрос – всегда целиком, история в БД – тоже
    keep_messages: int = 15
    # инструменты, которые агенту не показываем: их схемы уходили бы модели на каждом шаге, а пользы нет
    # (сырой импорт JSON, проверка JSON, статус сервиса); в VS Code и Claude Code они остаются
    hide_tools: tuple[str, ...] = ("import_inforesource", "validate_resource", "service_status")
    tool_timeout_s: int = 120                # платформа до 30 с + повторы
    web_host: str = "127.0.0.1"
    web_port: int = 8770
    web_token: str | None = None
    lock_dir: Path | None = None        # папка очереди к модели (общая с другой копией проекта на том же ключе)

    @classmethod
    def from_env(cls) -> "AgentSettings":
        load_env()
        return cls(
            model=os.getenv("AGENT_MODEL", "auto"),
            models_file=project_path(os.getenv("AGENT_MODELS_FILE", "config/models.toml")),
            prompt_file=project_path(os.getenv("AGENT_PROMPT_FILE", "config/agent_prompt.md")),
            mcp=os.getenv("AGENT_MCP", "stdio").lower(),
            mcp_url=os.getenv("MCP_URL", "http://127.0.0.1:8765/mcp"),
            mcp_token=os.getenv("MCP_BEARER_TOKEN") or None,
            db_file=project_path(os.getenv("AGENT_DB", "data/agent.sqlite")),
            sessions_file=project_path(os.getenv("AGENT_SESSIONS_DB", "data/sessions.sqlite")),
            trace_file=project_path(os.getenv("AGENT_TRACE", "logs/agent_trace.jsonl")),
            max_tool_calls=_int("AGENT_MAX_TOOL_CALLS", 25),
            max_model_calls=_int("AGENT_MAX_MODEL_CALLS", 25),
            keep_messages=_int("AGENT_KEEP_MESSAGES", 15),
            hide_tools=tuple(x.strip() for x in os.getenv(
                "AGENT_HIDE_TOOLS", "import_inforesource,validate_resource,service_status").split(",") if x.strip()),
            tool_timeout_s=_int("AGENT_TOOL_TIMEOUT", 120),
            web_host=os.getenv("WEB_HOST", "127.0.0.1"),
            web_port=_int("WEB_PORT", 8770),
            web_token=os.getenv("WEB_TOKEN") or None,
            lock_dir=project_path(os.getenv("MODEL_LOCK_DIR")) if os.getenv("MODEL_LOCK_DIR") else None,
        )
