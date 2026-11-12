"""Структурные логи: одна JSON-строка на событие. Только stderr и файл – НИКОГДА stdout
(в режиме stdio stdout занят протоколом MCP). Секреты маскируются до записи."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from pathlib import Path

request_id: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
SECRET_KEYS = {"api_key", "token", "authorization", "password", "x-api-key", "confirm_token",
               "accesstoken", "jwt", "credentials"}


def redact(value):
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in SECRET_KEYS else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id.get(),
        }
        entry.update(redact({k: v for k, v in record.__dict__.items() if k not in _RESERVED}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup(level: str = "INFO", file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if file:
        file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
    root = logging.getLogger("iacpaas_mcp")
    root.handlers[:] = handlers
    root.setLevel(level)
    root.propagate = False
