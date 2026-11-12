"""Стенд для проверки инструкции агента (docs/prompt-eval.md): те же MCP-инструменты и ровно тот текст ответа,
что видит модель. Только чтение – сервер собирается из .env; запись выключена настройками.
    .venv\\Scripts\\python scripts\\prompt_eval_tool.py --list                     описания и схемы инструментов
    .venv\\Scripts\\python scripts\\prompt_eval_tool.py <имя> "<json-аргументы>"   вызов инструмента
Лог вызовов: logs/prompt_eval_calls.jsonl (поле run = переменная окружения EVAL_RUN)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
logging.disable(logging.CRITICAL)

from mcp import Client                                          # noqa: E402

from iacpaas_mcp.config import Settings, load_env               # noqa: E402
from iacpaas_mcp.server import build_server, create_backend     # noqa: E402

LOG = ROOT / "logs" / "prompt_eval_calls.jsonl"


async def main() -> None:
    load_env()
    s = Settings.from_env()
    s.log_file = None
    backend = create_backend(s)
    server = build_server(s, backend)
    try:
        async with Client(server) as c:
            if sys.argv[1] == "--list":
                for t in (await c.list_tools()).tools:
                    props = json.dumps(t.input_schema.get("properties", {}), ensure_ascii=False)
                    print(f"### {t.name}\n{t.description}\nargs: {props}\nrequired: {t.input_schema.get('required', [])}\n")
                return
            name = sys.argv[1]
            try:
                args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
            except json.JSONDecodeError as e:
                print(f"ОШИБКА: аргументы – не JSON: {e}")
                return
            started = time.perf_counter()
            try:
                r = await c.call_tool(name, args)
                err, out = r.is_error, r.content[0].text           # ровно то, что видит модель
            except Exception as e:                                  # noqa: BLE001 – стенд показывает любую ошибку
                err, out = True, f"{type(e).__name__}: {e}"
            print(("ОШИБКА ИНСТРУМЕНТА: " if err else "") + out)
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run": os.getenv("EVAL_RUN", "?"), "tool": name, "args": args, "error": err,
                                    "ms": int((time.perf_counter() - started) * 1000), "out": out[:300]},
                                   ensure_ascii=False) + "\n")
    finally:
        if hasattr(backend, "aclose"):
            await backend.aclose()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    asyncio.run(main())
