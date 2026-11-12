"""Управление ключами: модели (config/models.toml) и доступ к IACPaaS. Значения пишутся ТОЛЬКО в .env,
наружу (в консоль, веб, логи) не выводятся – показывается лишь «задан / не задан».

  iacpaas-keys                      # что задано, чего не хватает
  iacpaas-keys set gigachat         # спросит ключ (ввод скрыт) и запишет в .env
  iacpaas-keys set iacpaas          # доступ к платформе: API-ключ ИЛИ логин/пароль
  iacpaas-keys unset yandex         # стереть ключи профиля
  iacpaas-keys use gigachat         # сделать модель моделью по умолчанию (AGENT_MODEL)
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path

from iacpaas_mcp.config import PROJECT_ROOT

from .models import profiles

VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")
SECRET_HINTS = ("KEY", "TOKEN", "PASSWORD", "CREDENTIALS", "SECRET", "JWT")

# Доступ к платформе – «псевдопрофиль»: два способа входа.
IACPAAS_MODES = {
    "key": {"IACPAAS_API_KEY": "API-ключ от администратора платформы (заголовок X-API-KEY)"},
    "login": {"IACPAAS_USERNAME": "логин (e-mail) на iacpaas.dvo.ru",
              "IACPAAS_PASSWORD": "пароль на iacpaas.dvo.ru"},
}

HINTS = {
    "GIGACHAT_CREDENTIALS": "Authorization key из личного кабинета GigaChat API (developers.sber.ru)",
    "YANDEX_API_KEY": "API-ключ сервисного аккаунта Yandex AI Studio",
    "YANDEX_FOLDER_ID": "ID каталога Yandex Cloud (не секрет)",
    "DEEPSEEK_API_KEY": "ключ с platform.deepseek.com",
    "OPENROUTER_API_KEY": "ключ с openrouter.ai",
    "OPENAI_API_KEY": "ключ с platform.openai.com",
    "ANTHROPIC_API_KEY": "ключ с console.anthropic.com",
    "LLM_BASE_URL": "адрес OpenAI-совместимого API, например http://localhost:1234/v1 (не секрет)",
    "LLM_MODEL": "имя модели на этом сервере (не секрет)",
    "LLM_API_KEY": "ключ этого сервера (если не нужен – любой текст)",
}


def env_file() -> Path:
    explicit = os.getenv("IACPAAS_ENV_FILE")
    return Path(explicit) if explicit else PROJECT_ROOT / ".env"


def is_secret(var: str) -> bool:
    return any(h in var for h in SECRET_HINTS)


def profile_vars(models_file: Path, name: str) -> list[str]:
    if name == "iacpaas":
        return [v for m in IACPAAS_MODES.values() for v in m]
    p = profiles(models_file).get(name)
    if p is None:
        raise KeyError(name)
    found: list[str] = []

    def scan(v):
        if isinstance(v, str):
            found.extend(VAR.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
    scan(p)
    return list(dict.fromkeys(found))


def write_env(values: dict[str, str | None], file: Path | None = None) -> Path:
    """Обновляет/добавляет KEY=VALUE в .env, сохраняя комментарии и порядок. None – очистить значение.
    Сразу применяет к текущему процессу (os.environ)."""
    file = file or env_file()
    lines = file.read_text(encoding="utf-8-sig").splitlines() if file.exists() else []
    pending = dict(values)
    for i, raw in enumerate(lines):
        key = raw.split("=", 1)[0].strip() if "=" in raw and not raw.lstrip().startswith("#") else None
        if key in pending:
            comment = ""
            if " #" in raw.split("=", 1)[1]:
                comment = "  #" + raw.split("=", 1)[1].split(" #", 1)[1]
            val = pending.pop(key) or ""
            lines[i] = f"{key}={val}{comment}"
    for key, val in pending.items():
        lines.append(f"{key}={val or ''}")
    tmp = file.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(file)
    for key, val in values.items():
        if val:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)
    return file


def status(models_file: Path) -> list[dict]:
    rows = [{"name": "iacpaas", "kind": "доступ к платформе",
             "vars": {v: bool(os.getenv(v)) for m in IACPAAS_MODES.values() for v in m},
             "ready": bool(os.getenv("IACPAAS_API_KEY") or os.getenv("IACPAAS_JWT")
                           or (os.getenv("IACPAAS_USERNAME") and os.getenv("IACPAAS_PASSWORD")))}]
    for name, p in profiles(models_file).items():
        if p.get("provider") == "fake":
            continue
        vs = profile_vars(models_file, name)
        kind = f"модель {p.get('provider')}:{p.get('model')}"
        if p.get("provider") == "ollama":
            kind += " (нужна установленная и запущенная Ollama)"
        rows.append({"name": name, "kind": kind,
                     "vars": {v: bool(os.getenv(v)) for v in vs}, "ready": all(os.getenv(v) for v in vs)})
    return rows


def _ask(var: str) -> str:
    hint = HINTS.get(var) or next((d for m in IACPAAS_MODES.values() for k, d in m.items() if k == var), "")
    prompt = f"{var}" + (f" ({hint})" if hint else "") + ": "
    return (getpass.getpass(prompt) if is_secret(var) else input(prompt)).strip()


def main(argv: list[str] | None = None) -> None:
    from iacpaas_mcp.config import load_env
    from .settings import AgentSettings
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    models_file = AgentSettings.from_env().models_file
    ap = argparse.ArgumentParser(prog="iacpaas-keys", description="Ключи моделей и доступ к IACPaaS (в .env)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="что задано (по умолчанию)")
    s = sub.add_parser("set", help="задать ключ(и) профиля")
    s.add_argument("name", help="профиль модели из config/models.toml или iacpaas")
    s.add_argument("--mode", choices=list(IACPAAS_MODES), help="для iacpaas: key – API-ключ, login – логин/пароль")
    u = sub.add_parser("unset", help="стереть ключи профиля")
    u.add_argument("name")
    us = sub.add_parser("use", help="модель по умолчанию (AGENT_MODEL)")
    us.add_argument("name")
    args = ap.parse_args(argv)

    if args.cmd in (None, "list"):
        print(f"Файл: {env_file()}")
        for r in status(models_file):
            mark = "✓" if r["ready"] else "·"
            vs = ", ".join(f"{k}{'' if v else ' (нет)'}" for k, v in r["vars"].items()) or "ключ не нужен"
            print(f" {mark} {r['name']:<13} {r['kind']:<46} {vs}")
        print(f"\nМодель по умолчанию: {os.getenv('AGENT_MODEL', 'auto')}.  Задать: iacpaas-keys set <имя>")
        return

    try:
        names = profile_vars(models_file, args.name)
    except KeyError:
        raise SystemExit(f"Нет профиля '{args.name}'. Список: iacpaas-keys list")

    if args.cmd == "use":
        if args.name == "iacpaas":
            raise SystemExit("iacpaas – это доступ к платформе, а не модель")
        write_env({"AGENT_MODEL": args.name})
        print(f"AGENT_MODEL={args.name}")
        return

    if args.cmd == "unset":
        write_env({v: None for v in names})
        print(f"Стёрто: {', '.join(names)}")
        return

    # set
    if args.name == "iacpaas":
        mode = args.mode or ("login" if input("Вход: 1 – API-ключ от администратора, 2 – логин/пароль [2]: ").strip()
                             in ("", "2") else "key")
        names = list(IACPAAS_MODES[mode])
    if not names:
        print("Этому профилю ключ не нужен.")
        return
    values = {}
    for var in names:
        v = _ask(var)
        if not v:
            raise SystemExit(f"Пусто – ничего не записано ({var}).")
        values[var] = v
    if args.name == "iacpaas":
        other = [v for m, vs in IACPAAS_MODES.items() if m != mode for v in vs]
        values.update({v: None for v in other})            # один способ входа, без путаницы
        if input("Переключить сервер на реальную платформу (IACPAAS_BACKEND=http)? [y/N]: ").strip().lower() in ("y", "д", "да"):
            values["IACPAAS_BACKEND"] = "http"
    f = write_env(values)
    print(f"Записано в {f}: {', '.join(k for k, v in values.items() if v)} (значения скрыты)")
    if args.name == "iacpaas":
        print("Перезапустите MCP-сервер: в Claude Code – /mcp → Reconnect, в VS Code – MCP: List Servers → Restart, "
              "агента/веб-чат – перезапустить. Проверка доступа: .venv\\Scripts\\python scripts\\check_platform.py")
    else:
        print(f"Готово. Запуск: iacpaas-agent --model {args.name}   (или iacpaas-keys use {args.name})")


if __name__ == "__main__":
    main()
