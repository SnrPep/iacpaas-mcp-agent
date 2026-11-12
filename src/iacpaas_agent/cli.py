"""Консольный агент.

  iacpaas-agent                        # модель AGENT_MODEL (auto), сервер запускается сам (stdio)
  iacpaas-agent --model gigachat       # конкретная модель
  iacpaas-agent --mcp http             # к уже запущенному `iacpaas-mcp --transport http`
  iacpaas-agent --session учеба        # продолжить сессию (контекст хранится в data/agent.sqlite)
  iacpaas-agent --new                  # новый диалог (без флага – продолжается последний диалог консоли)
  iacpaas-agent -q "вопрос"            # один вопрос без диалога (удобно для скриптов)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .core import COMPACT_ASK, AgentRuntime, TurnResult, is_question
from .models import describe
from .settings import AgentSettings

HELP = """Команды:
  /help                 эта справка
  /models               профили моделей и что для них нужно
  /model <имя>          сменить модель (контекст сессии сохраняется)
  /keys                 какие ключи заданы (значения не показываются)
  /key <имя>            задать ключ модели (ввод скрыт) или /key iacpaas – доступ к платформе
  /sessions             список сессий
  /session <id>         перейти в сессию (продолжить старый диалог)
  /new [название]       новая сессия
  /compact              сжать весь диалог: дальше ИИ видит прошлое выжимкой кодом (вопросы, числа, пути)
  /history              история текущей сессии
  /tools                инструменты MCP-сервера
  /exit                 выход (или пустая строка / Ctrl+C)"""


def _short(v, n: int = 160) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    s = " ".join(s.split())                     # многострочный JSON -> одна строка
    return s if len(s) <= n else s[: n - 1] + "…"


def print_event(e: dict) -> None:
    if e["event"] == "tool_call":
        print(f"  → {e['tool']}({_short(e.get('args', {}), 140)})", flush=True)
    elif e["event"] == "tool_result":
        mark = "✓" if e.get("status") == "success" else "✗"
        print(f"    {mark} {e.get('ms', '?')} мс  {_short(e.get('preview', ''), 160)}", flush=True)
    elif e["event"] == "tool_exception":
        print(f"    ✗ {e.get('error')}", flush=True)
    elif e["event"] == "model_wait":
        print(f"  … {e.get('text') or 'жду очереди к модели'}", flush=True)
    elif e["event"] == "verify" and e.get("status") == "retry":
        print("  ⚠ проверка кодом: ответ не совпал с данными – ИИ переписывает: "
              + _short("; ".join(e.get("problems") or []), 200), flush=True)


def print_wait(e: dict) -> None:
    """Из потока событий хода – ожидание модели и повтор после проверки (шаги печатает трасса)."""
    if e.get("event") in ("model_wait", "verify"):
        print_event(e)


def verify_line(v: dict) -> str:
    """Итог проверки ответа кодом одной строкой (как плашка под ответом в веб-чате)."""
    if v.get("skipped"):
        return "○ проверка кодом: ответ не из базы – сверять не с чем"
    if v.get("strict"):
        return "⚠ проверка кодом: ответ ИИ не прошёл проверку – выше данные из базы, собранные кодом"
    if v.get("problems"):
        return f"⚠ проверка кодом: не совпало с данными – {len(v['problems'])}"
    unv = f"; ○ не проверено – {len(v['unverified'])}" if v.get("unverified") else ""
    if n := v.get("checked"):
        word = "число" if n % 10 == 1 and n % 100 != 11 else "числа" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 \
            else "чисел"
        return ((f"✓ ответ собрал код из базы: {n} {word}" if v.get("blocks") else
                 f"✓ проверка кодом: сверено с данными {n} {word}")
                + (" (ответ переписан после проверки)" if v.get("retried") else "") + unv)
    return unv.lstrip("; ")


async def handle(rt: AgentRuntime, result: TurnResult, model: str, live: bool) -> TurnResult:
    while True:
        if result.status == "error":
            print(f"Ошибка: {result.error}")
            return result
        if result.status == "done":
            print(f"\nАгент ({result.model}): {result.answer}")
            if line := verify_line(result.verify):
                print(f"  {line}")
            return result
        # needs_approval
        flags = []
        for a in result.approvals:
            print(f"\n⚠ Агент хочет ИЗМЕНИТЬ данные: {a.tool}")
            print("  " + json.dumps(a.args, ensure_ascii=False, indent=2).replace("\n", "\n  "))
            try:
                ans = await asyncio.to_thread(input, "Разрешить? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                ans = "n"                                # нет ответа = отказ
            flags.append(ans.strip().lower() in ("y", "yes", "д", "да"))
        result = await rt.decide(result.session_id, flags, model=model)
        if not live:
            for e in result.events:
                print_event(e)


async def pick_session(rt: AgentRuntime, args) -> tuple[str, str]:
    """Какой диалог открыть: --session – этот; --new или -q – новый; иначе – последний диалог консоли (раньше
    каждый запуск начинал новый, и после перезапуска агент «не помнил» разговор)."""
    if args.session:
        if not await rt.sessions.get(args.session):
            await rt.sessions.create(session_id=args.session, source="cli")
        return args.session, ""
    if not (args.new or args.question) and (last := await rt.sessions.list(1, source="cli")):
        sid = last[0]["id"]
        n = sum(1 for m in await rt._saved(sid) if is_question(m))
        return sid, f"Продолжаю диалог «{last[0]['title'][:60]}» ({n} вопр.). Новый – /new или --new."
    return (await rt.sessions.create(source="cli"))["id"], ""


async def compact(rt: AgentRuntime, session: str) -> None:
    r = await rt.compact(session)
    if not r["ok"]:
        print(r["text"])
        return
    print(f"Контекст сжат кодом: {r['questions']} вопр. → выжимка (было ~{r['before']} ток., стало ~{r['after']}). "
          f"Дальше ИИ видит прошлое так:\n  {r['digest'][:2000]}")


async def repl(args) -> None:
    s = AgentSettings.from_env()
    if args.mcp:
        s.mcp = args.mcp
    if args.model:
        s.model = args.model
    model = s.model
    async with AgentRuntime(s) as rt:
        session, note = await pick_session(rt, args)
        print(f"IACPaaS-агент · модель: {model} · сессия: {session} · MCP: {s.mcp} · инструментов: {len(rt.tools)}")
        if note:
            print(note)
        if args.question:
            r = await rt.ask(session, args.question, model=model)
            for e in r.events:
                print_event(e)
            await handle(rt, r, model, live=False)
            return
        print("Пустая строка – выход, /help – команды.")
        try:
            pend = await rt.pending(session, model)
        except Exception as e:
            print(f"Модель недоступна: {e}. "
                  "/models – что нужно, /model <имя> – сменить")
            pend = []
        if pend:
            print("В этой сессии есть неподтверждённое изменение.")
            await handle(rt, TurnResult(session, model, "needs_approval", approvals=pend), model, live=False)
        while True:
            try:
                text = (await asyncio.to_thread(input, "\nВы: ")).strip().lstrip("﻿")   # BOM из канала PowerShell
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text == "/exit":
                break
            if text.startswith("/"):
                cmd, _, arg = text.partition(" ")
                arg = arg.strip()
                if cmd == "/help":
                    print(HELP)
                elif cmd == "/models":
                    for m in describe(s.models_file):
                        state = "готов" if m["ready"] else "нужно: " + ", ".join(m["missing"])
                        print(f"  {m['name']:<14} {m['provider']:<10} {m['model']:<40} {state}")
                elif cmd == "/keys":
                    from .keys import status
                    for k in status(s.models_file):
                        print(f"  {'✓' if k['ready'] else '·'} {k['name']:<13} "
                              + ", ".join(f"{v}{'' if ok else ' (нет)'}" for v, ok in k["vars"].items()))
                elif cmd == "/key" and arg:
                    from .keys import main as keys_main
                    try:
                        await asyncio.to_thread(keys_main, ["set", arg])
                        rt.reset_models()
                    except SystemExit as e:
                        if e.code not in (0, None):
                            print(e.code)
                elif cmd == "/model" and arg:
                    try:
                        _, _, model = rt.agent_for(arg)
                        print(f"Модель: {model}")
                    except Exception as e:
                        print(f"Ошибка: {e}")
                elif cmd == "/sessions":
                    for x in await rt.sessions.list():
                        mark = "*" if x["id"] == session else " "
                        print(f" {mark} {x['id']}  {x['title'][:60]}  ({x.get('model') or '-'})")
                elif cmd == "/session" and arg:
                    if not await rt.sessions.get(arg):
                        print("Нет такой сессии (/sessions – список)")
                    else:
                        session = arg
                        print(f"Сессия: {session}")
                elif cmd == "/new":
                    session = (await rt.sessions.create(title=arg, source="cli"))["id"]
                    print(f"Новая сессия: {session}")
                elif cmd == "/compact":
                    await compact(rt, session)
                elif cmd == "/history":
                    for h in await rt.history(session, model):
                        if h["role"] == "user":
                            print(f"Вы: {h['text']}")
                        elif h["role"] == "assistant":
                            print(f"Агент: {h['text']}")
                        elif h["role"] == "tool_call":
                            print(f"  → {h['tool']}({_short(h['args'], 120)})")
                        elif h["role"] == "compact":
                            print(f"  ── контекст сжат: {h['questions']} вопр. → выжимка ──")
                elif cmd == "/tools":
                    for t in rt.tools:
                        print(f"  {t.name:<24} {(t.description or '').splitlines()[0][:90]}")
                else:
                    print("Неизвестная команда. /help – справка.")
                continue
            if COMPACT_ASK.match(text):                  # «сожми контекст» – кодом, а не пересказом модели
                await compact(rt, session)
                continue
            try:
                _, trace, model = rt.agent_for(model)
            except Exception as e:
                print(f"Ошибка модели: {e}. "
                      "/models – что нужно для каждой модели, /model <имя> – сменить")
                continue
            trace.listeners.append(print_event)          # шаги агента печатаются по мере выполнения
            try:
                r = await rt.ask(session, text, model=model, emit=print_wait)
                await handle(rt, r, model, live=True)
            finally:
                trace.listeners.remove(print_event)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="iacpaas-agent", description="ИИ-агент для IACPaaS (MCP)")
    ap.add_argument("--model", help="профиль из config/models.toml (по умолчанию AGENT_MODEL или auto)")
    ap.add_argument("--mcp", choices=["stdio", "http", "inproc"], help="как подключаться к MCP-серверу")
    ap.add_argument("--session", help="id сессии для продолжения диалога")
    ap.add_argument("--new", action="store_true", help="начать новый диалог (без флага – продолжить последний)")
    ap.add_argument("-q", "--question", help="задать один вопрос и выйти")
    args = ap.parse_args(argv)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    try:
        asyncio.run(repl(args))
    except KeyboardInterrupt:
        pass
    except Exception as e:                       # понятное сообщение вместо трейсбека
        print(f"Не удалось запустить агента: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
