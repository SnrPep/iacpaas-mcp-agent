"""Веб-чат агента: http://127.0.0.1:8770 (только localhost).

  iacpaas-web                 # тот же агент, что в CLI: модели, сессии, подтверждения, трасса
Защита: слушает 127.0.0.1; запросы с чужим Origin отклоняются (CSRF/DNS-rebinding); WEB_TOKEN – по желанию."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..core import AgentRuntime, TurnResult
from ..models import describe
from ..settings import AgentSettings

STATIC = Path(__file__).parent / "static"
MAX_BODY = 1_000_000                      # байт на запрос
log = logging.getLogger("iacpaas_agent.web")


def _turn(r: TurnResult) -> dict:
    return asdict(r)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


class Guard:
    """Чистый ASGI-middleware (не BaseHTTPMiddleware): поток событий /api/stream идёт без буферизации,
    а разрыв соединения клиентом доходит до агента – он останавливается."""

    def __init__(self, app, settings: AgentSettings):
        self.app = app
        self.s = settings
        hosts = {"127.0.0.1", "localhost", settings.web_host}
        self.allowed_hosts = {f"{h}:{settings.web_port}" for h in hosts}
        self.allowed = {f"http://{h}" for h in self.allowed_hosts}

    def _reject(self, headers: dict[str, str], path: str) -> JSONResponse | None:
        # DNS-rebinding: чужой домен, указывающий на 127.0.0.1, приходит с чужим Host
        if headers.get("host") not in self.allowed_hosts:
            return JSONResponse({"error": "Недопустимый Host"}, status_code=421)
        origin = headers.get("origin")
        if origin and origin not in self.allowed:
            return JSONResponse({"error": "Запрос с чужого сайта отклонён"}, status_code=403)
        if self.s.web_token and path.startswith("/api/") and not hmac.compare_digest(
                (headers.get("x-web-token") or "").encode(), self.s.web_token.encode()):
            return JSONResponse({"error": "Нужен токен веб-чата (WEB_TOKEN)"}, status_code=401)
        if int(headers.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"error": "Слишком большой запрос"}, status_code=413)
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        if (bad := self._reject(headers, scope["path"])) is not None:
            return await bad(scope, receive, send)
        started = False

        async def tracked_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)
        try:
            await self.app(scope, receive, tracked_send)
        except Exception as e:                           # без трейсбеков наружу; детали – в консоль сервера
            log.exception("web_error")
            if not started:
                await JSONResponse({"error": f"Внутренняя ошибка: {type(e).__name__}"},
                                   status_code=500)(scope, receive, send)


def create_app(settings: AgentSettings | None = None, *, mcp_target=None) -> Starlette:
    s = settings or AgentSettings.from_env()
    rt = AgentRuntime(s, mcp_target=mcp_target)

    @asynccontextmanager
    async def lifespan(app):
        await rt.start()
        app.state.rt = rt
        yield
        await rt.close()

    async def index(_: Request):
        return FileResponse(STATIC / "index.html")

    async def local_tool(name: str, args: dict, default):
        """Вызов локального (без обращения к платформе) инструмента MCP для служебных сведений страницы."""
        tool = next((t for t in rt.tools if t.name == name), None)
        if not tool:
            return default
        try:
            raw = await tool.ainvoke(args)
            text = raw if isinstance(raw, str) else (raw[0].get("text") if isinstance(raw, list) and raw and
                                                    isinstance(raw[0], dict) else str(raw))
            return json.loads(text)
        except Exception as e:                      # сервер жив, но ответ не разобрался – не критично
            return {"error": str(e)} if isinstance(default, dict) else default

    async def health(_: Request):
        status = await local_tool("iacpaas_status", {}, {})
        irs = await local_tool("list_inforesources", {}, [])
        return JSONResponse({"ok": True, "mcp": s.mcp, "tools": len(rt.tools), "server": status,
                             "keep_messages": s.keep_messages,
                             "inforesources": irs if isinstance(irs, list) else [], "default_model": s.model})

    async def models(_: Request):
        from ..models import resolve_name

        def collect():                                    # проверка Ollama – сетевой запрос, не держим им сервер
            try:
                auto = resolve_name("auto", s.models_file)   # какую модель выберет «auto» прямо сейчас
            except Exception:
                auto = None
            return auto, describe(s.models_file)
        auto, items = await asyncio.to_thread(collect)
        return JSONResponse({"default": s.model, "auto": auto, "models": items})

    async def tools(_: Request):
        from ..core import is_write_tool, tool_kind
        return JSONResponse([{"name": t.name, "description": " ".join((t.description or "").split()),
                              "write": is_write_tool(t), "kind": tool_kind(t)} for t in rt.tools])

    async def sessions(request: Request):
        if request.method == "POST":
            body = await request.json() if await request.body() else {}
            return JSONResponse(await rt.sessions.create(title=(body.get("title") or "")[:80], source="web"))
        busy = rt.running()
        return JSONResponse([{**x, "running": x["id"] in busy} for x in await rt.sessions.list()])

    async def session_item(request: Request):
        sid = request.path_params["sid"]
        if request.method == "DELETE":
            await rt.sessions.delete(sid)
            await rt.checkpointer.adelete_thread(sid)
            return JSONResponse({"deleted": sid})
        meta = await rt.sessions.get(sid)
        model = request.query_params.get("model") or (meta or {}).get("model") or s.model
        hist = await rt.history(sid)                     # модель не нужна
        try:
            pend = [asdict(a) for a in await rt.pending(sid, model)]
        except Exception:                                # модели нет – подтверждений показать нельзя, история есть
            pend = []
        return JSONResponse({"session": meta, "history": hist, "pending": pend})

    async def keys(request: Request):
        # только ключи моделей; доступ к платформе – настройка MCP-сервера, а не чата: `iacpaas-keys set iacpaas`
        from ..keys import is_secret, profile_vars, status, write_env
        if request.method == "GET":
            rows = [r for r in status(s.models_file) if r["name"] != "iacpaas"]
            for r in rows:
                r["secret"] = {v: is_secret(v) for v in r["vars"]}
            return JSONResponse({"profiles": rows, "default_model": s.model})
        try:
            body = await request.json()
            name, values = body["name"], {k: str(v).strip() for k, v in (body.get("values") or {}).items()}
            if name == "iacpaas":
                return JSONResponse({"error": "Доступ к IACPaaS задаётся в консоли: iacpaas-keys set iacpaas"},
                                    status_code=400)
            allowed = set(profile_vars(s.models_file, name))
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON {name, values}"}, status_code=400)
        bad = [k for k in values if k not in allowed]
        if bad or not values or not all(values.values()):
            return JSONResponse({"error": f"Допустимые переменные: {sorted(allowed)}; все значения непустые"},
                                status_code=400)
        if body.get("make_default"):
            values["AGENT_MODEL"] = name
            s.model = name
        write_env(values)
        rt.reset_models()
        return JSONResponse({"ok": True, "saved": [k for k, v in values.items() if v],
                             "note": f"Ключ сохранён. Модель {name} готова."})

    async def chat(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "Ожидался JSON"}, status_code=400)
        text = (body.get("message") or "").strip()
        if not text:
            return JSONResponse({"error": "Пустое сообщение"}, status_code=400)
        sid = body.get("session") or (await rt.sessions.create(source="web"))["id"]
        return JSONResponse(_turn(await rt.ask(sid, text[:8000], model=body.get("model") or None)))

    async def decide(request: Request):
        try:
            body = await request.json()
            sid = body["session"]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON с полем session"}, status_code=400)
        return JSONResponse(_turn(await rt.decide(sid, body.get("approve", False),
                                                  model=body.get("model") or None)))

    async def stream(request: Request):
        """Ход агента потоком (Server-Sent Events): шаги модели и инструментов, текст ответа, итог."""
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "Ожидался JSON"}, status_code=400)
        text = (body.get("message") or "").strip()
        if not text and "approve" not in body:
            return JSONResponse({"error": "Пустое сообщение"}, status_code=400)
        sid = body.get("session") or (await rt.sessions.create(source="web"))["id"]
        model = body.get("model") or None
        try:                                             # длина памяти с ползунка; границы проверяет агент
            keep = int(body.get("keep") or 0) or None
            edit = int(body["edit"]) if body.get("edit") is not None else None   # правка вопроса номер edit
        except (TypeError, ValueError):
            return JSONResponse({"error": "keep и edit – числа"}, status_code=400)
        if text:
            events = rt.stream(sid, text=text[:8000], model=model, keep=keep, edit=edit)
        else:
            events = rt.stream(sid, approve=body.get("approve", False), model=model, keep=keep)

        async def gen():
            yield _sse({"event": "session", "session": sid})
            async for e in events:
                yield _sse(e)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def compact(request: Request):
        """«Сжать весь чат»: дальше модель видит прошлое выжимкой кодом; ответ – сама выжимка (что осталось модели)."""
        try:
            sid = (await request.json())["session"]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON с полем session"}, status_code=400)
        return JSONResponse(await rt.compact(str(sid)))

    async def context(request: Request):
        """Сколько прошлого диалога уйдёт модели со следующим вопросом при этой «Памяти диалога» (шапка чата)."""
        q = request.query_params
        if not q.get("session"):
            return JSONResponse({"error": "Нужен параметр session"}, status_code=400)
        try:
            keep = int(q.get("keep") or 0) or None
        except ValueError:
            return JSONResponse({"error": "keep – число"}, status_code=400)
        return JSONResponse(await rt.context(q["session"], keep, q.get("model") or None))

    async def stop(request: Request):
        try:
            sid = (await request.json())["session"]
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON с полем session"}, status_code=400)
        return JSONResponse({"stopped": rt.cancel(sid)})

    async def steer(request: Request):
        """Дописать сообщение, пока агент работает: модель учтёт его перед следующим шагом."""
        try:
            body = await request.json()
            sid, text = body["session"], str(body.get("message") or "")
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON с полями session и message"}, status_code=400)
        return JSONResponse({"accepted": rt.steer(sid, text[:8000], str(body.get("id") or "")[:40])})

    async def unsteer(request: Request):
        """Забрать дописанное сообщение, пока ИИ его не прочитал."""
        try:
            body = await request.json()
            sid, item = body["session"], str(body["id"])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Ожидался JSON с полями session и id"}, status_code=400)
        return JSONResponse({"removed": rt.unsteer(sid, item)})

    async def trace(request: Request):
        sid = request.path_params["sid"]
        body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rt.trace_for(sid))
        return Response(body, media_type="application/x-ndjson; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="trace-{sid}.jsonl"'})

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/models", models),
        Route("/api/tools", tools),
        Route("/api/sessions", sessions, methods=["GET", "POST"]),
        Route("/api/sessions/{sid}", session_item, methods=["GET", "DELETE"]),
        Route("/api/keys", keys, methods=["GET", "POST"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/decide", decide, methods=["POST"]),
        Route("/api/stream", stream, methods=["POST"]),
        Route("/api/stop", stop, methods=["POST"]),
        Route("/api/compact", compact, methods=["POST"]),
        Route("/api/context", context),
        Route("/api/steer", steer, methods=["POST"]),
        Route("/api/unsteer", unsteer, methods=["POST"]),
        Route("/api/sessions/{sid}/trace", trace),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),   # KaTeX и др. – локально, без интернета
    ]
    return Starlette(routes=routes, lifespan=lifespan, middleware=[Middleware(Guard, settings=s)])


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(prog="iacpaas-web", description="Веб-чат ИИ-агента IACPaaS")
    ap.add_argument("--port", type=int)
    ap.add_argument("--model")
    ap.add_argument("--mcp", choices=["stdio", "http", "inproc"])
    args = ap.parse_args(argv)
    s = AgentSettings.from_env()
    if args.port:
        s.web_port = args.port
    if args.model:
        s.model = args.model
    if args.mcp:
        s.mcp = args.mcp
    if s.web_host not in ("127.0.0.1", "localhost"):
        print("Внимание: веб-чат будет доступен из сети. Задайте WEB_TOKEN.", file=sys.stderr)
    print(f"Веб-чат: http://{s.web_host}:{s.web_port}  (Ctrl+C – остановить)", file=sys.stderr)
    uvicorn.run(create_app(s), host=s.web_host, port=s.web_port, log_level="warning")


if __name__ == "__main__":
    main()
