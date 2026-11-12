"""Первый контакт с реальной платформой IACPaaS – СТРОГО ТОЛЬКО ЧТЕНИЕ.

Запуск после получения доступа (заполните в .env IACPAAS_API_KEY или IACPAAS_USERNAME/IACPAAS_PASSWORD):
    .venv\\Scripts\\python scripts\\check_platform.py                    # все ИР из config/inforesources.json
    .venv\\Scripts\\python scripts\\check_platform.py "Папка/Имя ИР"     # конкретный ИР
    .venv\\Scripts\\python scripts\\check_platform.py --save             # + сохранить экспорт в fixtures/ для песочницы

Что проверяет: вход, экспорт с export-depth=0 и полный, форму ответа (сам ИР или обёртка), форматы meta/simple,
формат start-target-concept-path (метки от корня), время ответа, подсчёт вершин/дуг.
Отчёт: logs/platform_check.json. Изменяющих запросов скрипт не делает (клиент создаётся с запретом записи)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iacpaas_mcp import graph                                   # noqa: E402
from iacpaas_mcp.backend.base import BackendError               # noqa: E402
from iacpaas_mcp.backend.http import HttpBackend                # noqa: E402
from iacpaas_mcp.config import Settings, load_env               # noqa: E402
from iacpaas_mcp.paths import normalize_ir_path                 # noqa: E402
from iacpaas_mcp.registry import load_registry                  # noqa: E402


async def timed(coro):
    t = time.perf_counter()
    try:
        return await coro, round(time.perf_counter() - t, 2), None
    except BackendError as e:
        return None, round(time.perf_counter() - t, 2), f"[{e.kind}] {e}"


async def check(path: str, b: HttpBackend, save: bool) -> dict:
    rep: dict = {"path": path}
    head, sec, err = await timed(b.export(path, depth=0))
    rep["export_depth0"] = {"seconds": sec, "error": err,
                            "keys": sorted(head.keys())[:30] if head else None,
                            "title": head.get("title") if head else None,
                            "ontology": head.get("ontology") if head else None}
    if err:
        return rep
    full, sec, err = await timed(b.export(path))
    rep["export_full"] = {"seconds": sec, "error": err}
    if full:
        rep["export_full"]["counts"] = graph.count(full)
        rep["export_full"]["chars"] = len(json.dumps(full, ensure_ascii=False))
        rep["export_full"]["validate"] = graph.validate(full)[:10]
        first = (full.get("successors") or [{}])[0]
        if first:
            lab = graph.label(first)
            sub, sec, err = await timed(b.export(path, start=lab, depth=1))
            rep["start_path_labels"] = {"start": lab, "seconds": sec, "error": err,
                                        "ok": bool(sub and graph.find_node(sub, lab))}
        if save:
            base = (ROOT / "fixtures").resolve()
            out = (base / (normalize_ir_path(path).replace("\u2034", "") + ".json")).resolve()
            if base not in out.parents:
                raise SystemExit(f"Небезопасный путь для сохранения: {path!r}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
            rep["saved_to"] = str(out)
    for jt in ("meta", "simple"):
        data, sec, err = await timed(b.export(path, json_type=jt, depth=1))
        rep[f"export_{jt}"] = {"seconds": sec, "error": err, "keys": sorted(data.keys())[:15] if data else None}
    return rep


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--save", action="store_true", help="сохранить полный экспорт в fixtures/")
    args = ap.parse_args()
    load_env()
    s = Settings.from_env()
    if s.auth_mode == "none":
        raise SystemExit("Заполните в .env IACPAAS_API_KEY, IACPAAS_JWT или IACPAAS_USERNAME/IACPAAS_PASSWORD")
    b = HttpBackend(s.base_url, api_key=s.api_key, jwt=s.jwt, username=s.username, password=s.password,
                    timeout_s=s.timeout_s, write_enabled=False)          # запись запрещена всегда
    paths = args.paths or [e.path for e in load_registry(s.registry_file, s.known_irs, [])]
    if not paths:
        raise SystemExit("Укажите путь ИР аргументом или заполните config/inforesources.json")
    report = {"platform": s.base_url, "auth": s.auth_mode, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}
    try:
        for p in paths:
            r = await check(normalize_ir_path(p), b, args.save)
            report["results"].append(r)
            status = "OK" if not r["export_depth0"]["error"] else r["export_depth0"]["error"]
            print(f"{p}: {status}")
    finally:
        await b.aclose()
    out = ROOT / "logs" / "platform_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
