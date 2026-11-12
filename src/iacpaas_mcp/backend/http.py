"""Клиент REST API IACPaaS.

Документация: https://iacpaas.dvo.ru/docs/RemoteCalls/RemoteCalls.html, OpenAPI: /v3/api-docs.yaml.
Чтение:
  POST /api/signin                      – получить JWT по логину/паролю (данные не меняет)
  GET  /api/data/export/user-item       – экспорт ИР / подграфа
  GET  /api/service/{id}/running        – статус сервиса
Запись (ВЫКЛЮЧЕНА по умолчанию, включается только IACPAAS_REMOTE_WRITE=true):
  POST   /api/storage-unit/folder/create – создать папку в «Мой Фонд»
  POST   /api/data/import               – создать / перезаписать ИР
  PUT    /api/data/update               – изменить метку/комментарий вершины
  DELETE /api/data/delete               – удалить/очистить ИР или подграф
  POST   /api/service/import-run-export – запустить безынтерфейсный сервис
  PUT    /api/service/{id}/stop         – остановить сервис
Пока запись выключена, методы записи отказывают ДО отправки чего-либо в сеть.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from .. import graph
from .base import BackendError

log = logging.getLogger("iacpaas_mcp.http")

READ_ONLY_MESSAGE = ("Запись на платформу IACPaaS выключена (IACPAAS_REMOTE_WRITE=false): "
                     "с реальной платформой разрешено только чтение")



def _esc(path: str) -> str:
    """Платформа ещё раз декодирует пути как адрес: «%», «+», «&» в имени ИР или вершины («Ar-95% He-5%»)
    ломают путь, а «%25», «%2B», «%26» доходят как есть."""
    return path.replace("%", "%25").replace("+", "%2B").replace("&", "%26") if path else path

class HttpBackend:
    name = "http"

    def __init__(self, base_url: str, *, api_key: str | None = None, jwt: str | None = None,
                 username: str | None = None, password: str | None = None, timeout_s: float = 60.0,
                 write_enabled: bool = False, transport: httpx.AsyncBaseTransport | None = None):
        if not (api_key or jwt or (username and password)):
            raise ValueError("Для IACPAAS_BACKEND=http нужен IACPAAS_API_KEY, IACPAAS_JWT "
                             "или пара IACPAAS_USERNAME/IACPAAS_PASSWORD")
        self.writable = write_enabled          # по умолчанию False: никаких изменяющих запросов
        self._api_key, self._jwt = api_key, jwt
        self._username, self._password = username, password
        self._signin_lock = asyncio.Lock()
        # Платформа может обрабатывать запрос до 30 с – таймаут чтения должен быть больше.
        self._http = httpx.AsyncClient(base_url=base_url, transport=transport,
                                       timeout=httpx.Timeout(timeout_s, connect=10.0),
                                       headers={"Accept": "application/json"})

    # ------------------------------------------------------------------ auth
    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"X-API-KEY": self._api_key}
        if self._jwt:
            return {"Authorization": f"Bearer {self._jwt}"}
        return {}

    async def _signin(self) -> None:
        async with self._signin_lock:
            log.info("iacpaas_signin", extra={"user": self._username})
            try:
                resp = await self._http.post("/api/signin", json={"username": self._username,
                                                                  "password": self._password})
            except httpx.HTTPError as e:
                raise BackendError(f"Не удалось войти в IACPaaS: {e}", kind="network", retryable=True) from e
            if resp.status_code != 200:
                raise BackendError(f"Вход в IACPaaS отклонён (HTTP {resp.status_code}): проверьте логин и пароль",
                                   kind="auth", status=resp.status_code)
            token = (resp.json() or {}).get("accessToken")
            if not token:
                raise BackendError("Ответ /api/signin без accessToken", kind="auth")
            self._jwt = token

    # --------------------------------------------------------------- request
    async def _request(self, method: str, url: str, **kw: Any) -> Any:
        if method != "GET" and not self.writable:          # последний рубеж: в сеть не уходит ничего
            raise BackendError(READ_ONLY_MESSAGE, kind="read_only")
        if not self._api_key and not self._jwt and self._username:
            await self._signin()
        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                resp = await self._http.request(method, url, headers=self._auth_headers(), **kw)
            except httpx.TimeoutException as e:
                raise BackendError("IACPaaS не ответила вовремя (платформа отводит на запрос до 30 с)",
                                   kind="timeout", retryable=method == "GET") from e
            except httpx.HTTPError as e:
                raise BackendError(f"Сеть недоступна: {e.__class__.__name__}: {e}", kind="network",
                                   retryable=method == "GET") from e
            log.info("iacpaas_http", extra={"method": method, "url": url, "status": resp.status_code,
                                            "ms": int((time.perf_counter() - started) * 1000)})
            if resp.status_code == 401 and attempt == 1 and self._username and not self._api_key:
                self._jwt = None                            # JWT истёк – входим заново один раз
                await self._signin()
                continue
            return self._parse(resp, retry_ok=method == "GET")
        raise BackendError("Нет доступа к IACPaaS после повторного входа", kind="auth")

    @staticmethod
    def _parse(resp: httpx.Response, retry_ok: bool) -> Any:
        if resp.status_code in (401, 403):
            raise BackendError("Нет доступа к IACPaaS: проверьте ключ/токен (доступны только ваши ИР)",
                               kind="auth", status=resp.status_code)
        if resp.status_code == 404:
            raise BackendError("Ресурс не найден на платформе", kind="not_found", status=404)
        if resp.status_code in (502, 503, 504):
            raise BackendError(f"Платформа временно недоступна (HTTP {resp.status_code})", kind="unavailable",
                               retryable=retry_ok, status=resp.status_code)
        if resp.status_code >= 400:
            raise BackendError(f"IACPaaS вернула HTTP {resp.status_code}: {resp.text[:300]}",
                               status=resp.status_code, retryable=retry_ok and resp.status_code >= 500)
        if not resp.content:
            return {}
        try:
            data = resp.json()
        except ValueError as e:
            raise BackendError(f"Ответ платформы – не JSON: {resp.text[:200]!r}") from e
        # Обёртки Result/CodeResult/ExportResult: success, error, explanation(Safe), platformUnavailable
        if isinstance(data, dict) and data.get("success") is False:
            msg = data.get("explanationSafe") or data.get("explanation") or data.get("error") or "ошибка"
            unavailable = bool(data.get("platformUnavailable"))
            low = str(msg).lower()
            missing = any(w in low for w in ("найден", "not found", "неверный путь"))
            kind = "unavailable" if unavailable else ("not_found" if missing else "platform_error")
            raise BackendError(f"IACPaaS: {msg}", kind=kind, retryable=unavailable and retry_ok)
        return data

    # ------------------------------------------------------------------ read
    async def export(self, path, json_type="universal", start=None, depth=None, no_blob_data=True):
        if json_type in ("simple", "meta"):
            # платформа отдаёт simple/meta некорректным JSON (пустые значения) и без учёта начальной вершины,
            # поэтому они строятся из universal – тот же граф, тот же фрагмент
            ir = await self.export(path, "universal", start, depth, no_blob_data)
            return graph.to_meta(ir) if json_type == "meta" else {graph.label(ir): graph.to_simple(ir)}
        params: dict[str, Any] = {"path": _esc(path), "json-type": json_type,
                                  "no-blob-data": "true" if no_blob_data else "false"}
        if start:
            # платформа требует путь от корня с ведущим '/' (корень не указывается): "/Правила/1"
            params["start-target-concept-path"] = "/" + _esc(start.lstrip("/"))
        if depth is not None:
            params["export-depth"] = str(depth)
        data = await self._request("GET", "/api/data/export/user-item", params=params)
        # Ответ – сам ИР или обёртка: {"success": true, "result": "<JSON строкой>"} либо
        # {"code": ..., "filename": "<имя>.universal.json", "data": "<JSON строкой>"} (так отвечает iacpaas.dvo.ru)
        if isinstance(data, dict) and not ({"title", "$title"} & data.keys()):
            key = "result" if "success" in data else "data" if {"data", "filename"} <= data.keys() else None
            if key:
                payload = data.get(key)
                data = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(data, dict):
            raise BackendError("Неожиданный формат ответа экспорта (ожидался JSON-объект ИР)")
        return data

    async def service_running(self, service_id):
        data = await self._request("GET", f"/api/service/{service_id}/running")
        return {"running": bool(data.get("running")) if isinstance(data, dict) else None}

    # ----------------------------------------------- write (выключено по умолчанию)
    async def create_folder(self, parent, name, description=""):
        parent, name = _esc(parent), _esc(name)
        # POST /api/storage-unit/folder/create (есть в OpenAPI, нет в RemoteCalls): path – папка-родитель в «Мой Фонд»
        return await self._request("POST", "/api/storage-unit/folder/create",
                                   json={"path": parent, "name": name, "description": description})

    async def import_ir(self, folder_path, ir_json, clear_if_exists):
        # path – ПАПКА в «Мой Фонд» (без имени ИР), json – весь ИР одной строкой
        body = {"path": _esc(folder_path), "json": json.dumps(ir_json, ensure_ascii=False),
                "clearIfExists": bool(clear_if_exists)}
        return await self._request("POST", "/api/data/import", json=body)

    async def update_concept(self, path, concept_path, metaconcept_path, mark, comment=None):
        body = {"path": _esc(path), "concept_path": _esc(concept_path), "metaconcept_path": _esc(metaconcept_path),
                "mark": mark}
        if comment is not None:
            body["comment"] = comment
        return await self._request("PUT", "/api/data/update", json=body)

    async def delete(self, path, mode, concept_path=None, metaconcept_path=None):
        body: dict[str, Any] = {"path": _esc(path), "mode": mode}
        if concept_path:
            body["concept_path"] = _esc(concept_path)
        if metaconcept_path:
            body["metaconcept_path"] = _esc(metaconcept_path)
        return await self._request("DELETE", "/api/data/delete", json=body)

    async def run_service(self, service_id, imports, exports, no_blob_data=True):
        # Платформа обрабатывает такие запросы последовательно; сервис должен уложиться в 30 с.
        body = {
            "id": service_id,
            "noBlobData": bool(no_blob_data),
            "import": [{"path": i["path"],
                        "json": i["json"] if isinstance(i["json"], str) else json.dumps(i["json"], ensure_ascii=False),
                        "clearIfExists": bool(i.get("clear_if_exists", False))} for i in imports],
            "export": [_export_item(e) for e in exports],
        }
        data = await self._request("POST", "/api/service/import-run-export", json=body)
        return data if isinstance(data, list) else [data]

    async def stop_service(self, service_id):
        return await self._request("PUT", f"/api/service/{service_id}/stop")

    async def aclose(self):
        await self._http.aclose()


def _export_item(e: dict[str, Any]) -> dict[str, Any]:
    # Внимание: в import-run-export поле называется jsonType (а в /export – json-type).
    item: dict[str, Any] = {"path": e["path"], "jsonType": e.get("json_type", "universal")}
    if e.get("start"):
        item["startTargetConceptPath"] = e["start"]
    if e.get("depth") is not None:
        item["exportDepth"] = str(e["depth"])
    return item
