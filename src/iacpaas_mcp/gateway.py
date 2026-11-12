"""Шлюз к платформе поверх бэкенда: кэш экспорта, повторы чтения, ограничение параллельности,
последовательная запись, сброс кэша после изменений. Учитывает ограничения IACPaaS:
до 30 с на запрос, сервисы import-run-export выполняются последовательно."""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any

from .backend.base import BackendError, IacpaasBackend

log = logging.getLogger("iacpaas_mcp.gateway")


class Gateway:
    def __init__(self, backend: IacpaasBackend, *, cache_ttl_s: float = 0.0, retries: int = 2,
                 max_concurrency: int = 2):
        self.backend = backend
        self.cache_ttl_s = cache_ttl_s
        self.retries = retries
        self._read_sem = asyncio.Semaphore(max(1, max_concurrency))
        self._write_lock = asyncio.Lock()
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._inflight: dict[tuple, asyncio.Future] = {}
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "errors": 0}

    # ------------------------------------------------------------------ read
    async def export(self, path: str, json_type: str = "universal", start: str | None = None,
                     depth: int | None = None) -> dict[str, Any]:
        key = (path, json_type, start or "", depth)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl_s:
            self.stats["cache_hits"] += 1
            return copy.deepcopy(hit[1])
        if key in self._inflight:                        # одинаковые запросы не дублируем
            return copy.deepcopy(await asyncio.shield(self._inflight[key]))
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            data = await self._read(lambda: self.backend.export(path, json_type, start, depth))
            self._cache[key] = (time.monotonic(), data)
            fut.set_result(data)
            return copy.deepcopy(data)
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()                               # чтобы не было «exception never retrieved»
            raise
        finally:
            self._inflight.pop(key, None)

    async def service_running(self, service_id: str) -> dict[str, Any]:
        return await self._read(lambda: self.backend.service_running(service_id))

    async def _read(self, call):
        delay = 1.0
        for attempt in range(self.retries + 1):
            async with self._read_sem:
                self.stats["requests"] += 1
                try:
                    return await call()
                except BackendError as e:
                    if not e.retryable or attempt == self.retries:
                        self.stats["errors"] += 1
                        raise
                    self.stats["retries"] += 1
                    log.warning("iacpaas_retry", extra={"attempt": attempt + 1, "kind": e.kind, "error": str(e)})
            await asyncio.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    # ----------------------------------------------------------------- write
    async def write(self, path_hint: str | None, call):
        """Запись: строго по одной, без повторов (неидемпотентна), затем сброс кэша."""
        async with self._write_lock:
            self.stats["requests"] += 1
            try:
                return await call()
            finally:
                self.invalidate(path_hint)

    def invalidate(self, path: str | None = None) -> None:
        if path is None:
            self._cache.clear()
        else:
            for k in [k for k in self._cache if k[0] == path or k[0].startswith(path.rstrip("/") + "/")]:
                del self._cache[k]
