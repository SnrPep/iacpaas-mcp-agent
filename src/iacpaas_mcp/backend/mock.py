"""Песочница: инфоресурсы в памяти, загруженные из fixtures/. Ведёт себя как REST API IACPaaS
(те же параметры и виды ошибок), но ничего не отправляет в сеть. Изменения живут до перезапуска процесса.

Раскладка фикстур: fixtures/<папка>/<Имя ИР>.json  ->  путь ИР "<папка>/<Имя ИР>".
Сюда удобно класть реальные экспорты с платформы (universal)."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path

from .. import graph
from ..paths import split_labels
from .base import BackendError


class MockBackend:
    name = "mock"
    writable = True

    def __init__(self, fixtures_dir: Path, latency_ms: int | None = None):
        self.fixtures_dir = fixtures_dir
        self.latency = (latency_ms if latency_ms is not None else int(os.getenv("MOCK_LATENCY_MS", "0"))) / 1000
        self.store: dict[str, dict] = {}
        self.services: dict[str, bool] = {}
        if fixtures_dir.is_dir():
            for f in sorted(fixtures_dir.rglob("*.json")):
                rel = f.relative_to(fixtures_dir).with_suffix("")
                ir = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(ir, dict) and ir.get("type") == "КОРЕНЬ":
                    self.store["/".join(rel.parts)] = ir

    async def _tick(self) -> None:
        if self.latency:
            await asyncio.sleep(self.latency)

    def _get(self, path: str) -> dict:
        if path not in self.store:
            raise BackendError(f"Инфоресурс '{path}' не найден в «Мой Фонд»", kind="not_found")
        return self.store[path]

    @staticmethod
    def _walk_to(ir: dict, concept_path: str, *, keep_only_path: bool) -> dict:
        node = ir
        for lab in split_labels(concept_path):
            children = node.get("successors", []) or []
            match = next((c for c in children if graph.label(c) == lab), None)
            if match is None:
                raise BackendError(f"Вершина '{lab}' не найдена (путь '{concept_path}'). "
                                   f"Потомки: {graph.children_labels(node)}", kind="not_found")
            if keep_only_path:
                node["successors"] = [match]     # как на платформе: путь от корня тоже экспортируется
            node = match
        return node

    # ------------------------------------------------------------------ read
    async def export(self, path, json_type="universal", start=None, depth=None, no_blob_data=True):
        await self._tick()
        ir = copy.deepcopy(self._get(path))
        node = self._walk_to(ir, start, keep_only_path=True) if start else ir
        if depth is not None and depth >= 0:
            def cut(n: dict, d: int) -> None:
                if d <= 0:
                    n.pop("successors", None)
                    return
                for c in n.get("successors", []) or []:
                    cut(c, d - 1)
            cut(node, depth)
        if json_type == "meta":
            return graph.to_meta(ir)
        if json_type == "simple":
            return {graph.label(ir): graph.to_simple(ir)}
        return ir

    async def service_running(self, service_id):
        await self._tick()
        return {"running": self.services.get(service_id, False)}

    # ----------------------------------------------------------------- write
    async def create_folder(self, parent, name, description=""):
        await self._tick()
        path = f"{parent.strip('/')}/{name}" if parent.strip("/") else name
        self.folders = getattr(self, "folders", set()) | {path}
        return {"success": True, "path": path}

    async def import_ir(self, folder_path, ir_json, clear_if_exists):
        await self._tick()
        title = ir_json.get("title")
        if not title:
            raise BackendError("В JSON нет title", kind="invalid")
        path = f"{folder_path.strip('/')}/{title}" if folder_path.strip("/") else title
        if path in self.store and not clear_if_exists:
            raise BackendError(f"ИР '{path}' уже существует; для перезаписи нужен clearIfExists=true",
                               kind="conflict")
        self.store[path] = copy.deepcopy(ir_json)
        return {"success": True, "code": str(abs(hash(path)) % 10**12), "path": path}

    async def update_concept(self, path, concept_path, metaconcept_path, mark, comment=None):
        await self._tick()
        ir = self._get(path)
        labels, metas = split_labels(concept_path), split_labels(metaconcept_path)
        if len(labels) != len(metas):
            raise BackendError("metaconcept_path должен содержать по одной метке метавершины на каждую "
                               f"вершину concept_path ({len(labels)} ≠ {len(metas)})", kind="invalid")
        node = ir
        for lab, meta in zip(labels, metas):
            node = next((c for c in node.get("successors", []) or [] if graph.label(c) == lab), None)
            if node is None:
                raise BackendError(f"Вершина '{lab}' не найдена (путь '{concept_path}')", kind="not_found")
            if node.get("meta") != meta:
                raise BackendError(f"У вершины '{lab}' метавершина '{node.get('meta')}', а не '{meta}'",
                                   kind="invalid")
        if "link" in node:
            raise BackendError("Это ссылка на вершину; меняйте саму вершину по её пути", kind="invalid")
        if mark:
            node["value" if node.get("type") == "ТЕРМИНАЛ-ЗНАЧЕНИЕ" else "name"] = mark
        if comment is not None:
            node["comment"] = comment
        return {"success": True}

    async def delete(self, path, mode, concept_path=None, metaconcept_path=None):
        await self._tick()
        ir = self._get(path)
        if mode == "deleteIR" or mode == "trashIR":
            del self.store[path]
        elif mode == "clearIR":
            ir["successors"] = []
        elif mode in ("deleteSubGraph", "clearSubGraph"):
            if not concept_path:
                raise BackendError("Для удаления подграфа нужен concept_path", kind="invalid")
            labels = split_labels(concept_path)
            parent = self._walk_to(ir, "/".join(labels[:-1]), keep_only_path=False) if len(labels) > 1 else ir
            target = next((c for c in parent.get("successors", []) or [] if graph.label(c) == labels[-1]), None)
            if target is None:
                raise BackendError(f"Вершина '{concept_path}' не найдена", kind="not_found")
            if mode == "deleteSubGraph":
                parent["successors"].remove(target)
            else:
                target["successors"] = []
        else:
            raise BackendError(f"Неизвестный режим удаления {mode!r}", kind="invalid")
        return {"success": True}

    async def run_service(self, service_id, imports, exports, no_blob_data=True):
        await self._tick()
        if self.services.get(service_id):
            raise BackendError(f"Сервис {service_id} уже запущен", kind="conflict")
        for item in imports:
            data = item["json"] if isinstance(item["json"], dict) else json.loads(item["json"])
            await self.import_ir(item["path"], data, item.get("clear_if_exists", False))
        results = []
        for e in exports:
            try:
                data = await self.export(e["path"], e.get("json_type", "universal"), e.get("start"), e.get("depth"))
                results.append({"success": True, "result": json.dumps(data, ensure_ascii=False)})
            except BackendError as err:
                results.append({"success": False, "error": str(err)})
        return results

    async def stop_service(self, service_id):
        self.services[service_id] = False
        return {"success": True}

    async def aclose(self):
        return None
