"""Интерфейс доступа к IACPaaS и общие ошибки."""
from __future__ import annotations

from typing import Any, Protocol

JSON_TYPES = ("universal", "meta", "simple")
DELETE_MODES = ("deleteIR", "trashIR", "clearIR", "deleteSubGraph", "clearSubGraph")


class BackendError(Exception):
    """Ошибка платформы в понятном агенту виде.
    kind: auth | not_found | timeout | network | unavailable | read_only | conflict | invalid | platform_error"""

    def __init__(self, message: str, *, kind: str = "platform_error", retryable: bool = False,
                 status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status = status


class IacpaasBackend(Protocol):
    name: str
    writable: bool        # можно ли в принципе менять данные через этот бэкенд

    async def export(self, path: str, json_type: str = "universal", start: str | None = None,
                     depth: int | None = None, no_blob_data: bool = True) -> dict[str, Any]: ...

    async def service_running(self, service_id: str) -> dict[str, Any]: ...

    # --- изменяющие операции (выключены по умолчанию, см. policy) ---
    async def create_folder(self, parent: str, name: str, description: str = "") -> dict[str, Any]: ...
    async def import_ir(self, folder_path: str, ir_json: dict[str, Any], clear_if_exists: bool) -> dict[str, Any]: ...
    async def update_concept(self, path: str, concept_path: str, metaconcept_path: str, mark: str,
                             comment: str | None) -> dict[str, Any]: ...
    async def delete(self, path: str, mode: str, concept_path: str | None, metaconcept_path: str | None) -> dict[str, Any]: ...
    async def run_service(self, service_id: str, imports: list[dict], exports: list[dict],
                          no_blob_data: bool) -> list[dict[str, Any]]: ...
    async def stop_service(self, service_id: str) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...
