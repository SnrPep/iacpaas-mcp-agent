"""Реестр известных ИР. REST API IACPaaS не умеет перечислять «Мой Фонд», поэтому список задаётся
в config/inforesources.json (или IACPAAS_KNOWN_IRS). В песочнице дополняется ИР из fixtures/."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .paths import PathError, normalize_ir_path

log = logging.getLogger("iacpaas_mcp.registry")


@dataclass
class IrEntry:
    path: str
    title: str
    description: str = ""
    kind: str = ""          # например: база знаний, онтология, база данных


def load_registry(file: Path, extra_paths: list[str], sandbox_paths: list[str]) -> list[IrEntry]:
    entries: dict[str, IrEntry] = {}
    if file.is_file():
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{file}: неверный JSON ({e})") from e
        for item in raw.get("inforesources", raw if isinstance(raw, list) else []):
            try:
                p = normalize_ir_path(item["path"])
            except (KeyError, PathError) as e:
                log.warning("registry_skip", extra={"item": item, "error": str(e)})
                continue
            entries[p] = IrEntry(path=p, title=item.get("title") or p.split("/")[-1],
                                 description=item.get("description", ""), kind=item.get("kind", ""))
    for p in [*extra_paths, *sandbox_paths]:
        p = normalize_ir_path(p)
        entries.setdefault(p, IrEntry(path=p, title=p.split("/")[-1]))
    return list(entries.values())
