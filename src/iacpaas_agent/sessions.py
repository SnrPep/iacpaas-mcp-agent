"""Список сессий (диалогов): имя, модель, даты, откуда (консоль / веб), отметка «сжато вручную». Сами сообщения
хранит LangGraph-чекпоинтер (data/agent.sqlite)."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import aiosqlite

SCHEMA = """CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT, created REAL NOT NULL, updated REAL NOT NULL)"""
# новые столбцы добавляются к старой таблице (ALTER TABLE): source – cli / web, compact_id – id последнего
# сообщения, сжатого вручную («сжать весь чат»); mode – режим ответа: code – ответ собирает код, model – ответ
# пишет модель, сверяет код
EXTRA = {"source": "TEXT", "compact_id": "TEXT"}
COLS = ("id", "title", "model", "created", "updated", *EXTRA)


class SessionStore:
    def __init__(self, file: Path):
        self.file = file
        self.db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.file)
        await self.db.execute(SCHEMA)
        async with self.db.execute("PRAGMA table_info(sessions)") as cur:
            have = {r[1] for r in await cur.fetchall()}
        for col, kind in EXTRA.items():
            if col not in have:
                await self.db.execute(f"ALTER TABLE sessions ADD COLUMN {col} {kind}")
        await self.db.commit()

    async def close(self) -> None:
        if self.db:
            await self.db.close()
            self.db = None

    async def create(self, title: str = "", model: str | None = None, session_id: str | None = None,
                     source: str | None = None) -> dict:
        sid = session_id or uuid.uuid4().hex[:12]
        now = time.time()
        await self.db.execute("INSERT OR IGNORE INTO sessions (id, title, model, created, updated, source) "
                              "VALUES (?,?,?,?,?,?)",
                              (sid, title or time.strftime("Диалог %d.%m %H:%M"), model, now, now, source))
        await self.db.commit()
        return await self.get(sid)

    async def get(self, sid: str) -> dict | None:
        async with self.db.execute(f"SELECT {','.join(COLS)} FROM sessions WHERE id=?", (sid,)) as cur:
            row = await cur.fetchone()
        return dict(zip(COLS, row)) if row else None

    async def touch(self, sid: str, model: str | None = None, title: str | None = None) -> None:
        await self.db.execute("UPDATE sessions SET updated=?, model=COALESCE(?,model), title=COALESCE(?,title) "
                              "WHERE id=?", (time.time(), model, title, sid))
        await self.db.commit()

    async def set_compact(self, sid: str, message_id: str | None) -> None:
        await self.db.execute("UPDATE sessions SET compact_id=? WHERE id=?", (message_id, sid))
        await self.db.commit()

    async def list(self, limit: int = 50, source: str | None = None) -> list[dict]:
        where, args = ("WHERE source=? ", (source, limit)) if source else ("", (limit,))
        async with self.db.execute(f"SELECT {','.join(COLS)} FROM sessions {where}ORDER BY updated DESC LIMIT ?",
                                   args) as cur:
            rows = await cur.fetchall()
        return [dict(zip(COLS, r)) for r in rows]

    async def delete(self, sid: str) -> None:
        await self.db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await self.db.commit()
