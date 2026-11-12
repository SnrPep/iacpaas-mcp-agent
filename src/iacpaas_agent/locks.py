"""Общая очередь к модели между процессами. Консоль и веб-чат – разные процессы с одним ключом GigaChat, а ключ
принимает один запрос за раз (второй – 429). Семафор в памяти их не связывает, поэтому – файловая блокировка:
ОС снимает её сама, если процесс упал или его закрыли.

Очередь честная: кто не смог взять блокировку, записывает себя в файл ожидания (.wait), и освободившийся процесс
не забирает модель снова, пока первый в очереди её не возьмёт. Иначе окно, у которого шаги идут подряд (модель →
быстрый инструмент → модель), держало бы модель почти без перерыва."""
from __future__ import annotations

import os
import time
from pathlib import Path

if os.name == "nt":
    import msvcrt

    def _lock(f) -> None:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(f) -> None:
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(f) -> None:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(f) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)

FRESH_S = 2.0          # ожидающий обновляет файл .wait при каждой попытке; старше – ожидающий ушёл


class SharedLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.wait_file = self.path.with_suffix(".wait")
        self.me = f"{os.getpid()}-{id(self)}"
        self._f = None

    def _first_in_line(self) -> str | None:
        try:
            if time.time() - self.wait_file.stat().st_mtime > FRESH_S:
                return None
            return self.wait_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def try_acquire(self, waiting: bool = False) -> bool:
        """Взять блокировку без ожидания. waiting=True – уже ждём: встать в очередь (или обновить своё место)."""
        if self._f is not None:
            return True
        first = self._first_in_line()
        if first and first != self.me:
            return False                                  # другое окно ждёт дольше – его очередь
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")
        try:
            f.seek(0)
            _lock(f)
        except OSError:
            f.close()
            if waiting:
                try:
                    self.wait_file.write_text(self.me, encoding="utf-8")
                except OSError:
                    pass
            return False
        self._f = f
        if first == self.me:
            try:
                self.wait_file.unlink()
            except OSError:
                pass
        return True

    def release(self) -> None:
        f, self._f = self._f, None
        if f is None:
            return
        try:
            f.seek(0)
            _unlock(f)
        except OSError:
            pass
        finally:
            f.close()
