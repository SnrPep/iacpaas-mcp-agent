"""Очередь к модели между процессами: консоль и веб-чат на одном ключе GigaChat (один запрос за раз)."""
import asyncio
from types import SimpleNamespace

import pytest

from conftest import make_server
from iacpaas_agent.core import MODEL_429, MODEL_BUSY, AgentRuntime, ModelGate
from iacpaas_agent.locks import SharedLock
from test_agent import settings, write_models

pytestmark = pytest.mark.anyio


def test_shared_lock_is_fair(tmp_path):
    """Кто ждёт, тот следующий: освободившееся окно не забирает модель снова, пока ожидающее её не возьмёт."""
    a, b = SharedLock(tmp_path / "m.lock"), SharedLock(tmp_path / "m.lock")
    assert a.try_acquire() and not b.try_acquire(waiting=True)      # b встал в очередь
    a.release()
    assert not a.try_acquire()                                       # очередь b – a не перехватывает
    assert b.try_acquire(waiting=True) and not (tmp_path / "m.wait").exists()
    assert not a.try_acquire(waiting=True)
    b.release()
    assert a.try_acquire(waiting=True)
    a.release()


async def test_two_windows_share_one_model(tmp_path):
    """Два процесса (здесь – два AgentRuntime с общей папкой data/): шаги модели идут по очереди, второе окно
    видит «модель занята другим окном», оба ответа готовы, ошибки 429 нет."""
    server, _ = make_server()
    models = write_models(tmp_path, {
        "g": {"provider": "fake", "model": "s", "cycle": True, "delay": 0.3, "parallel": 1, "parallel_group": "gc",
              "script": [{"say": "ок"}]}})
    second = settings(tmp_path, models)                 # своя история у окна, общая папка data/ – общий файл очереди
    second.db_file, second.sessions_file = tmp_path / "console.sqlite", tmp_path / "console_sessions.sqlite"
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as web, \
            AgentRuntime(second, mcp_target=server) as console:
        async def run(rt, sid):
            return [e async for e in rt.stream(sid, text="вопрос", model="g")]
        a, b = await asyncio.gather(run(web, "w1"), run(console, "c1"))
    waits = [e for e in a + b if e["event"] == "model_wait"]
    assert len(waits) == 1 and "другим окном" in waits[0]["text"]
    assert a[-1]["status"] == b[-1]["status"] == "done"
    assert (tmp_path / "model-gc.lock").exists()


async def test_gate_busy_and_429_give_clear_answer(tmp_path):
    """Другое окно держит модель дольше срока – понятный ответ без 429; 429 от провайдера – сразу понятный ответ,
    без минут повторов; блокировка после этого свободна."""
    rt = SimpleNamespace(_emit={})
    other = SharedLock(tmp_path / "m.lock")
    assert other.try_acquire()
    gate = ModelGate(rt, asyncio.Semaphore(1), SharedLock(tmp_path / "m.lock"), wait_s=0.3)
    calls = []

    async def handler(request):
        calls.append(1)
        return "ответ"
    busy = await gate.awrap_model_call(None, handler)
    assert busy.content == MODEL_BUSY and not calls                  # к модели не ходили
    other.release()

    class RateLimitError(Exception):
        status_code = 429

    async def limited(request):
        raise RateLimitError("429 Too Many Requests")
    assert (await gate.awrap_model_call(None, limited)).content == MODEL_429
    assert await gate.awrap_model_call(None, handler) == "ответ"     # блокировка освобождена
    assert other.try_acquire()
    other.release()
