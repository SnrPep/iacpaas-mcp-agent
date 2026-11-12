"""Standalone-агент целиком: сценарная модель -> LangChain -> MCP (сервер в процессе) -> песочница IACPaaS."""
import json

import pytest
from mcp import Client

from conftest import IR, make_server
from iacpaas_agent.core import AgentRuntime
from iacpaas_agent.settings import AgentSettings

pytestmark = pytest.mark.anyio

UPDATE = {"path": IR, "concept_path": "Основные жалобы/Светобоязнь", "metaconcept_path": "Группа/Наблюдение",
          "mark": "Фотофобия"}


def write_models(tmp_path, profiles: dict) -> object:
    lines = []
    for name, p in profiles.items():
        lines.append(f"[{name}]")
        for k, v in p.items():
            lines.append(f"{k} = {json.dumps(v, ensure_ascii=False)}" if k != "script" else
                         "script = [" + ", ".join(_toml_step(s) for s in v) + "]")
    f = tmp_path / "models.toml"
    f.write_text("\n".join(lines), encoding="utf-8")
    return f


def _toml(v) -> str:
    """Значение TOML: словарь – встроенная таблица, список – массив (вложенные – тоже), остальное – как JSON."""
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml(x)}" for k, x in v.items()) + " }"
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)


def _toml_step(step: dict) -> str:
    if "tool" in step:
        args = ", ".join(f"{k} = {_toml(v)}" for k, v in step.get("args", {}).items())
        say = f', say = {json.dumps(step["say"], ensure_ascii=False)}' if "say" in step else ""
        return f'{{ tool = "{step["tool"]}", args = {{ {args} }}{say} }}'
    tin = f', tokens_in = {step["tokens_in"]}' if "tokens_in" in step else ""
    return f'{{ say = {json.dumps(step["say"], ensure_ascii=False)}{tin} }}'


def settings(tmp_path, models_file, **kw) -> AgentSettings:
    return AgentSettings(model=kw.pop("model", "a"), models_file=models_file,
                         prompt_file=tmp_path / "none.md", db_file=tmp_path / "agent.sqlite",
                         sessions_file=tmp_path / "sessions.sqlite", trace_file=tmp_path / "trace.jsonl",
                         **kw)


async def test_follow_up_gets_code_list_of_previous_questions(tmp_path, monkeypatch):
    """Уточнение («а из них какие жидкие?») – модель получает справку кодом: прошлые вопросы, как код их понял и
    какие газы вышли, даже давние (ушедшие в выжимку); обычный вопрос – без справки."""
    from iacpaas_agent import testing
    seen = []
    orig = testing.ScriptedModel._agenerate

    async def spy(self, messages, *a, **kw):
        seen.append(list(messages))
        return await orig(self, messages, *a, **kw)
    monkeypatch.setattr(testing.ScriptedModel, "_agenerate", spy)
    server, _ = make_server()
    script = [{"tool": "gas_query", "args": {"op": "list", "gases": "аргон"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "gases": "гелий", "by": "чистота"}}, {"say": "{{@gas_query}}"},
              {"say": "Уточню."}]
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": script}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("f1", "Какие аргоны есть в базе?")
        assert "Справка агента" not in seen[-1][-1].content
        await rt.ask("f1", "Какой гелий самый чистый?")
        await rt.ask("f1", "а из них какие жидкие?")
    note = seen[-1][-1].content
    assert "Справка агента: это уточнение к прошлому" in note
    assert "1) «Какие аргоны есть в базе?» → понял так: список: газы – «Газы на основе Аргона»" in note
    assert "Аргон жидкий высший сорт" in note and "2) «Какой гелий самый чистый?»" in note


async def test_same_follow_up_twice_goes_to_its_own_question(tmp_path, monkeypatch):
    """Вопрос 1 – уточнение «сколько там связей»; вопрос 2 – то же уточнение символ в символ: кэша нет, модель
    спрошена заново, а в справке последним стоит вопрос 2 с его вершиной (вопросы о платформе – тоже в справке)."""
    from iacpaas_agent import testing
    seen = []
    orig = testing.ScriptedModel._agenerate

    async def spy(self, messages, *a, **kw):
        seen.append(list(messages))
        return await orig(self, messages, *a, **kw)
    monkeypatch.setattr(testing.ScriptedModel, "_agenerate", spy)
    server, _ = make_server()
    tree = lambda root: {"tool": "get_subgraph", "args": {"path": IR, "root": root, "as_tree": True}}
    count = lambda root: {"tool": "count_graph", "args": {"path": IR, "root": root}}
    script = [tree("Основные жалобы"), {"say": "{{@get_subgraph}}"}, count("Основные жалобы"),
              {"say": "{{@count_graph}}"}, tree("Дополнительные жалобы"), {"say": "{{@get_subgraph}}"},
              count("Дополнительные жалобы"), {"say": "{{@count_graph}}"}]
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": script}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("t1", "покажи основные жалобы деревом")
        first = await rt.ask("t1", "сколько там связей?")
        await rt.ask("t1", "покажи дополнительные жалобы деревом")
        n = len(seen)
        second = await rt.ask("t1", "сколько там связей?")
    assert len(seen) > n                                                     # модель спрошена заново
    note = seen[n][-1].content
    last = note.split("Справка агента")[1].split("Определи")[0].strip().splitlines()[-1]
    assert "«покажи дополнительные жалобы деревом»" in last and "root=«Дополнительные жалобы»" in last
    assert first.answer != second.answer


def snapshot_server():
    """Сервер на снимке настоящей базы газов (tests/snapshot)."""
    from pathlib import Path
    snap = Path(__file__).resolve().parent / "snapshot"
    return make_server(fixtures_dir=snap, registry_file=snap / "none.json")[0]


async def ask_all(tmp_path, script: list, questions: list[str]) -> list:
    tmp_path.mkdir(exist_ok=True)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": script}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=snapshot_server()) as rt:
        return [await rt.ask("fw", q) for q in questions]


OXY = "Кислород технический 1 сорт"
H2_B = "Водород газообразный особо чистый марки В"


async def test_follow_up_is_answered_among_gases_of_its_question(tmp_path):
    """«Какие газы на основе кислорода» → «а какой из них хуже всего по водяным парам?»: модель ищет по всей базе
    (водород В) и не исправляет по замечанию – запрос в рамках кислородов выполняет код. Следующее «а из них какие
    жидкие?» идёт от исправленного запроса (кислороды), а не от запроса модели."""
    wrong = {"tool": "gas_query", "args": {"op": "max", "by": "водяные пары"}}
    script = [{"tool": "gas_query", "args": {"op": "list", "gases": "кислород"}}, {"say": "{{@gas_query}}"},
              wrong, {"say": "{{@gas_query}}"}, wrong, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "list", "state": "жидкий"}}, {"say": "{{@gas_query}}"},
              {"say": "{{@gas_query}}"}]
    _, worst, liquid = await ask_all(tmp_path, script, ["Какие газы на основе кислорода?",
                                                        "а какой из них хуже всего по водяным парам?",
                                                        "а из них какие жидкие?"])
    assert OXY in worst.answer and H2_B not in worst.answer
    assert worst.verify["fixed"] and worst.verify["retried"] == 1
    assert "Жидкий кислород" in liquid.answer and "Аргон жидкий" not in liquid.answer


async def test_back_to_earlier_question_goes_to_it(tmp_path):
    """«вернёмся к аргонам без метана – какой из них самый чистый?» через вопрос о гелиях. Модель берёт по замечанию
    отбор того вопроса – два аргона высшего сорта, не гелий 7.0; модель повторяет ошибку – запрос исправляет код,
    причём с «без метана» (только по слову «аргонам» вышел бы аргон 6.0)."""
    back = ["Какие аргоны без метана?", "Какие гелии есть в базе?",
            "вернёмся к аргонам без метана – какой из них самый чистый?"]
    head = [{"tool": "gas_query", "args": {"op": "list", "gases": "аргон", "without": "метан"}},
            {"say": "{{@gas_query}}"}, {"tool": "gas_query", "args": {"op": "list", "gases": "гелий"}},
            {"say": "{{@gas_query}}"}, {"tool": "gas_query", "args": {"op": "max", "by": "чистота"}},
            {"say": "{{@gas_query}}"}]
    right = [{"tool": "gas_query", "args": {"op": "max", "by": "чистота", "gases": "аргон", "without": "метан"}},
             {"say": "{{@gas_query}}"}]
    *_, self_fixed = await ask_all(tmp_path / "a", head + right, back)
    *_, code_fixed = await ask_all(tmp_path / "b", head + [{"say": "{{@gas_query}}"}], back)
    for r in (self_fixed, code_fixed):
        assert "Аргон газообразный высший сорт" in r.answer and "Аргон жидкий высший сорт" in r.answer
        assert "7.0" not in r.answer and "6.0" not in r.answer and r.verify["retried"] == 1
    assert not self_fixed.verify.get("fixed") and code_fixed.verify["fixed"]


async def test_back_to_question_with_names_of_its_gases_is_right(tmp_path):
    """«вернёмся к аргонам без метана – самый чистый»: модель перечислила в gases три аргона того вопроса (без
    without) – это верно, замечания «в вопросе „без …“» нет: ссылка на прошлый вопрос – не новое условие. А «вернёмся к
    кислородам», когда вопроса о кислородах не было, – отбор из самого вопроса: водороды не пройдут."""
    names = "Аргон газообразный высший сорт; Аргон газообразный первый сорт; Аргон жидкий высший сорт"
    script = [{"tool": "gas_query", "args": {"op": "list", "gases": "аргон", "without": "метан"}},
              {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "by": "чистота", "gases": names}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "list", "gases": "водород"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "by": "чистота", "gases": "водород"}},
              {"say": "{{@gas_query}}"}, {"say": "{{@gas_query}}"}]
    _, best, _, oxy = await ask_all(tmp_path, script, [
        "Какие аргоны без метана?", "вернёмся к аргонам без метана – какой из них самый чистый?",
        "Какие водороды есть в базе?", "вернёмся к кислородам – какой из них самый чистый?"])
    assert "Аргон жидкий высший сорт" in best.answer and best.verify["retried"] == 0 and not best.verify["unverified"]
    assert "Кислород газообразный особой чистоты (марка 5.0)" in oxy.answer and "Водород" not in oxy.answer
    assert oxy.verify["fixed"]


async def test_follow_fix_works_when_model_puts_class_into_path(tmp_path):
    """Модель на уточнение дала класс в path без «без метана» – аргон 6.0 (с метаном) и упорствует. Исправленный
    кодом запрос не должен тащить этот path (сервер не примет path-класс вместе с gases)."""
    by_path = {"tool": "gas_query", "args": {"op": "max", "by": "чистота", "unit": "%",
                                             "path": "Моногазы/Газы на основе Аргона"}}
    script = [{"tool": "gas_query", "args": {"op": "list", "gases": "аргон", "without": "метан"}},
              {"say": "{{@gas_query}}"}, {"tool": "gas_query", "args": {"op": "count"}}, {"say": "{{@gas_query}}"},
              by_path, {"say": "{{@gas_query}}"}, {"say": "{{@gas_query}}"}]
    *_, r = await ask_all(tmp_path, script, ["Какие аргоны без метана?", "Сколько всего газов в базе?",
                                              "вернёмся к аргонам без метана – какой из них самый чистый?"])
    assert "Аргон жидкий высший сорт" in r.answer and "6.0" not in r.answer and r.verify["fixed"]


async def test_right_query_with_own_text_is_not_retried(tmp_path):
    """Живой случай: на «вернёмся к аргонам без метана – самый чистый» модель сразу дала верный запрос, но
    переписала в тексте газы и числа. Раньше – замечание, повтор и неверный новый запрос (аргон 6.0); теперь текст
    модели не показывается, ответ – кусок верного запроса, без повтора."""
    names = "Аргон газообразный высший сорт; Аргон газообразный первый сорт; Аргон жидкий высший сорт"
    script = [{"tool": "gas_query", "args": {"op": "list", "gases": "аргон", "without": "метан"}},
              {"say": "{{@gas_query}}"}, {"tool": "gas_query", "args": {"op": "count"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "by": "чистота", "gases": names}},
              {"say": "{{@gas_query}}\n\nСамый чистый – «Аргон жидкий высший сорт», не менее 99.993%."}]
    *_, r = await ask_all(tmp_path, script, ["Какие аргоны без метана?", "Сколько всего газов в базе?",
                                              "вернёмся к аргонам без метана – какой из них самый чистый?"])
    assert "Аргон газообразный высший сорт" in r.answer and "Аргон жидкий высший сорт" in r.answer
    assert "6.0" not in r.answer and "Самый чистый –" not in r.answer and r.verify["retried"] == 0


async def test_follow_up_to_whole_base_question_is_not_narrowed(tmp_path):
    """Прошлый вопрос – по всей базе («Какие газы есть и сколько их»): уточнение «какой из них самый чистый» – по всей
    базе, замечаний нет; вопрос без «из них» уточнением не считается."""
    script = [{"tool": "gas_query", "args": {"op": "list"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "by": "чистота"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "list", "gases": "кислород"}}, {"say": "{{@gas_query}}"},
              {"tool": "gas_query", "args": {"op": "max", "by": "чистота"}}, {"say": "{{@gas_query}}"}]
    _, best, _, plain = await ask_all(tmp_path, script, [
        "Какие газы есть и сколько их?", "а какой из них самый чистый?", "Какие газы на основе кислорода?",
        "Какой газ самый чистый?"])
    for r in (best, plain):
        assert "7.0" in r.answer and r.verify["retried"] == 0 and not r.verify["unverified"]


async def test_step_limit_shows_pieces_collected_by_code(tmp_path):
    """Модель не уложилась в предел шагов – вместо «не удалось ответить» человек видит куски, которые код уже
    собрал, с пометкой; в истории – то же."""
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [
        {"tool": "count_graph", "args": {"path": IR}}]}})
    async with AgentRuntime(settings(tmp_path, models, max_model_calls=3), mcp_target=server) as rt:
        r = await rt.ask("lim", "Сколько вершин в базе наблюдений?")
        assert r.answer.startswith("⚠ **Модель не уложилась в предел шагов**") and "вершин 11, дуг 11" in r.answer
        assert r.verify["source"] == "code" and r.verify["limit"]
        hist = await rt.history("lim")
    assert "вершин 11, дуг 11" in hist[-1]["text"]


async def test_read_scenario_trace_and_persistent_context(tmp_path):
    server, _ = make_server()
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "script": [
            {"tool": "list_inforesources", "args": {"filter": "наблюд"}},
            {"tool": "count_graph", "args": {"path": IR}},
            {"say": "{{@count_graph}}"}]},
        "b": {"provider": "fake", "model": "s", "script": [{"say": "ответ модели b"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("s1", "Сколько вершин в базе наблюдений?")
        assert r.status == "done" and "вершин 11, дуг 11" in r.answer
        assert [e["tool"] for e in r.events if e["event"] == "tool_call"] == ["list_inforesources", "count_graph"]
        assert all(e["status"] == "success" for e in r.events if e["event"] == "tool_result")
        # переключение модели внутри той же сессии: контекст общий
        r2 = await rt.ask("s1", "ещё вопрос", model="b")
        assert r2.answer == "ответ модели b" and r2.model == "b"
    trace = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {t["event"] for t in trace} == {"model_start", "model_end", "tool_call", "tool_result"}
    assert all(t["session"] == "s1" for t in trace)
    # модель выбрала инструмент – видно в model_end (веб-чат показывает «модель выбрала: …»)
    assert [t["tool_calls"] for t in trace if t["event"] == "model_end"][:2] == [["list_inforesources"], ["count_graph"]]

    # новый процесс (новый runtime) – история на месте
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        hist = await rt.history("s1")
        roles = [h["role"] for h in hist]
        assert roles.count("user") == 2 and "tool_call" in roles and hist[-1]["text"] == "ответ модели b"
        sessions = await rt.sessions.list()
        assert sessions[0]["id"] == "s1" and sessions[0]["title"].startswith("Сколько вершин")


async def test_write_requires_human_and_survives_restart(tmp_path):
    """Запись кнопками: вызов без токена сразу даёт карточку с предпросмотром; «Разрешить» – агент сам
    выполняет тот же вызов с токеном сервера. Ожидание переживает перезапуск."""
    server, backend = make_server(allow_write=True)
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "script": [{"tool": "update_concept", "args": UPDATE}, {"say": "Готово"}]},
        # после перезапуска продолжаем другой моделью (сценарная модель не помнит, где остановилась)
        "b": {"provider": "fake", "model": "s", "script": [{"say": "Готово"}]}})
    node = lambda: backend.store[IR]["successors"][0]["successors"][1]["name"]   # noqa: E731

    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("w1", "Переименуй Светобоязнь в Фотофобия")
        assert r.status == "needs_approval"
        assert r.approvals[0].tool == "update_concept" and r.approvals[0].args["станет"] == {"label": "Фотофобия"}
        assert node() == "Светобоязнь"                             # предпросмотр ничего не изменил
        blocked = await rt.ask("w1", "другой вопрос")
        assert blocked.status == "error" and "подтверждения" in blocked.error

    # «перезапуск» посреди ожидания подтверждения: ожидание сохранено в SQLite
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        pend = await rt.pending("w1", model="b")
        assert len(pend) == 1 and pend[0].tool == "update_concept"
        done = await rt.decide("w1", True, model="b")
        assert done.status == "done" and done.answer == "Готово"
    assert node() == "Фотофобия"


async def test_reject_button_keeps_data(tmp_path):
    server, backend = make_server(allow_write=True)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "update_concept", "args": UPDATE}, {"say": "Изменение отменено"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("rb", "переименуй")).status == "needs_approval"
        r = await rt.decide("rb", False)
        assert r.status == "done" and r.answer == "Изменение отменено"
        res = next(e for e in r.events if e["event"] == "tool_result")
        assert res["preview"].startswith("[rejected]")
    assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"


async def test_reject_keeps_data(tmp_path):
    server, backend = make_server(allow_write=True)
    async with Client(server) as c:
        token = (await c.call_tool("update_concept", UPDATE)).structured_content["confirm_token"]
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "update_concept", "args": {**UPDATE, "confirm_token": token}},
        {"say": "Изменение не выполнено"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("r1", "переименуй")).status == "needs_approval"
        r = await rt.decide("r1", False)
        assert r.status == "done" and r.answer == "Изменение не выполнено"
        assert not any(e["event"] == "tool_call" for e in r.events)     # инструмент не вызывался
    assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"


async def test_limits_stop_runaway_agent(tmp_path):
    server, _ = make_server()
    from iacpaas_agent import testing
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "count_graph", "args": {"path": IR}}]}})
    orig = testing.ScriptedModel.__init__

    def looping(self, script, cycle=False, **kw):                     # модель «зациклилась»
        orig(self, script, cycle=True, **kw)
    testing.ScriptedModel.__init__ = looping
    try:
        async with AgentRuntime(settings(tmp_path, models, max_tool_calls=3, max_model_calls=6),
                                mcp_target=server) as rt:
            r = await rt.ask("l1", "посчитай")
            calls = [e for e in r.events if e["event"] == "tool_call"]
            assert r.status == "done" and len(calls) <= 3
    finally:
        testing.ScriptedModel.__init__ = orig


async def test_missing_model_key_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    server, _ = make_server()
    models = write_models(tmp_path, {"g": {"provider": "gigachat", "model": "GigaChat-2",
                                           "credentials": "${GIGACHAT_CREDENTIALS}"}})
    async with AgentRuntime(settings(tmp_path, models, model="g"), mcp_target=server) as rt:
        r = await rt.ask("e1", "привет")
        assert r.status == "error" and "GIGACHAT_CREDENTIALS" in r.error


async def test_code_request_gets_hidden_note(tmp_path):
    """Просят код – к сообщению для модели добавлено напоминание «результаты не выдумывай»; в истории его нет."""
    from iacpaas_agent.core import CODE_NOTE
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [{"say": "код"}, {"say": "ок"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("s1", "Напиши код на Python и покажи результат таблицей")
        await rt.ask("s1", "Сколько вершин?")
        saved = await rt.checkpointer.aget_tuple(rt._config("s1"))
        sent = [m.content for m in saved.checkpoint["channel_values"]["messages"] if m.type == "human"]
        assert sent[0].endswith(CODE_NOTE) and not sent[1].endswith(CODE_NOTE)
        users = [h["text"] for h in await rt.history("s1") if h["role"] == "user"]
        assert users == ["Напиши код на Python и покажи результат таблицей", "Сколько вершин?"]


async def test_model_sees_old_tool_results_as_stubs_history_stays_full(tmp_path, monkeypatch):
    """Экономия токенов: результаты более ранних вопросов уходят модели заглушкой, текущего и предыдущего –
    целиком («сохрани эти газы» – по данным прошлого ответа); в БД и в истории чата – полный текст."""
    from iacpaas_agent import testing
    seen = []
    orig = testing.ScriptedModel._agenerate

    async def spy(self, messages, *a, **kw):
        seen.append(list(messages))
        return await orig(self, messages, *a, **kw)
    monkeypatch.setattr(testing.ScriptedModel, "_agenerate", spy)
    server, _ = make_server()
    step = {"tool": "summarize_inforesource", "args": {"path": IR}}
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        step, {"say": "{{@summarize_inforesource}}"}, step, {"say": "{{@summarize_inforesource}}"}, step, {"say": "{{@summarize_inforesource}}"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("c1", "обзор")
        await rt.ask("c1", "ещё раз обзор")
        await rt.ask("c1", "и ещё раз")
        hist = await rt.history("c1")
    from iacpaas_agent.core import _text
    tools = [[_text(m.content) for m in req if m.type == "tool"] for req in seen]
    assert len(tools[1][0]) > 300                               # первый вопрос: результат целиком
    assert all(len(t) > 300 for t in tools[3])                  # второй вопрос: и прошлый результат целиком
    last = tools[-1]                                            # третий вопрос, после вызова
    assert last[0].startswith("[Результат «summarize_inforesource» из более раннего вопроса")
    assert len(last[1]) > 300 and len(last[2]) > 300
    full = [h["text"] for h in hist if h["role"] == "tool_result"]
    assert len(full) == 3 and all(len(t) > 300 for t in full)   # история – полная


def test_compact_keeps_recent_questions_and_lists_dropped():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from iacpaas_agent.core import compact_messages

    def turn(i):
        return [HumanMessage(f"вопрос {i}"),
                AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": f"c{i}", "type": "tool_call"}]),
                ToolMessage("x" * 500, tool_call_id=f"c{i}", name="t"), AIMessage(f"ответ {i}")]
    msgs = turn(1) + turn(2) + turn(3) + [HumanMessage("вопрос 4")]
    out = compact_messages(msgs, keep=3)               # в окне считаются только вопросы и ответы (по 2 на ход)
    assert [m.content for m in out if m.type == "ai" and m.content] == ["ответ 3"]       # вопрос 3 – всегда
    assert "1) «вопрос 1» → ответ: ответ 1 2) «вопрос 2» → ответ: ответ 2" in out[0].content
    assert out[0].content.endswith("вопрос 3")
    assert "не другой сеанс" in out[0].content
    assert out[-1].content == "вопрос 4" and all(len(m.content) < 600 for m in out)
    assert len(compact_messages(msgs, keep=6)) == len(msgs)                            # 6 сообщений чата – влезли все
    # тяжёлый вопрос с 10 вызовами инструментов не выталкивает из окна остальные (раньше хватало двух вопросов)
    heavy = [HumanMessage("первый вопрос")]
    for i in range(10):
        heavy += [AIMessage("", tool_calls=[{"name": "t", "args": {"i": i}, "id": f"h{i}", "type": "tool_call"}]),
                  ToolMessage("y" * 500, tool_call_id=f"h{i}", name="t")]
    heavy += [AIMessage("ответ на первый")] + turn(2) + turn(3) + [HumanMessage("какой был мой первый вопрос?")]
    seen = compact_messages(heavy, keep=15)
    assert seen[0].content == "первый вопрос" and any(m.content == "ответ на первый" for m in seen)
    # длинный диалог: номера ранних вопросов не сбиваются, первые не теряются
    many = [m for i in range(1, 21) for m in turn(i)] + [HumanMessage("последний")]
    note = compact_messages(many, keep=4)[0].content
    assert "1) «вопрос 1» → ответ: ответ 1" in note and "18) «вопрос 18» → ответ: ответ 18" in note
    assert msgs[2].content == "x" * 500                                                 # исходные не тронуты
    assert len(compact_messages(msgs, keep=0)) == len(msgs)                             # 0 – без окна


async def test_same_successful_call_is_not_repeated(tmp_path):
    """Тот же удачный вызов в одном ответе второй раз не выполняется – модель получает короткое [repeat]."""
    server, _ = make_server()
    step = {"tool": "summarize_inforesource", "args": {"path": IR}}
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [step, step, {"say": "ок"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("r1", "обзор")
    res = [e for e in r.events if e["event"] == "tool_result"]
    assert res[0]["status"] == res[1]["status"] == "success" and res[1]["preview"].startswith("[repeat]")


async def test_write_without_request_is_refused(tmp_path):
    """Изменение без просьбы пользователя не выполняется и кнопок не показывает; «да» на предложение – можно."""
    from langchain_core.messages import AIMessage, HumanMessage

    from iacpaas_agent.core import write_asked
    server, backend = make_server(allow_write=True)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "update_concept", "args": UPDATE}, {"say": "Ответ на вопрос"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("n1", "Сколько вершин в базе?")
    assert r.status == "done" and not r.approvals
    assert next(e for e in r.events if e["event"] == "tool_result")["preview"].startswith("[not_requested]")
    assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"
    offer = AIMessage("Могу сохранить подборку в вашу папку.")
    assert write_asked([HumanMessage("Какие газы чище?"), offer, HumanMessage("да, давай")])
    assert not write_asked([HumanMessage("Какие газы чище?"), AIMessage("17 газов"), HumanMessage("да")])
    assert write_asked([HumanMessage("Сохрани подборку")])


async def test_interim_without_steer_and_memory_length(tmp_path, monkeypatch):
    """Текст модели рядом с вызовом без дописанного сообщения – часть работы (без пометки steer, страница кладёт
    его внутрь шага). Длина памяти с ползунка доходит до прослойки и ограничена 10–100."""
    from iacpaas_agent import core
    used = []
    orig = core.compact_messages
    monkeypatch.setattr(core, "compact_messages", lambda msgs, keep, *a: used.append(keep) or orig(msgs, keep, *a))
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [
        {"tool": "count_graph", "args": {"path": IR}, "say": "Сначала посчитаю вершины."}, {"say": "{{@count_graph}}"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("m1", "сколько?", keep=3)
        await rt.ask("m1", "ещё", keep=500)
        await rt.ask("m1", "и ещё")
        hist = await rt.history("m1")
    interim = [h for h in hist if h["role"] == "interim"]
    assert interim and not any(h.get("steer") for h in interim)
    assert used == [10, 10, 100, 100, 15, 15]


async def test_false_save_claim_gets_warning(tmp_path):
    """Запись вернула ошибку (выдуманные пути), а модель пишет «подборка сохранена» – агент дописывает
    предупреждение; честный ответ («не удалось») не трогается."""
    server, _ = make_server(allow_write=True)
    bad = {"tool": "save_selection", "args": {"source_path": IR, "title": "Проба", "elements": ["Нет такой вершины"]}}
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "script": [bad, {"say": "Подборка сохранена в вашей папке."}]},
        "b": {"provider": "fake", "model": "s", "script": [bad, {"say": "Сохранить не удалось: таких газов нет."}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("f1", "Сохрани подборку в мою папку")
        assert r.answer.startswith("Подборка сохранена") and "изменение не выполнено" in r.answer
        assert "изменение не выполнено" in [h for h in await rt.history("f1") if h["role"] == "assistant"][-1]["text"]
        ok = await rt.ask("f2", "Сохрани подборку в мою папку", model="b")
        assert ok.answer == "Сохранить не удалось: таких газов нет."


async def test_question_about_dialog_gets_list_of_questions(tmp_path):
    """«Какой был мой первый вопрос?» – к вопросу для модели приложен список вопросов диалога по порядку;
    пользователь в истории видит только свой текст. Служебные справки не считаются просьбой о записи."""
    from iacpaas_agent.core import DIALOG_ASK, write_asked
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "cycle": True, "script": [{"say": "ок"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("d1", "Сохрани подборку")
        await rt.ask("d1", "Сколько вершин?")
        await rt.ask("d1", "Какой был мой самый первый вопрос?")
        saved = await rt.checkpointer.aget_tuple(rt._config("d1"))
        msgs = saved.checkpoint["channel_values"]["messages"]
        last = [m for m in msgs if m.type == "human"][-1]
        assert "1) «Сохрани подборку» 2) «Сколько вершин?»" in last.content
        users = [h["text"] for h in await rt.history("d1") if h["role"] == "user"]
        assert users[-1] == "Какой был мой самый первый вопрос?"
        assert not write_asked(msgs)                                  # «Сохрани» внутри справки – не просьба
        # про прошлые ответы – приложены и сами ответы: первый ответ мог уйти из окна памяти (модель выдумала суммы)
        await rt.ask("d1", "Какие суммы примесей были в первом ответе?")
        saved = await rt.checkpointer.aget_tuple(rt._config("d1"))
        last = [m for m in saved.checkpoint["channel_values"]["messages"] if m.type == "human"][-1]
        assert "твои ответы на вопросы этого диалога" in last.content and "на вопрос 1 – «ок»" in last.content
        assert "на вопрос 3 –" not in last.content                     # ответ на вопрос о диалоге не берётся
    assert DIALOG_ASK.search("что я спрашивал раньше?") and not DIALOG_ASK.search("Какой состав у гелия?")
    assert DIALOG_ASK.search("какие суммы были в первом ответе") and DIALOG_ASK.search("что ты отвечал про аргон")


@pytest.mark.anyio
async def test_context_like_claude_code(tmp_path):
    """Контекст следующего запроса – как в Claude Code: инструкция + схемы инструментов + история, доля окна модели.
    Символы на токен: модель не отвечала и в профиле не задано – «примерно»; задано – из профиля; модель ответила
    (сообщила, сколько прочитала) – замер по её ответу; замер вне разумного (сбой) – не берётся. Окна в профиле нет –
    0 (доля не показывается)."""
    server, _ = make_server()
    count = {"tool": "count_graph", "args": {"path": IR}}
    models = write_models(tmp_path, {
        "a": {"provider": "fake", "model": "s", "script": [count, {"say": "В базе 11 вершин."}]},
        "b": {"provider": "fake", "model": "s", "window_tokens": 128000, "chars_per_token": 4,
              "script": [count, {"say": "В базе 11 вершин.", "tokens_in": 10 ** 9}]},
        "c": {"provider": "fake", "model": "s", "window_tokens": 32000, "script": [count, {"say": "В базе 11 вершин."}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("s1", "Сколько вершин?")
        c = await rt.context("s1", 10, "a")
        assert c["cpt_from"] == "approx" and c["window"] == 0 and c["tools"] > 0 and c["system"] > 0 and c["tools_n"] > 0
        await rt.ask("s2", "Сколько вершин?", model="b")
        c = await rt.context("s2", 10, "b")
        # 10⁹ токенов – не замер (сбой или чужие данные): символы на токен – из профиля
        assert c["window"] == 128000 and c["cpt_from"] == "profile" and c["cpt"] == 4 and c["last_in"] == 10 ** 9
        chars = c["system"] * 4 + c["tools"] * 4 + c["tokens"] * 4
        script = [count, {"say": "В базе 11 вершин.", "tokens_in": max(chars // 4, 1)}]
    models = write_models(tmp_path, {"c": {"provider": "fake", "model": "s", "window_tokens": 32000, "script": script}})
    async with AgentRuntime(settings(tmp_path, models, model="c"), mcp_target=server) as rt:
        await rt.ask("s3", "Сколько вершин?")
        c = await rt.context("s3", 10, "c")
        assert c["cpt_from"] == "measured" and 2.5 < c["cpt"] < 6 and c["window"] == 32000


async def test_context_chars_per_token_over_several_steps(tmp_path, monkeypatch):
    """Символы на токен – по нескольким последним шагам модели вместе, а не по одному последнему: иначе инструкция и
    схемы инструментов (они не меняются) «прыгали» после каждого вопроса."""
    from iacpaas_agent import core
    server, _ = make_server()
    count = {"tool": "count_graph", "args": {"path": IR}}
    models = write_models(tmp_path, {"c": {"provider": "fake", "model": "s", "window_tokens": 32000, "script": [
        count, {"say": "В базе 11 вершин.", "tokens_in": 4000}, count, {"say": "Снова 11.", "tokens_in": 4000}]}})
    async with AgentRuntime(settings(tmp_path, models, model="c"), mcp_target=server) as rt:
        await rt.ask("j1", "Сколько вершин?")
        await rt.ask("j1", "А сейчас?")
        msgs = await rt._saved("j1")
        steps = [m for m in msgs if isinstance(m, core.AIMessage) and core.usage(m).get("tokens_in")]
        assert len(steps) == 2
        # последний шаг «прочитал» больше токенов на похожий текст – по нему одному делитель заметно меньше
        steps[-1].usage_metadata = {**(steps[-1].usage_metadata or {}), "input_tokens": 6000}

        async def saved(_sid):
            return msgs
        monkeypatch.setattr(rt, "_saved", saved)
        both = await rt.context("j1", 10, "c")
        steps[0].usage_metadata = {**(steps[0].usage_metadata or {}), "input_tokens": None}
        only_last = await rt.context("j1", 10, "c")
        assert both["cpt_from"] == only_last["cpt_from"] == "measured"
        assert both["steps"] == 2 and only_last["steps"] == 1, (both, only_last)
        assert both["cpt"] > only_last["cpt"], (both["cpt"], only_last["cpt"])
