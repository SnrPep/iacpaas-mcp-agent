"""Ответы о газах: запрос (query.py) сверяется с независимым расчётом по данным для каждого класса и компонента;
агент (на подставной модели) – куски по меткам, замечания проверки и повтор, строгий итог, «не из базы»."""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from conftest import FIXTURES, make_server
from iacpaas_agent.core import STRICT_EMPTY, AgentRuntime
from iacpaas_mcp import knowledge, query
from iacpaas_mcp.answers import AnswerError
from iacpaas_mcp.config import load_synonyms
from test_agent import settings, write_models

IR = json.loads((FIXTURES / "Загрузки" / "База технологических газов.json").read_text(encoding="utf-8"))
SYN = load_synonyms()
BASE = "Загрузки/База технологических газов"


def _classes() -> dict[str, list[dict]]:
    """Класс → газы с составом – прежним кодом (knowledge.element_rows), без query."""
    out: dict[str, list[dict]] = {}
    for r in knowledge.element_rows(IR, "Газ", with_composition=True):
        if r.get("composition"):
            out.setdefault(r["group"], []).append(r)
    return out


def _imp(r: dict, comp: str) -> Decimal | None:
    t = r["composition"].get(comp, "")
    first = next(iter(r["composition"]))
    return Decimal(t.split()[2]) if t.startswith("не более") and t.endswith("%") and comp != first else None


def _names(res: dict) -> list[str]:
    return [x["name"] for x in res["rows"]]


def test_extremes_without_threshold_sort_match_data_for_every_class_and_component():
    """Крайние (все равные), «без X», порог и сортировка – ровно те газы, что дают данные."""
    checked = 0
    for group, rows in _classes().items():
        subst = group.removeprefix("Газы на основе ")
        comps = sorted({c for r in rows for c in r["composition"] if "+" not in c})
        for comp in comps:
            vals = {r["name"]: v for r in rows if (v := _imp(r, comp)) is not None}
            if not vals:
                continue
            for op, f in (("max", max), ("min", min)):
                res = query.run(IR, op=op, gases=subst, by=comp, synonyms=SYN)
                assert set(_names(res)) == {n for n, v in vals.items() if v == f(vals.values())}, (group, comp, op)
            no = {r["name"] for r in rows if comp not in r["composition"]
                  and not any(comp in knowledge._parts_of(c) for c in r["composition"])
                  and next(iter(r["composition"])) != comp}
            assert set(_names(query.run(IR, op="list", gases=subst, without=comp, synonyms=SYN))) == no, (group, comp)
            mid = sorted(vals.values())[len(vals) // 2]
            res = query.run(IR, op="sort", gases=subst, where=f"{comp} <= {mid}", by=comp, order="asc", synonyms=SYN)
            got = _names(res)
            assert set(got) == {n for n, v in vals.items() if v <= mid}, (group, comp)
            assert [vals[n] for n in got] == sorted(vals[n] for n in got)
            checked += 1
    assert checked >= 15


def test_purity_ties_unfilled_errors_and_empty():
    rows = [r for r in knowledge.element_rows(IR, "Газ") if r["main_value"] is not None]
    top = max(r["main_value"] for r in rows)
    res = query.run(IR, op="max", by="чистота", synonyms=SYN)
    assert set(_names(res)) == {r["name"] for r in rows if r["main_value"] == top}
    empty = [r for r in knowledge.element_rows(IR, "Газ") if not r["filled"]]
    assert query.run(IR, op="unfilled", synonyms=SYN)["total"] == len(empty)
    ox = query.run(IR, op="sort", gases="кислород", by="O", synonyms=SYN)["answer"]
    assert "O – основной компонент" not in ox or "не менее" in ox
    none = query.run(IR, op="list", where="N > 99", synonyms=SYN)
    assert none["total"] == 0 and "не подходит ни один" in none["answer"]
    bad = [r["name"] for r in knowledge.element_rows(IR, "Газ") if r.get("impurities_max_sum", "")[:1].isdigit()
           and "rest_to_100" in r and Decimal(r["impurities_max_sum"]) > Decimal(r["rest_to_100"])]
    assert set(_names(query.run(IR, op="errors", synonyms=SYN))) == set(bad)


def test_calc_matches_values_of_two_gases():
    rows = [r for r in _classes().values() if len(r) >= 2 and sum(1 for x in r if _imp(x, "N")) >= 2][0]
    a, b = [r for r in rows if _imp(r, "N")][:2]
    va, vb = _imp(a, "N"), _imp(b, "N")
    res = query.calc(IR, f"[{a['name']} | азот] - [{b['name']} | азот]", synonyms=SYN)
    assert res["result"] == knowledge.dec_text(va - vb)
    res = query.calc(IR, f"[{a['name']} | азот] + [{a['name']} | азот]", synonyms=SYN)
    assert res["result"] == knowledge.dec_text(va + va)


def args(**kw) -> dict:
    """Параметры gas_query так, как их собирает сервер (по ним – метка куска)."""
    base = {"op": "list", "gases": "", "where": "", "has": "", "without": "", "name_has": "", "name_not": "",
            "state": "", "grade": "", "standard": "", "by": "", "order": "desc", "top": 0, "show": "", "unit": "%",
            "expr": "", "path": BASE}
    return {**base, **kw}


def mark(**kw) -> str:
    return query.block_id("gas_query", args(**kw))


def piece(**kw) -> str:
    return query.run(IR, **{k: v for k, v in args(**kw).items() if k not in ("path", "expr")}, synonyms=SYN)["answer"]


async def _ask(tmp_path, script, question, sid="s1", allow_write=False):
    server, _ = make_server(allow_write=allow_write)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": script}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask(sid, question)
        notes = []
        if r.status == "done":
            agent, _, _ = rt.agent_for("a")
            msgs = (await agent.aget_state(rt._config(sid))).values["messages"]
            notes = [m.content for m in msgs if (m.response_metadata or {}).get("iacpaas", {}).get("check")]
        return r, notes


@pytest.mark.anyio
async def test_agent_answer_by_data_is_pieces_only(tmp_path):
    """Ответ по данным – только куски кода: модель пересказала число в тексте – это не показывается и повтора нет
    (раньше – замечание и повтор, на котором модель путала уже верный запрос)."""
    m = mark(op="max", by="чистота")
    r, notes = await _ask(tmp_path, [
        {"tool": "gas_query", "args": {"op": "max", "by": "чистота"}},
        {"say": "Самый чистый – 99.9999 %:\n" + m}], "Какой газ самый чистый?")
    assert r.answer == piece(op="max", by="чистота").strip() and "99.9999 %:" not in r.answer
    assert r.verify["source"] == "code" and r.verify["problems"] == [] and r.verify["retried"] == 0 and not notes
    assert r.verify["understood"] and "самые чистые" in r.verify["understood"][0]


@pytest.mark.anyio
async def test_agent_plan_must_match_question(tmp_path):
    """«без метана» без without – замечание, второй запрос с without – ответ из его куска."""
    r, notes = await _ask(tmp_path, [
        {"tool": "gas_query", "args": {"op": "list", "gases": "аргон"}},
        {"say": "Аргоны:\n" + mark(gases="аргон")},
        {"tool": "gas_query", "args": {"op": "list", "gases": "аргон", "without": "метан"}},
        {"say": "Аргоны без метана:\n" + mark(gases="аргон", without="метан")}], "Какие аргоны без примеси метана?")
    assert any("without" in n for n in notes)
    assert piece(gases="аргон", without="метан") in r.answer and r.verify["unverified"] == []


@pytest.mark.anyio
async def test_agent_base_question_without_query_is_not_shown(tmp_path):
    say = {"say": "В базе аргоны высокой чистоты, их много."}
    r, notes = await _ask(tmp_path, [say, say, say], "Какие аргоны есть в базе?")
    assert any("сначала gas_query" in n for n in notes)
    assert r.answer == STRICT_EMPTY and r.verify["strict"]


@pytest.mark.anyio
async def test_agent_general_question_is_model_answer(tmp_path):
    r, notes = await _ask(tmp_path, [{"say": "Здравствуйте! Спросите о газах базы."}], "привет")
    assert notes == [] and r.verify["source"] == "model" and r.verify.get("skipped")


@pytest.mark.anyio
async def test_write_confirmation_needs_no_mark(tmp_path):
    six = "Моногазы/Газы на основе Аргона/Аргон газообразный высокой чистоты марка 6.0"
    server, _ = make_server(allow_write=True)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "gas_query", "args": {"op": "max", "gases": "аргон", "by": "чистота"}},
        {"tool": "save_selection", "args": {"source_path": BASE, "title": "Самый чистый аргон", "elements": [six]}},
        {"say": "Подборка «Самый чистый аргон» сохранена в «Песочница ИИ»: 1 газ."},
        {"say": mark(op="max", gases="аргон", by="чистота")}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("w1", "Сохрани самый чистый аргон в подборку")).status == "needs_approval"
        r = await rt.decide("w1", True)
        assert r.status == "done" and "сохранена" in r.answer and r.verify["retried"] == 0, (r.answer, r.verify)
        assert r.verify["problems"] == []


@pytest.mark.anyio
async def test_code_fixes_query_when_model_ignores_remark(tmp_path):
    """Модель дважды берёт «хуже всего по кислороду» по всей базе – код сам выполняет запрос среди аргонов."""
    wrong = {"tool": "gas_query", "args": {"op": "max", "by": "кислород"}}
    m = mark(op="max", by="кислород")
    r, notes = await _ask(tmp_path, [wrong, {"say": "Хуже всего: " + m}, wrong, {"say": "Хуже всего: " + m}],
                          "Какой аргон хуже всего по кислороду?")
    assert any("вызови ровно gas_query(op=\"max\", by=\"кислороду\", gases=\"аргон\")" in n for n in notes)
    want = query.run(IR, op="max", gases="аргон", by="кислороду", synonyms=SYN)["answer"]
    assert want in r.answer and r.verify["fixed"] and r.verify["unverified"] == []
    assert "«Газы на основе Аргона»" in r.verify["understood"][0]


@pytest.mark.anyio
async def test_same_question_is_asked_to_model_again(tmp_path):
    """Кэша ответов нет: тот же вопрос второй раз – снова модель и снова запрос к данным (данные свежие)."""
    server, _ = make_server()
    m = mark(op="max", gases="аргон", by="чистота")
    turn = [{"tool": "gas_query", "args": {"op": "max", "gases": "аргон", "by": "чистота"}},
            {"say": "Самый чистый аргон:\n" + m}]
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": turn + turn}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        first = await rt.ask("c1", "Какой аргон самый чистый?")
        again = await rt.ask("c2", "Какой аргон самый чистый?")
    assert first.status == again.status == "done" and again.answer == first.answer
    assert [e["event"] for e in again.events].count("model_end") == 2          # модель спрошена заново
    assert [e["event"] for e in again.events].count("tool_call") == 1          # и данные запрошены заново


@pytest.mark.anyio
async def test_failed_query_and_words_only_answer_gets_code_piece(tmp_path):
    """Живой S17: where «сумма примесей > чистота» – ошибка инструмента, модель «не могу» – код добавляет op=errors."""
    bad = {"tool": "gas_query", "args": {"op": "list", "where": "сумма примесей > чистота"}}
    r, notes = await _ask(tmp_path, [bad, {"say": "Так по базе посчитать не могу."}, bad,
                                     {"say": "Так по базе посчитать не могу."}],
                          "у какого газа примесей больше чем самого газа?")
    assert any("не удался" in n for n in notes)
    assert query.run(IR, op="errors", synonyms=SYN)["answer"] in r.answer and "не могу" not in r.answer
    assert r.verify["fixed"] and not r.verify.get("strict")


@pytest.mark.anyio
async def test_loop_of_same_query_ends_with_code_answer(tmp_path):
    """Живой T27: модель после замечания по кругу повторяла тот же запрос до предела шагов – теперь после «[stop]» ход
    кончается, ответ – из кусков, неверный запрос исправляет код."""
    good = {"tool": "gas_query", "args": {"op": "max", "gases": "кислород", "by": "водяные пары"}}
    wrong = {"tool": "gas_query", "args": {"op": "min", "by": "водяные пары"}}
    say = {"say": mark(op="max", gases="кислород", by="водяные пары") + " " + mark(op="min", by="водяные пары")}
    r, _ = await _ask(tmp_path, [good, wrong, say, *[wrong] * 6, {"say": "конец"}],
                      "У какого кислорода больше всего водяных паров, а у какого меньше всего?")
    assert r.status == "done" and "предел" not in r.answer, r.answer
    want = query.run(IR, op="min", gases="кислород", by="водяные пары", synonyms=SYN)["answer"]
    assert want in r.answer and r.verify["fixed"]


@pytest.mark.anyio
async def test_loop_end_leaves_no_call_without_result(tmp_path):
    """Ход после «[stop]» кончает код: у последнего сообщения модели убирается и function_call GigaChat – иначе в
    истории вызов без результата, и GigaChat отвечает 422 («every assistant function call must have a result») на
    каждый следующий вопрос диалога."""
    from langchain_gigachat.chat_models.gigachat import _convert_message_to_dict
    from langchain_core.messages import HumanMessage
    from iacpaas_agent.core import compact_messages
    wrong = {"tool": "gas_query", "args": {"op": "min", "by": "водяные пары"}}
    server, _ = make_server()
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        wrong, {"say": mark(op="min", by="водяные пары")}, *[wrong] * 6, {"say": "конец"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        r = await rt.ask("lp", "У какого кислорода меньше всего водяных паров?")
        agent, _, _ = rt.agent_for("a")
        msgs = (await agent.aget_state(rt._config("lp"))).values["messages"]
    assert r.status == "done"
    sent = [_convert_message_to_dict(m) for m in compact_messages(msgs + [HumanMessage("а из них какой?")], 10)]
    calls = [i for i, d in enumerate(sent) if d.role == "assistant" and d.function_call]
    assert calls and all(sent[i + 1].role == "function" for i in calls)
    # functions_state_id – только у вызовов с результатом (у отменённого кодом вызова его не остаётся)
    done = {sent[i].functions_state_id for i in calls}
    assert all(d.functions_state_id in done for d in sent if d.role == "assistant" and d.functions_state_id)


@pytest.mark.anyio
async def test_write_turn_drops_copied_mark_and_skips_meaning(tmp_path):
    """Живой T58: смесь создана, модель вставила метку-образец – метка убирается, «создана» остаётся."""
    server, _ = make_server(allow_write=True)
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "save_mixture", "args": {"source_path": BASE, "name": "Аргон-Гелий 80-20", "components": [
            {"gas": "Аргон газообразный первый сорт", "percent": 80}, {"gas": "Гелий 5.0", "percent": 20}]}},
        {"say": "Смесь создана: {{#a1b2}}"}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("m1", "Создай черновик смеси: аргон первого сорта 80 % и гелий 5.0 20 %")).status \
            == "needs_approval"
        r = await rt.decide("m1", True)
        assert r.status == "done" and r.answer.startswith("Смесь создана") and not r.verify["problems"], r.verify


def test_grade_with_class_word_is_rejected_and_fixed():
    from iacpaas_agent.check import corrections
    with pytest.raises(AnswerError, match="класс газов"):
        query.run(IR, op="list", grade="аргон", synonyms=SYN)
    got = corrections("а есть аргон чище чем 6.0?", [("{{#aaaa}}", {"op": "list", "grade": "аргон",
                                                                     "where": "чистота > Аргон 6.0"})])
    assert got and "grade" not in got[0]["args"] and got[0]["args"]["gases"] == "аргон"


@pytest.mark.anyio
async def test_rejected_write_is_said_by_code(tmp_path):
    """Живой T59: пользователь отклонил запись, модель показала только карточку – «отклонено» пишет код."""
    server, _ = make_server(allow_write=True)
    six = "Моногазы/Газы на основе Аргона/Аргон газообразный высокой чистоты марка 6.0"
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "save_selection", "args": {"source_path": BASE, "title": "Аргон 6.0", "elements": [six]}},
        {"say": "Вот газ."}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("w2", "Сохрани аргон 6.0 в мою папку")).status == "needs_approval"
        r = await rt.decide("w2", False)
        assert r.status == "done" and r.answer.startswith("**Изменение отклонено"), r.answer


@pytest.mark.anyio
async def test_write_with_mark_delete_from_base_and_general_numbers(tmp_path):
    """Живые T57, T60, T63: метка куска вместо газов – газы подставляет код, «сохранено» – тоже код; «удали из базы
    газов» – отказ без удаления; «что такое ppm» – числа знаний можно."""
    server, backend = make_server(allow_write=True)
    m = mark(gases="гелий", where="чистота >= 99.999")
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "gas_query", "args": {"op": "list", "gases": "гелий", "where": "чистота >= 99.999"}},
        {"tool": "save_selection", "args": {"source_path": BASE, "title": "Гелии не ниже 99.999", "elements": m}},
        {"say": m}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        assert (await rt.ask("w3", "Сохрани все гелии с чистотой не ниже 99.999 % в подборку")).status == "needs_approval"
        r = await rt.decide("w3", True)
        assert r.status == "done" and r.answer.startswith("**Готово: подборка сохранена"), r.answer
    want = {x["path"] for x in query.run(IR, op="list", gases="гелий", where="чистота >= 99.999", synonyms=SYN)["rows"]}
    saved = [k for k in backend.store if k.endswith("Гелии не ниже 99.999")]
    dump = json.dumps(backend.store[saved[0]], ensure_ascii=False) if saved else ""
    assert want and all(p.split("/")[-1] in dump for p in want)       # в подборке – все газы куска
    r, notes = await _ask(tmp_path, [
        {"tool": "delete_from_fund", "args": {"path": "Загрузки/Песочница ИИ/Аргон 6.0", "mode": "deleteIR"}},
        {"say": "Эту базу менять нельзя: она только для чтения."}], "Удали аргон 6.0 из базы газов", sid="w4",
        allow_write=True)
    assert r.status == "done" and "только для чтения" in r.answer
    r, notes = await _ask(tmp_path, [{"say": "ppm – миллионная доля: 1 ppm = 0.0001 %."}], "Что такое ppm?", sid="w5")
    assert notes == [] and "миллионная" in r.answer


@pytest.mark.anyio
async def test_no_mark_after_retries_shows_turn_pieces(tmp_path):
    """Живой T29: модель трижды ответила текстом без метки – показан кусок запроса этого хода, а не «не собран»."""
    call = {"tool": "gas_query", "args": {"op": "max", "by": "неон"}}
    say = {"say": "Больше всего неона – в гелиях."}
    r, _ = await _ask(tmp_path, [call, say, say, say], "Где больше всего неона в примесях?")
    assert query.run(IR, op="max", by="неон", synonyms=SYN)["answer"] in r.answer and r.verify["strict"]


@pytest.mark.anyio
async def test_bare_calc_piece_is_dropped_and_write_args_show_real_gases(tmp_path):
    """Живые S36 и T57: «19» калькулятором – кусок убирается; метка вместо газов в save_selection – в истории сам
    вызов уже с газами (что на самом деле сохранено)."""
    srt = {"tool": "gas_query", "args": {"op": "sort", "gases": "аргон", "by": "метан", "order": "asc"}}
    bare = {"tool": "gas_query", "args": {"op": "calc", "expr": "3"}}
    say = {"say": mark(op="sort", gases="аргон", by="метан", order="asc") + "\n" + mark(op="calc", expr="3")}
    r, _ = await _ask(tmp_path, [srt, bare, say, say], "отсортируй аргоны по увеличению примесей метана")
    assert "3 = 3" not in r.answer and r.verify["unverified"] == [] and r.verify["fixed"]
    server, _ = make_server(allow_write=True)
    m = mark(gases="гелий", where="чистота >= 99.999")
    models = write_models(tmp_path, {"a": {"provider": "fake", "model": "s", "script": [
        {"tool": "gas_query", "args": {"op": "list", "gases": "гелий", "where": "чистота >= 99.999"}},
        {"tool": "save_selection", "args": {"source_path": BASE, "title": "Гелии 99.999", "elements": m}},
        {"say": "Сохранено."}]}})
    async with AgentRuntime(settings(tmp_path, models), mcp_target=server) as rt:
        await rt.ask("w6", "Сохрани все гелии с чистотой не ниже 99.999 % в подборку")
        agent, _, _ = rt.agent_for("a")
        msgs = (await agent.aget_state(rt._config("w6"))).values["messages"]
        call = next(tc for x in msgs if getattr(x, "tool_calls", None) for tc in x.tool_calls
                    if tc["name"] == "save_selection")
        assert "{{#" not in call["args"]["elements"] and "Гелий" in call["args"]["elements"]
