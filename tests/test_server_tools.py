"""Инструменты MCP-сервера через настоящий MCP-клиент в памяти (без процессов и сети)."""
import pytest
from mcp import Client

from conftest import IR, make_server, make_settings

pytestmark = pytest.mark.anyio

READ_TOOLS = {"iacpaas_status", "list_inforesources", "get_inforesource", "get_subgraph", "count_graph",
              "query_graph", "summarize_inforesource", "get_ontology", "export_json", "validate_resource",
              "service_status", "check_inforesource",
              "map_inforesources", "gas_query"}
WRITE_TOOLS = {"update_concept", "import_inforesource", "delete_from_fund", "save_selection", "save_mixture", "run_service",
               "stop_service"}


async def call(server, name, args=None):
    async with Client(server) as c:
        return await c.call_tool(name, args or {})


async def test_read_only_by_default_no_write_tools():
    server, _ = make_server()
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == READ_TOOLS
    for t in tools.values():
        assert t.annotations.read_only_hint is True
        assert "ctx" not in t.input_schema.get("properties", {})
        # Схемы без anyOf: их принимают все провайдеры моделей (включая GigaChat)
        assert "anyOf" not in str(t.input_schema), t.name


async def test_sandbox_write_tools_and_annotations():
    server, _ = make_server(allow_write=True)
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == READ_TOOLS | WRITE_TOOLS
    for name in WRITE_TOOLS:
        assert tools[name].annotations.read_only_hint is False
        assert tools[name].annotations.destructive_hint is True


async def test_status_and_list():
    server, _ = make_server()
    st = (await call(server, "iacpaas_status")).structured_content
    assert st["backend"] == "mock" and st["write_tools"] == "off"
    res = (await call(server, "list_inforesources", {"filter": "наблюд"})).structured_content
    lst = res["items"]
    assert [i["path"] for i in lst] == ["Учебные/База наблюдений", "Учебные/Онтология наблюдений"]
    assert lst[0]["description"] and res["total"] == 2


async def test_find_then_read_with_full_platform_path():
    server, _ = make_server()
    r = await call(server, "get_inforesource", {"path": "student@example.ru/Мой Фонд/Учебные/База наблюдений$;"})
    info = r.structured_content
    assert info["path"] == IR and info["top_level"] == ["Основные жалобы", "Дополнительные жалобы"]
    assert info["ontology"] == "Учебные/Онтология наблюдений"      # короткий путь, без e-mail и «$;»


async def test_subgraph_depth_and_truncation():
    server, _ = make_server()
    r = (await call(server, "get_subgraph", {"path": IR, "root": "Дополнительные жалобы", "depth": 1})).structured_content
    labels = [n["label"] for n in r["nodes"]]
    assert labels == ["Дополнительные жалобы", "Головная боль", "Температура", "Светобоязнь"]
    assert r["nodes"][3]["link"].endswith("Светобоязнь;")
    t = (await call(server, "get_subgraph", {"path": IR, "depth": 6, "max_nodes": 3})).structured_content
    assert t["truncated"] and t["returned"] == 3 and "depth" in t["hint"]


async def test_count_whole_and_subgraph():
    server, _ = make_server()
    whole = (await call(server, "count_graph", {"path": IR})).structured_content
    assert (whole["vertices"], whole["arcs"], whole["links"]) == (11, 11, 1)
    sub = (await call(server, "count_graph", {"path": IR, "root": "Дополнительные жалобы/Температура"})).structured_content
    assert (sub["vertices"], sub["arcs"]) == (5, 4)          # Температура, тип, 3 значения


async def test_query_summary_export():
    server, _ = make_server()
    q = (await call(server, "query_graph", {"path": IR, "meta": "Наблюдение", "limit": 2})).structured_content
    assert q["total"] == 5 and q["returned"] == 2
    s = (await call(server, "summarize_inforesource", {"path": IR})).structured_content
    assert s["counts"]["vertices"] == 11 and s["meta_stats"][0] == {"meta": "Наблюдение", "count": 4}
    for jt in ("universal", "meta", "simple"):
        e = await call(server, "export_json", {"path": IR, "json_type": jt, "depth": 1})
        assert not e.is_error and e.structured_content["chars"] > 0


async def test_errors_are_diagnosable_for_the_model():
    server, _ = make_server()
    bad = await call(server, "get_inforesource", {"path": "Учебные/База наблюдени"})
    assert bad.is_error and "[not_found]" in bad.content[0].text and "Учебные/База наблюдений" in bad.content[0].text
    deep = await call(server, "get_subgraph", {"path": IR, "depth": 50})
    assert deep.is_error and "less than or equal to 6" in deep.content[0].text
    nocrit = await call(server, "query_graph", {"path": IR})
    assert nocrit.is_error and "критерий" in nocrit.content[0].text
    server2, _ = make_server(max_result_chars=200)
    big = await call(server2, "export_json", {"path": IR})
    assert big.is_error and "[too_large]" in big.content[0].text
    vertex = await call(server, "get_subgraph", {"path": IR, "root": "Нет такой"})
    assert vertex.is_error and "not_found" in vertex.content[0].text and "Похожие ИР" not in vertex.content[0].text


async def test_write_two_step_in_sandbox():
    server, backend = make_server(allow_write=True)
    args = {"path": IR, "concept_path": "Основные жалобы/Светобоязнь",
            "metaconcept_path": "Группа/Наблюдение", "mark": "Фотофобия"}
    async with Client(server) as c:
        prev = (await c.call_tool("update_concept", args)).structured_content
        assert prev["status"] == "pending_confirmation"
        assert prev["preview"]["было"]["label"] == "Светобоязнь"
        assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"
        token = prev["confirm_token"]
        wrong = await c.call_tool("update_concept", {**args, "mark": "Другое", "confirm_token": token})
        assert wrong.is_error and "[confirmation]" in wrong.content[0].text
        ok = (await c.call_tool("update_concept", {**args, "confirm_token": token})).structured_content
        assert ok["status"] == "done"
        again = await c.call_tool("update_concept", {**args, "confirm_token": token})
        assert again.is_error and "использован" in again.content[0].text
        found = (await c.call_tool("query_graph", {"path": IR, "text": "фотофоб"})).structured_content
        assert found["total"] == 1                                # кэш сброшен после записи


async def test_sandbox_delete_import_and_meta_mismatch():
    server, backend = make_server(allow_write=True)
    async with Client(server) as c:
        # неверный metaconcept_path отсекается уже на предпросмотре – токена нет, есть правильное значение
        bad = await c.call_tool("update_concept", {"path": IR, "concept_path": "Основные жалобы/Светобоязнь",
                                                   "metaconcept_path": "Основные жалобы/Наблюдение", "mark": "x"})
        assert bad.is_error and "нужно 'Группа/Наблюдение'" in bad.content[0].text
        assert backend.store[IR]["successors"][0]["successors"][1]["name"] == "Светобоязнь"
        miss = await c.call_tool("update_concept", {"path": IR, "concept_path": "Основные жалобы/Нет такой",
                                                    "metaconcept_path": "Группа/Наблюдение", "mark": "x"})
        assert miss.is_error and "[not_found]" in miss.content[0].text
        d = {"path": IR, "mode": "clearSubGraph", "concept_path": "Дополнительные жалобы"}
        prev = (await c.call_tool("delete_from_fund", d)).structured_content
        assert prev["preview"]["затронет"].startswith("7 вершин")
        await c.call_tool("delete_from_fund", {**d, "confirm_token": prev["confirm_token"]})
        assert backend.store[IR]["successors"][1]["successors"] == []
        ir = {"title": "Новый", "json_type": "universal", "ontology": "x/Онтология$;", "name": "Новый",
              "type": "КОРЕНЬ", "meta": "Онтология", "successors": []}
        p = (await c.call_tool("import_inforesource", {"folder_path": "Учебные", "ir_json": ir})).structured_content
        assert p["preview"]["уже существует"] is False
        await c.call_tool("import_inforesource", {"folder_path": "Учебные", "ir_json": ir,
                                                  "confirm_token": p["confirm_token"]})
        assert "Учебные/Новый" in backend.store
        invalid = await c.call_tool("import_inforesource", {"folder_path": "Учебные", "ir_json": {"title": "x"}})
        assert invalid.is_error and "[invalid_json]" in invalid.content[0].text


async def test_prompt_and_resource():
    server, _ = make_server()
    async with Client(server) as c:
        prompts = (await c.list_prompts()).prompts
        assert prompts[0].name == "analyze_inforesource"
        res = await c.read_resource("iacpaas://guide")
        assert "Лимиты" in res.contents[0].text


async def test_ontology_of_inforesource():
    server, _ = make_server()
    o = (await call(server, "get_ontology", {"path": IR})).structured_content
    assert o["ontology_path"] == "Учебные/Онтология наблюдений" and o["for_inforesource"] == IR
    by = {c["concept"]: c for c in o["concepts"]}
    assert by["Группа"]["cardinality"] == "непустой список"
    assert by["Наблюдение"]["creation"] == "ссылка или новая"
    assert by["нижняя граница"]["valtype"] == "REAL" and by["нижняя граница"]["type"] == "ТЕРМИНАЛ-СОРТ"
    direct = (await call(server, "get_ontology", {"path": "Учебные/Онтология наблюдений"})).structured_content
    assert direct.get("for_inforesource") is None and len(direct["concepts"]) == 6


async def test_model_text_is_compact_and_structured_is_full():
    server, _ = make_server()
    r = await call(server, "get_subgraph", {"path": IR, "root": "Дополнительные жалобы", "depth": 1})
    text = r.content[0].text
    # без пустых полей и отступов (": " внутри текста куска answer – не разделитель JSON)
    assert "null" not in text and "\n" not in text and '": ' not in text and '", "' not in text
    assert '"label"' not in text.split('"nodes"')[1].split("}")[0]            # метка = конец пути, не дублируется
    node = r.structured_content["nodes"][0]
    assert node["label"] == "Дополнительные жалобы" and "valtype" not in node  # structured: без значений по умолчанию


async def test_subgraph_by_meta_counts_direct_children():
    """by_meta – готовый подсчёт прямых потомков root по метапонятиям (элементы группы против вложенных групп)."""
    server, _ = make_server()
    r = (await call(server, "get_subgraph", {"path": IR, "depth": 1})).structured_content
    assert r["by_meta"] == {"Группа": 2}
    assert "counts" not in r              # числа «в пределах depth» GigaChat выдавал за размер раздела – их нет
    leaf =(await call(server, "get_subgraph", {"path": IR, "depth": 0})).structured_content
    assert not leaf.get("by_meta")                               # depth=0 – потомков не смотрели


async def test_export_hides_owner_email():
    """Шапка экспорта (path, ontology) – без e-mail владельца: он уходил бы облачной модели."""
    server, _ = make_server()
    r = await call(server, "export_json", {"path": IR, "depth": 0})
    assert "@" not in r.content[0].text
    assert r.structured_content["data"]["path"] == IR


async def test_summary_sections_largest_and_export_parts():
    """«Самый большой раздел» – два смысла (вершины и дуги) готовой фразой; [too_large] – сразу с частями и размерами."""
    from iacpaas_mcp import graph
    server, _ = make_server()
    s = (await call(server, "summarize_inforesource", {"path": IR})).structured_content
    assert s["sections"] and all({"label", "vertices", "arcs"} <= set(x) for x in s["sections"])
    assert [x["vertices"] for x in s["sections"]] == sorted((x["vertices"] for x in s["sections"]), reverse=True)
    assert s["largest"].startswith("Самый большой раздел")
    two = graph.largest([{"label": "А", "vertices": 10, "arcs": 11}, {"label": "Б", "vertices": 9, "arcs": 20}])
    assert "по вершинам – «А» (10), по дугам – «Б» (20)" in two
    server2, _ = make_server(max_result_chars=200)
    big = (await call(server2, "export_json", {"path": IR})).content[0].text
    assert "[too_large]" in big and "Части первого уровня" in big and "«" in big


async def test_query_graph_word_stems_and_empty_hint():
    """Фраза целиком не находится – поиск по основам слов с hint; ничего нет – hint «в базе этого нет»."""
    server, backend = make_server()
    gas = lambda name: {"type": "НЕТЕРМИНАЛ", "name": name, "meta": "Газ"}                  # noqa: E731
    cls = {"type": "НЕТЕРМИНАЛ", "name": "Газы на основе Гелия", "meta": "Класс газов",
           "successors": [gas("Гелий газообразный высокой чистоты, марка 6.0"), gas("Гелий жидкий")]}

    async def export(path, json_type="universal", start=None, depth=None, no_blob_data=True):
        return {"type": "КОРЕНЬ", "name": "База", "successors": [{"type": "НЕТЕРМИНАЛ", "name": "Моногазы",
                                                                    "meta": "Моногазы", "successors": [cls]}]}
    backend.export = export
    r = (await call(server, "query_graph", {"path": IR, "text": "гелий высокой чистоты 6.0"})).structured_content
    # газ по названию находится раньше поиска по частям слов: совпадение – сам газ, с путём
    assert r["total"] == 1 and r["matches"][0]["label"].endswith("марка 6.0") and "это газ" in r["hint"]
    r = (await call(server, "query_graph", {"path": IR, "text": "гелий высокой чистоты жидкий"})).structured_content
    assert r["total"] >= 1 and "гелий, высоко, чисто" in r["hint"]          # газа такого нет – части слов
    none = (await call(server, "query_graph", {"path": IR, "text": "фтор"})).structured_content
    assert none["total"] == 0 and "в базе этого нет" in none["hint"] and "не из базы" in none["hint"]


async def test_write_only_inside_allowed_folders():
    """IACPAAS_WRITE_PATHS: запись только внутри разрешённых папок – вне их отказ ещё до предпросмотра и токена."""
    server, backend = make_server(allow_write=True, write_paths=["Учебные"])
    upd = {"concept_path": "Основные жалобы/Светобоязнь", "metaconcept_path": "Группа/Наблюдение", "mark": "Фотофобия"}
    async with Client(server) as c:
        ok = (await c.call_tool("update_concept", {"path": IR, **upd})).structured_content
        assert ok["status"] == "pending_confirmation"
        out = await c.call_tool("import_inforesource", {"folder_path": "Загрузки", "clear_if_exists": False,
                                                         "ir_json": {"title": "Т", "name": "Т", "json_type": "universal", "type": "КОРЕНЬ",
                                                                     "meta": "О", "ontology": "О$;"}})
        assert out.is_error and "[read_only]" in out.content[0].text and "«Учебные»" in out.content[0].text
        near = await c.call_tool("delete_from_fund", {"path": "Учебные2/База", "mode": "deleteIR"})
        assert near.is_error and "[read_only]" in near.content[0].text          # префикс – по папке, не по строке


def test_remote_write_needs_folders():
    """На реальную платформу запись включается только вместе с папками для записи."""
    from iacpaas_mcp.server import write_mode

    class Http:
        name, writable = "http", True
    s = make_settings(allow_write=True, remote_write=True)
    assert write_mode(s, Http()) == "off"
    s.write_paths = ["Песочница ИИ"]
    assert write_mode(s, Http()) == "remote"


async def test_mock_create_folder():
    _, backend = make_server()
    r = await backend.create_folder("", "Песочница ИИ", "записи агента")
    assert r["success"] and "Песочница ИИ" in backend.folders


async def test_gas_query_and_calc_over_mcp():
    server, _ = make_server()
    r = await call(server, "gas_query", {"op": "impurities", "gases": "гелий; аргон"})
    d = r.structured_content
    assert d["block"].startswith("{{#") and "Общие примеси" in d["answer"]
    assert "общие примеси" in d["understood"] and d["data_version"]
    r = await call(server, "gas_query", {"op": "calc", "expr": "[Аргон 4.8 | азот] / [Аргон 6.0 | азот]"})
    assert "раза" in r.structured_content["answer"]
    bad = await call(server, "gas_query", {"op": "calc", "expr": "import os"})
    assert bad.is_error and "только числа" in bad.content[0].text


async def test_gas_query_hint_depends_on_client():
    """VS Code и любой клиент без подстановки меток – пересказать answer дословно (иначе человек видит «{{#c038}}»);
    метку вставлять – только нашему агенту (pieces=marks)."""
    args = {"op": "max", "by": "аргон", "top": 3}
    text = (await call(make_server()[0], "gas_query", args)).structured_content["hint"]
    assert "дословно" in text and "{{#" not in text
    marks = (await call(make_server(pieces="marks")[0], "gas_query", args)).structured_content
    assert marks["block"] in marks["hint"]


async def test_platform_tools_give_ready_pieces():
    """Ответы о платформе пишет код: у инструментов чтения – готовый кусок answer и метка block; схема набора ИР и
    дерево – кусок в блоке ```; подсказка – метка нашему агенту, «дословно» – VS Code."""
    server = make_server()[0]
    cnt = (await call(server, "count_graph", {"path": IR})).structured_content
    assert cnt["block"].startswith("{{#") and "вершин 11, дуг 11" in cnt["answer"] and "дословно" in cnt["hint"]
    lst = (await call(server, "list_inforesources", {})).structured_content
    assert f"Всего ИР: {lst['total']} – выданных {lst['given']}" in lst["answer"] and "«База наблюдений»" in lst["answer"]
    tree = (await call(server, "get_subgraph", {"path": IR, "as_tree": True})).structured_content
    assert tree["answer"].startswith("```\n") and tree["tree"] in tree["answer"]
    marks = make_server(pieces="marks")[0]
    m = (await call(marks, "map_inforesources", {})).structured_content
    assert m["answer"] == "```\n" + m["scheme"] + "\n```" and m["block"] in m["hint"]
    assert (await call(marks, "count_graph", {"path": IR})).structured_content["block"] in \
        (await call(marks, "count_graph", {"path": IR})).structured_content["hint"]


async def test_inexact_ir_name_resolves_to_the_only_match():
    """Модель пишет путь ИР неточно («Загрузки/База газов») – если все слова есть в имени ровно одного известного ИР,
    сервер берёт его; неоднозначно – прежняя ошибка с подсказкой."""
    server = make_server()[0]
    ok = (await call(server, "count_graph", {"path": "База наблюдений"})).structured_content
    assert ok["vertices"] == 11
    gas = await call(server, "count_graph", {"path": "Загрузки/База газов"})
    assert not gas.is_error and gas.structured_content["vertices"] > 500
    bad = await call(server, "count_graph", {"path": "Учебные/база"})            # «база» – у двух ИР
    assert bad.is_error and "Похожие ИР" in bad.content[0].text
    typo = await call(server, "count_graph", {"path": "Учебные/База наблюдени"})  # опечатка – не угадываем
    assert typo.is_error


async def test_gases_in_path_of_gas_query():
    """Газы или класс в path (ИР базы) – модель путает path с gases: при пустом gases сервер берёт их как gases (видно в
    understood), иначе – ошибка с подсказкой, а не «вызовите list_inforesources»."""
    server = make_server()[0]
    r = await call(server, "gas_query", {"op": "sort", "by": "кислород", "path": "Моногазы/Газы на основе Аргона"})
    assert not r.is_error and "«Газы на основе Аргона»" in r.structured_content["understood"]
    bad = await call(server, "gas_query", {"op": "list", "gases": "азот", "path": "Азот 6.0"})
    assert bad.is_error and "в gases" in bad.content[0].text


async def test_ontology_piece_is_a_tree():
    """Понятия онтологии приходят по уровням; в куске под родителем – его дети, а не «Класс газов» под «Смесями»."""
    from types import SimpleNamespace as N
    from iacpaas_mcp import pieces
    c = [N(path=p, concept=p.rsplit("/", 1)[-1], cardinality=None, creation=None, choice=None, valtype=None, link=None)
         for p in ("Моногазы", "Смеси", "Моногазы/Класс газов", "Смеси/Класс смесей", "Моногазы/Класс газов/Газ")]
    text = pieces.ontology(N(title="О", ontology_path="О", for_inforesource=None, concepts=c, truncated=False))
    assert [ln.strip() for ln in text.splitlines()[1:]] == [
        "- «Моногазы»", "- «Класс газов»", "- «Газ»", "- «Смеси»", "- «Класс смесей»"]
    assert "\n    - «Газ»" in text and "\n  - «Класс смесей»" in text


def test_agent_starts_server_with_marks(tmp_path):
    """Сервер, который запускает наш агент (stdio), отдаёт метки – их подставляет VerifyMiddleware."""
    from iacpaas_agent.core import AgentRuntime
    from test_agent import settings
    s = settings(tmp_path, tmp_path / "models.toml", mcp="stdio")
    assert AgentRuntime(s)._mcp_target().transport.env["IACPAAS_PIECES"] == "marks"


async def test_search_hints_point_to_gas_query():
    """Поиск по слову-веществу и по названию газа – подсказка ведёт к gas_query, а не к удалённым инструментам."""
    server, _ = make_server()
    r = (await call(server, "query_graph", {"path": "Загрузки/База технологических газов", "text": "неон"})).structured_content
    assert "gas_query(op=\"list\", has=" in r["hint"]
    hints = [(await call(server, "query_graph", {"path": "Загрузки/База технологических газов", "text": t}))
             .structured_content.get("hint", "") for t in ("аргон 6.0", "гелий 7.0")]
    assert all(not h or "gas_query" in h for h in hints)


async def test_count_and_subgraph_accept_gas_name_as_root():
    """«Сколько вершин и дуг у аргона 6.0?» – модель передала название вместо пути: сервер находит газ сам."""
    server, _ = make_server()
    base = "Загрузки/База технологических газов"
    r = (await call(server, "count_graph", {"path": base, "root": "Аргон 6.0"})).structured_content
    assert r["scope"].endswith("Аргон газообразный высокой чистоты марка 6.0") and r["vertices"] > 5
    t = await call(server, "get_subgraph", {"path": base, "root": "Аргон 6.0", "depth": 1})
    assert not t.is_error
