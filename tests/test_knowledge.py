"""Карточки, таблица, проверка по онтологии и подборка – на фрагменте базы газов из fixtures/Загрузки."""
import json

import pytest
from mcp import Client

from conftest import FIXTURES, make_server
from iacpaas_mcp import graph, knowledge

pytestmark = pytest.mark.anyio

GAS = "Загрузки/База технологических газов"
ROOT_FIX = FIXTURES / "Загрузки"
HE6 = "Моногазы/Газы на основе Гелия/Гелий газообразный (сжатый) высокой чистоты, марка 6.0"
MIX = "Многокомпонентные газовые смеси/Газовые смеси Аргон-Гелий/Аргон-Гелий Ar-95% He-5%"


async def call(server, name, args=None):
    async with Client(server) as c:
        return await c.call_tool(name, args or {})


def test_closest_element_for_invented_paths():
    """Модель сочиняет пути по названию – сервер подсказывает настоящий, только если совпали вещество, класс и
    отличительное (число или буква марки, сорт); иначе ничего (не «марка 5.0 → марка А»)."""
    from iacpaas_mcp.knowledge import closest_element
    he = "Моногазы/Газы на основе Гелия/"
    known = [he + "Гелий газообразный марка А", he + "Гелий газообразный (сжатый) высокой чистоты, марка 5.0",
             "Моногазы/Газы на основе Аргона/Аргон газообразный высший сорт"]
    assert closest_element(he + "Гелий газообразный марка 5.0", known) == known[1]
    assert closest_element(he + "Гелий газообразный марка 7.0", known) is None
    assert closest_element("Моногазы/Газы на основе Углекислого газа/Углекислый газ высший сорт", known) is None


def test_impurity_fields_no_data_and_input_error():
    from decimal import Decimal

    from iacpaas_mcp.knowledge import impurity_fields

    def part(name, bound, v):
        return {"name": name, "text": f"{bound} {v} %", "bound": bound, "value": Decimal(v), "unit": "%"}
    main = part("H", "не менее", "99.9999")
    assert impurity_fields([main], 99.9999)["impurities_max_sum"] == "не указаны"      # не «0»
    water = part("Водяные пары", "не более", "0.0002")                                 # в 2 раза – бывает, не ошибка
    assert "exceeds_rest" not in impurity_fields([main, water], 99.9999)
    bad = impurity_fields([main, part("O", "не более", "0.000002"), part("Водяные пары", "не более", "2.0")], 99.9999)
    assert bad["impurities_max_sum"] == "2.000002" and bad["exceeds_rest"] == ["Водяные пары – не более 2.0 %"]


async def test_summary_groups_and_list_total():
    server, _ = make_server()
    s = (await call(server, "summarize_inforesource", {"path": GAS})).structured_content
    assert s["groups"][0] == {"section": "Моногазы", "group": "Газы на основе Гелия", "elements": 10,
                              "element_meta": "Газ"}
    assert "«Моногазы»: 27 элем." in s["groups_hint"] and "«Газы на основе Гелия» – 10" in s["groups_hint"]
    lst = (await call(server, "list_inforesources")).structured_content
    assert lst["total"] == 5 and lst["returned"] == 5


async def test_check_by_ontology_finds_unfilled_elements():
    server, _ = make_server()
    r = (await call(server, "check_inforesource", {"path": GAS})).structured_content
    assert r["ontology"] == "Онтология базы технологических газов"
    assert "Онтология описания характеристик и их значений" in r["ontologies_used"]
    places = {p["path"]: p["problems"] for p in r["problems"]}
    empty = "Моногазы/Газы на основе Двуокиси углерода/Двуокись углерода жидкая высший сорт"
    assert "нет обязательного «Стандарт (норматив)» (в точности один)" in places[empty]
    assert any("не задан ни один вариант" in x for x in
               places["Моногазы/Газы на основе Кислорода/Жидкий кислород/Объемные доли компонентов/2/CO₂"])
    assert r["problems_total"] == 15 and r["places"] == 6
    part = (await call(server, "check_inforesource", {"path": GAS, "root": "Моногазы/Газы на основе Гелия"}))
    assert part.structured_content["problems_total"] == 0


async def test_ontology_marks_alternatives():
    server, _ = make_server()
    r = (await call(server, "get_ontology", {"path": GAS, "root": "Моногазы/Класс газов/Газ/Объемные доли "
                                                                  "компонентов/Компонент", "depth": 1}))
    choices = [c for c in r.structured_content["concepts"] if c.get("choice")]
    assert {c["concept"] for c in choices} >= {"Химическое обозначение", "Совокупность химических веществ",
                                              "Водяные пары"}


async def test_save_selection_two_steps_and_links():
    server, backend = make_server(allow_write=True)
    args = {"source_path": GAS, "title": "Смесь и её газы", "elements": [MIX, HE6], "comment": "проба"}
    pre = (await call(server, "save_selection", args)).structured_content
    assert pre["status"] == "pending_confirmation"
    assert set(pre["preview"]["добавлены, т.к. на них ссылаются"]) == {"Аргон газообразный первый сорт",
                                                                         "Гелий газообразный марка А"}
    assert "Загрузки/Песочница ИИ/Смесь и её газы" not in backend.store
    done = (await call(server, "save_selection", {**args, "confirm_token": pre["confirm_token"]})).structured_content
    assert done["status"] == "done"
    new = backend.store["Загрузки/Песочница ИИ/Смесь и её газы"]
    links = [n["link"] for _, n in graph.walk(new) if "link" in n]
    assert len(links) == 2 and all(link.startswith("user@example.com / Мой Фонд / Загрузки / Песочница ИИ / Смесь и её газы$/Моногазы/") for link in links)
    dumped = json.dumps(new, ensure_ascii=False)
    assert '"link_id"' not in dumped and '"id":' not in dumped     # иначе ссылки вели бы в исходную базу
    lst = (await call(server, "list_inforesources", {"filter": "Смесь и её"})).structured_content
    assert lst["total"] == 1 and lst["items"][0]["kind"].startswith("своя")
    chk = (await call(server, "check_inforesource", {"path": "Загрузки/Песочница ИИ/Смесь и её газы"}))
    assert chk.structured_content["ontology"] == "Онтология базы технологических газов"
    again = await call(server, "save_selection", args)
    assert again.is_error and "[conflict]" in again.content[0].text
    assert "Свободное имя: «Смесь и её газы 2»" in again.content[0].text      # готовое имя – модель не гадает


async def test_selection_keeps_required_sections():
    server, backend = make_server(allow_write=True)
    args = {"source_path": GAS, "title": "Один гелий", "elements": [HE6]}
    pre = (await call(server, "save_selection", args)).structured_content
    await call(server, "save_selection", {**args, "confirm_token": pre["confirm_token"]})
    new = backend.store["Загрузки/Песочница ИИ/Один гелий"]
    assert [n["name"] for n in new["successors"]] == ["Моногазы", "Многокомпонентные газовые смеси"]
    chk = (await call(server, "check_inforesource", {"path": "Загрузки/Песочница ИИ/Один гелий"})).structured_content
    assert chk["problems_total"] == 0, chk


async def test_selection_notes_become_comments():
    server, backend = make_server(allow_write=True)
    args = {"source_path": GAS, "title": "С пометкой", "elements": [HE6],
            "notes": [{"path": HE6, "text": "проверено по ТУ"}]}
    pre = (await call(server, "save_selection", args)).structured_content
    assert pre["preview"]["пометки"] == {HE6.split("/")[-1]: "проверено по ТУ"}
    await call(server, "save_selection", {**args, "confirm_token": pre["confirm_token"]})
    node = graph.find_node(backend.store["Загрузки/Песочница ИИ/С пометкой"], HE6)
    assert node["comment"].endswith("проверено по ТУ")
    stray = await call(server, "save_selection", {**args, "notes": [{"path": MIX, "text": "x"}]})
    assert stray.is_error and "вне подборки" in stray.content[0].text


async def test_save_mixture_draft():
    server, backend = make_server(allow_write=True)
    ar = "Моногазы/Газы на основе Аргона/Аргон газообразный высший сорт"
    co2 = "Моногазы/Газы на основе Двуокиси углерода/Двуокись углерода жидкая высокой чистоты 99,995%"
    args = {"source_path": GAS, "title": "Черновик Ar-CO2", "name": "Аргон-Двуокись углерода Ar-82% CO2-18%",
            "components": [{"gas": ar, "percent": 82}, {"gas": co2, "percent": 18}], "standard": "ГОСТ Р (черновик)", "symbol": "Ar + CO₂"}
    bad = await call(server, "save_mixture", {**args, "components": [{"gas": ar, "percent": 80},
                                                                     {"gas": co2, "percent": 18}]})
    assert bad.is_error and "Сумма долей 98" in bad.content[0].text
    pre = (await call(server, "save_mixture", args)).structured_content
    assert pre["preview"]["состав"] == ["Аргон газообразный высший сорт – 82 %",
                                        "Двуокись углерода жидкая высокой чистоты 99,995% – 18 %"]
    assert "«Двуокись углерода жидкая высокой чистоты 99,995%» в базе записан только названием" in pre["preview"]["внимание"]
    assert "заполненных газов этого вещества в базе нет" in pre["preview"]["внимание"]      # все CO₂ пустые
    assert "ничего ещё не создано" in pre["message"]
    await call(server, "save_mixture", {**args, "confirm_token": pre["confirm_token"]})
    path = "Загрузки/Песочница ИИ/Черновик Ar-CO2"
    links = [n["link"] for _, n in graph.walk(backend.store[path]) if "link" in n]
    assert links == [f"user@example.com / Мой Фонд / Загрузки / Песочница ИИ / Черновик Ar-CO2$/{ar};", f"user@example.com / Мой Фонд / Загрузки / Песочница ИИ / Черновик Ar-CO2$/{co2};"]   # полный путь
    mix = "Многокомпонентные газовые смеси/Черновики смесей/Аргон-Двуокись углерода Ar-82% CO2-18%"
    card = (await call(server, "gas_query", {"op": "mixtures", "path": path})).structured_content
    assert "доля в смеси 82" in card["answer"] and "Двуокись углерода" in card["answer"]
    assert "Ar – не менее" in card["answer"]                                  # чистота газа-компонента по ссылке
    chk = (await call(server, "check_inforesource", {"path": path})).structured_content
    assert all(p["path"] != mix for p in chk["problems"])                 # смесь заполнена по онтологии
    assert any(p["path"] == co2 for p in chk["problems"])                    # а этот CO₂ в базе пустой


async def test_save_selection_guards():
    server, backend = make_server(allow_write=True, write_paths=["Загрузки/Песочница ИИ"])
    outside = await call(server, "save_selection", {"source_path": GAS, "title": "X", "elements": [HE6],
                                                    "folder_path": "Загрузки"})
    assert outside.is_error and "[read_only]" in outside.content[0].text
    missing = await call(server, "save_selection", {"source_path": GAS, "title": "X", "elements": ["Моногазы/Нет"]})
    assert missing.is_error and "[not_found]" in missing.content[0].text
    slash = await call(server, "save_selection", {"source_path": GAS, "title": "a/b", "elements": [HE6]})
    assert slash.is_error and "[invalid_request]" in slash.content[0].text
    assert not any(k.startswith("Загрузки/Песочница ИИ/") for k in backend.store)


def test_value_text_variants():
    from iacpaas_mcp import knowledge as k
    interval = {"type": "НЕТЕРМИНАЛ", "name": "Доля", "meta": "Доля", "successors": [
        {"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": "%", "valtype": "STRING", "meta": "%"},
        {"type": "НЕТЕРМИНАЛ", "name": "Числовой интервал", "meta": "Числовой интервал", "successors": [
            {"type": "НЕТЕРМИНАЛ", "name": "Нижняя граница", "meta": "Нижняя граница", "successors": [
                {"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "value": 1.0, "valtype": "REAL", "meta": "Числовое значение"}]},
            {"type": "НЕТЕРМИНАЛ", "name": "Верхняя граница", "meta": "Верхняя граница"}]}]}
    assert k.value_text(interval) == "от 1.0 до (не указано) %"
    assert k.fmt_num(5e-06) == "0.000005" and k.fmt_num(99.5) == "99.5"


async def test_mixture_row_shows_component_purity():
    server, _ = make_server()
    t = (await call(server, "gas_query", {"op": "mixtures"})).structured_content
    assert "Аргон газообразный первый сорт – доля в смеси 95" in t["answer"]
    assert "(сам газ: Ar – не менее" in t["answer"]


async def test_ir_name_rule():
    """Имя нового ИР – только буквы, цифры, пробел, точка и дефис; иначе отказ с подсказкой и ничего не создаётся."""
    server, backend = make_server(allow_write=True)
    for bad in ("Газы (не менее 99.999 %)", "Ar + CO2", 'Смесь "сварочная"'):
        r = await call(server, "save_selection", {"source_path": GAS, "title": bad, "elements": [HE6]})
        assert r.is_error and "только буквы, цифры" in r.content[0].text, bad
    r = await call(server, "save_selection", {"source_path": GAS, "title": "Чистые газы ≥ 99.999 %", "elements": [HE6]})
    assert "Возьмите «Чистые газы 99.999»" in r.content[0].text                 # готовое имя по правилу
    ok = (await call(server, "save_selection", {"source_path": GAS, "title": "Газы не менее 99.999",
                                                "elements": [HE6]})).structured_content
    assert ok["status"] == "pending_confirmation" and ok["target"].endswith("/Газы не менее 99.999")
    mix = (await call(server, "save_mixture", {"source_path": GAS, "name": "Ar-82% CO2-18%", "components": [
        {"gas": HE6, "percent": 50},
        {"gas": "Моногазы/Газы на основе Аргона/Аргон газообразный высший сорт", "percent": 50}]})).structured_content
    assert mix["target"].endswith("/Черновик смеси Ar-82 CO2-18")      # имя по умолчанию – уже по правилу
    assert not any(k.startswith("Загрузки/Песочница ИИ/") for k in backend.store)


async def test_folder_path_explains_and_lists_irs():
    """Путь папки вместо ИР – подсказка «это папка» и список ИР в ней, а не «нет доступа»."""
    server, _ = make_server(allow_write=True)
    args = {"source_path": GAS, "title": "Один гелий", "elements": [HE6]}
    pre = (await call(server, "save_selection", args)).structured_content
    await call(server, "save_selection", {**args, "confirm_token": pre["confirm_token"]})
    r = await call(server, "get_subgraph", {"path": "Загрузки/Песочница ИИ", "depth": 1})
    text = r.content[0].text
    assert r.is_error and "это папка, а не ИР" in text and "Загрузки/Песочница ИИ/Один гелий" in text



async def test_subgraph_as_tree_shows_values_links_and_empty():
    server, _ = make_server()
    r = (await call(server, "get_subgraph", {"path": GAS, "root": MIX, "depth": 6, "as_tree": True})).structured_content
    tree = r["tree"]
    assert r["nodes"] == [] and tree.startswith("Аргон-Гелий Ar-95% He-5%")
    assert "→ Аргон газообразный первый сорт  [Газ]  (ссылка)" in tree
    assert "95.0  [Числовое значение]" in tree and "└── " in tree
    short = (await call(server, "get_subgraph", {"path": GAS, "root": MIX, "depth": 1, "as_tree": True})).structured_content
    assert short["tree"] == tree                                   # дерево всегда целиком, depth не важен
    whole = (await call(server, "get_subgraph", {"path": GAS, "as_tree": True})).structured_content["tree"]
    assert whole.splitlines()[0] == "База технологических газов"  # у корня онтология не пишется
    assert "(+" in whole and "внутри)" in whole and "Газы на основе Гелия  [Класс газов]" in whole


async def test_big_tree_only_top_levels():
    """Вся база (у настоящей 1478 вершин) резалась на 300 строках, модель рисовала один газ и «…»: теперь у
    большого подграфа – только верхние уровни с числом частей и подсказка; газ целиком – по root."""
    server, _ = make_server()
    r = (await call(server, "get_subgraph", {"path": GAS, "as_tree": True})).structured_content
    lines = r["tree"].splitlines()
    assert r["truncated"] and r["levels"] >= 1 and len(lines) <= 61 and "целиком не рисуется" in r["hint"]
    assert "дерево обрезано" not in r["tree"]
    one = (await call(server, "get_subgraph", {"path": GAS, "root": HE6, "as_tree": True})).structured_content
    assert not one.get("truncated") and "99.9999" in one["tree"]


async def test_ir_list_splits_given_and_own_and_map_shows_links():
    """«Сколько ИР»: выданные и свои раздельно (на сайте видно выданные и папку «Песочница ИИ»). Схема набора ИР:
    онтология каждого ИР, ссылки и клоны между ИР, ИР вне набора."""
    server, _ = make_server(allow_write=True)
    args = {"source_path": GAS, "title": "Один гелий", "elements": [HE6]}
    async with Client(server) as c:
        pre = (await c.call_tool("save_selection", args)).structured_content
        await c.call_tool("save_selection", {**args, "confirm_token": pre["confirm_token"]})
        lst = (await c.call_tool("list_inforesources")).structured_content
        m = (await c.call_tool("map_inforesources")).structured_content
        part = (await c.call_tool("map_inforesources", {"path": "Загрузки/Онтология описания характеристик и их "
                                                                 "значений"})).structured_content
        bad = await c.call_tool("map_inforesources", {"path": "Загрузки/Нет такого"})
    assert (lst["total"], lst["given"], lst["own"]) == (6, 5, 1)
    assert {i["origin"] for i in lst["items"] if i["path"].startswith("Загрузки/Песочница ИИ/")} == {"свой"}
    assert "выданных 5" in lst["hint"] and "своих 1 в папке «Песочница ИИ»" in lst["hint"]
    assert "свои ИР лежат внутри неё" in lst["hint"]
    links = {(x["source"], x["kind"], x["target"]): x for x in m["links"]}
    assert ("База технологических газов", "онтология", "Онтология базы технологических газов") in links
    clones = links[("База технологических газов", "клоны", "База химических элементов и веществ")]
    assert clones["count"] == 160 and "He" in clones["examples"] and clones["in_set"] is False
    assert links[("Онтология базы технологических газов", "ссылки",
                  "Онтология описания характеристик и их значений")]["count"] == 8
    assert ("Один гелий", "онтология", "Онтология базы технологических газов") in links
    assert "База химических элементов и веществ" in m["outside"]
    scheme = m["scheme"].splitlines()
    assert scheme[0] == "Набор ИР: выданных 5 (только чтение), своих 1"
    assert "└── Песочница ИИ – свои ИР" in scheme and "клоны → База химических элементов и веществ: 160" in m["scheme"]
    assert part["scheme"].startswith("Связи ИР «Онтология описания характеристик и их значений»")
    assert [x["source"] for x in part["links"]] == ["Онтология базы технологических газов"]
    assert bad.is_error and "[not_found]" in bad.content[0].text


async def test_delete_preview_lists_similar_own_irs():
    """«Удали подборку газов высокой чистоты» при нескольких похожих: в карточке удаления – похожие свои ИР,
    чтобы человек увидел, что модель выбрала одну из нескольких."""
    server, _ = make_server(allow_write=True)
    async with Client(server) as c:
        for title in ("Газы высокой чистоты", "Газы высокой чистоты 2"):
            args = {"source_path": GAS, "title": title, "elements": [HE6]}
            pre = (await c.call_tool("save_selection", args)).structured_content
            await c.call_tool("save_selection", {**args, "confirm_token": pre["confirm_token"]})
        d = (await c.call_tool("delete_from_fund", {"path": "Загрузки/Песочница ИИ/Газы высокой чистоты",
                                                    "mode": "deleteIR"})).structured_content
    assert d["preview"]["похожие свои ИР – тот ли выбран?"] == ["Газы высокой чистоты 2"]


AR = "Моногазы/Газы на основе Аргона/Аргон газообразный высокой чистоты марка "


def test_component_names_from_data_and_synonyms():
    """Название вещества → обозначение берётся из самой базы; синонимы – из config/components.toml."""
    from iacpaas_mcp import knowledge
    from iacpaas_mcp.config import load_synonyms
    ir = json.loads((ROOT_FIX / "База технологических газов.json").read_text(encoding="utf-8"))
    names = knowledge.component_names(ir)
    assert names["аргона"] == "Ar" and names["гелия"] == "He"          # вещество – из класса «Газы на основе …»
    assert "газообразный" not in names                                 # не первое слово названия
    assert knowledge.resolve_component("аргон", {"Ar"}, names) == "Ar"
    known = {"Ar", "CH₄", "Водяные пары", "O + Ar (суммарно)"}
    syn = {**names, **load_synonyms()}
    assert knowledge.resolve_component("Аргона", known, syn) == "Ar"
    assert knowledge.resolve_component("CH4", known, syn) == "CH₄"
    assert knowledge.resolve_component("метаном", known, syn) == "CH₄"
    assert knowledge.resolve_component("водяных паров", known, syn) == "Водяные пары"
    assert knowledge.resolve_component("фтор", known, syn) is None
    # молекула «H₂» – элемент H из состава; CO₂ остаётся CO₂
    gases = {"H", "N", "O", "CO", "CO₂"}
    assert [knowledge.resolve_component(x, gases, syn) for x in ("H₂", "N2", "O₂", "CO2", "CO")] == \
        ["H", "N", "O", "CO₂", "CO"]
    assert knowledge.resolve_component("H₂", known, syn) is None


def test_rank_exact_impurity_order(monkeypatch):
    """Отдельная примесь Ar (как у водородов на платформе): по убыванию, основной компонент – не примесь."""
    from decimal import Decimal
    from iacpaas_mcp import knowledge
    data = {"H1": [("H", "не менее", "99.999"), ("Ar", "не более", "0.0002")],
            "H2": [("H", "не менее", "99.994"), ("Ar", "не более", "0.002")],
            "H3": [("H", "не менее", "99.9999"), ("Ar", "не более", "0.00002")],
            "A1": [("Ar", "не менее", "99.993"), ("O", "не более", "0.0007")]}
    ir = {"name": "База", "successors": [{"name": "Класс", "successors": [
        {"name": n, "meta": "Газ", "successors": []} for n in data]}]}
    monkeypatch.setattr(knowledge, "composition", lambda node: [
        {"name": c, "text": f"{b} {v} %", "bound": b, "value": Decimal(v), "unit": "%"} for c, b, v in data[node["name"]]])
    res = knowledge.rank_component(ir, "Газ", "Ar")
    assert [(r["name"], r["value"]) for r in res["rows"]] == [("H2", "0.002"), ("H1", "0.0002"), ("H3", "0.00002")]
    assert res["as_main"] == ["A1 (Ar – не менее 99.993 %)"] and res["rows"][0]["rank"] == 1


def test_find_elements_by_names_without_guessing():
    """Названия как в вопросе или как написала модель → пути базы. Числа марки и буквы – точно, слова – без
    окончаний; несколько подходят – все варианты (код не угадывает); нет – похожие только того же вещества."""
    ir = json.loads((FIXTURES / "Загрузки" / "База технологических газов.json").read_text(encoding="utf-8"))
    syn = {"углекислота": "CO₂", "двуокись углерода": "CO₂"}

    def one(q):
        r = knowledge.find_elements(ir, [q], ("Газ", "Газовая смесь"), groups=True, synonyms=syn)
        if r["picked"]:
            return r["picked"][0]["path"].split("/")[-1]
        return r["ambiguous"].get(q) and sorted(x.split("/")[-1] for x in r["ambiguous"][q]) or ("нет", r["missing"][q])
    assert one("аргон 4.8") == "Аргон газообразный высокой чистоты марка 4.8"
    assert one("Ar 6") == one("аргон 6,0") == "Аргон газообразный высокой чистоты марка 6.0"
    assert one("аргона первого сорта") == "Аргон газообразный первый сорт"
    assert one("Гелий газообразный марка Б") == "Гелий газообразный (сжатый), марка Б"    # как писала модель
    assert one("гелий 5.0") == "Гелий газообразный (сжатый) высокой чистоты, марка 5.0"   # не смесь «He-5%»
    assert one("CO2 высший сорт") == "Двуокись углерода жидкая высший сорт"              # обозначение – из синонимов
    assert one("Газы на основе Гелия") == "Газы на основе Гелия"                          # класс целиком
    assert one("гелий А") == ["Гелий газообразный (сжатый, марка А)", "Гелий газообразный марка А"]
    assert one("аргон высший сорт") == ["Аргон газообразный высший сорт", "Аргон жидкий высший сорт"]
    assert len(one("углекислота")) == 4
    assert one("золото") == ("нет", [])
    no, near = one("азот газообразный марка 6.0")             # азота в песочнице нет – гелий 6.0 не подсказывать
    assert no == "нет" and not near
    assert knowledge.split_items("аргон 4.8; «аргон 6.0»\nаргон 4.8") == ["аргон 4.8", "аргон 6.0"]


async def test_save_selection_by_names_and_ambiguous_is_asked():
    server, _ = make_server(allow_write=True)
    r = (await call(server, "save_selection", {"source_path": GAS, "title": "Два аргона",
                                               "elements": "аргон 4.8; аргон 6.0"})).structured_content
    assert r["status"] == "pending_confirmation"
    assert r["preview"]["элементы"] == ["Аргон газообразный высокой чистоты марка 4.8",
                                        "Аргон газообразный высокой чистоты марка 6.0"]
    async with Client(server) as c:
        amb = await c.call_tool("save_selection", {"source_path": GAS, "title": "Гелий А", "elements": "гелий А"})
    assert amb.is_error and "[ambiguous]" in amb.content[0].text and "спроси пользователя" in amb.content[0].text




async def test_save_mixture_accepts_gas_names():
    """Живой T58: модель назвала газы как в ответе gas_query, а не путями – сервер находит пути сам."""
    server, _ = make_server(allow_write=True)
    pre = await call(server, "save_mixture", {"source_path": GAS, "name": "Аргон-Гелий 80-20", "components": [
        {"gas": "Аргон газообразный первый сорт", "percent": 80}, {"gas": "Гелий 5.0", "percent": 20}]})
    assert not pre.is_error, pre.content[0].text
    assert pre.structured_content["preview"]["состав"][0] == "Аргон газообразный первый сорт – 80 %"
