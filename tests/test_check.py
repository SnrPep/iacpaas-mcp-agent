"""Проверка ответа (один режим): куски по меткам, связка без газов и чисел, вопрос о базе без запроса, запрос ↔
вопрос. Случаи – живые вопросы и ошибки модели из прошлых версий."""
from decimal import Decimal

from iacpaas_agent.check import check, glue_problems, plan_remarks, subjects

GASES = {"Аргон газообразный высокой чистоты марка 6.0", "Водород газообразный особо чистый марки Б"}
GRADES = {"6.0", "4.8", "5.5", "1", "2"}
PIECE = {"block": "{{#ab12}}", "answer": "**2 газа:**\n- **Водород газообразный особо чистый марки Б** – H не "
         "менее 99.9999 %", "understood": "больше всего: газы – «Газы на основе Водорода»; самые чистые"}


def run(text, question, calls=(("gas_query", {"op": "max", "gases": "водород", "by": "чистота"}),),
        results=(("gas_query", PIECE),), wrote=False):
    return check(text, question, list(results), list(results), list(calls), GASES, GRADES, wrote)


def test_piece_is_filled_and_answer_is_from_code():
    r = run("Самые чистые водороды: {{#ab12}}", "Какой водород самый чистый?")
    assert r["problems"] == [] and r["remarks"] == []
    assert "99.9999 %" in r["text"] and r["source"] == "code" and r["understood"] == [PIECE["understood"]]


def test_answer_with_pieces_is_pieces_only():
    """Ответ по данным – только куски кода: текст модели вокруг них (с числами и газами) не показывается и повтора
    не вызывает; без кусков – текст модели проверяется, как прежде."""
    for glue in ("У него чистота 99.9999 %.", "Лучший – Водород газообразный особо чистый марки Б."):
        r = run("{{#ab12}} " + glue, "Какой водород самый чистый?")
        assert r["problems"] == [] and glue not in r["text"] and r["text"] == PIECE["answer"]
    r = run("У водорода марки Б чистота 99.9999 %.", "Какой водород самый чистый?", calls=(), results=())
    assert any("числа 99.9999" in p for p in r["problems"])


def test_grade_numbers_items_and_question_numbers_are_allowed():
    assert glue_problems("1. Аргон 6.0 и 4.8:\n### 2. Азот 1 сорт", "", GASES, GRADES, set()) == []
    assert glue_problems("Порог 0,01 % – из вопроса", "больше 0,01 %", GASES, GRADES, set()) == []
    assert glue_problems("6.0 %", "", GASES, GRADES, set())           # с % – значение, а не марка


def test_unknown_mark_and_missing_mark():
    assert any("такого куска" in p for p in run("{{#ffff}}", "Какой водород самый чистый?")["problems"])
    assert any("нет меток" in p for p in run("Вот ответ.", "Какой водород самый чистый?")["problems"])


def test_base_question_without_query_is_error_general_is_not():
    r = run("Аргон – инертный газ, в базе его много.", "Какие аргоны есть в базе?", calls=(), results=())
    assert any("сначала gas_query" in p for p in r["problems"])
    r = run("ppm – миллионная доля.", "Что такое ppm?", calls=(), results=())
    assert r["problems"] == [] and r["source"] == "model"
    r = run("Привет! Спрашивайте о газах.", "привет", calls=(), results=())
    assert r["problems"] == []


def test_refusal_to_change_base_needs_no_query():
    """«Удали аргон 6.0 из базы» – отказ без запроса к базе верный; раньше проверка требовала gas_query, модель
    путалась и ответом становилось «ответ по базе не собран»."""
    r = run("Удалять аргон 6.0 из выданной базы нельзя – она только для чтения. Могу сохранить копию в вашу папку.",
            "Удали аргон 6.0 из базы", calls=(), results=())
    assert r["problems"] == [] and r["source"] == "model"


def test_model_arithmetic_must_go_to_calc():
    r = run("Будет 391.", "Сколько будет 17 умножить на 23?", calls=(), results=())
    assert any("op=\"calc\"" in p for p in r["problems"])


def test_platform_numbers_from_tool_are_allowed():
    res = (("list_inforesources", {"total": 9, "items": []}),)
    r = run("Всего ИР: 9.", "какие ир есть?", calls=(("list_inforesources", {}),), results=res)
    assert r["problems"] == [] and r["source"] == "data"


def test_platform_piece_must_be_inserted_by_mark():
    """Схему или ответ о платформе ИИ не переписывает сама: кусок кода есть – нужна его метка; с меткой – «собрал
    код», без – замечание (переписанную схему с перепутанными строками числа бы не поймали)."""
    piece = {"block": "{{#cd34}}", "answer": "```\nНабор ИР: выданных 9\n├── База газов\n```", "given": 9}
    res = (("map_inforesources", piece),)
    ok = run("Вот схема: {{#cd34}}", "выведи структуру ИР", calls=(("map_inforesources", {}),), results=res)
    assert ok["problems"] == [] and ok["source"] == "code" and "├── База газов" in ok["text"]
    bad = run("Набор ИР: выданных 9\n- База газов", "выведи структуру ИР", calls=(("map_inforesources", {}),),
              results=res)
    assert any("нет меток" in p for p in bad["problems"])
    # кусок вставлен, а его числа ещё раз пересказаны ИИ – пересказ не показывается (ответ – только кусок)
    twice = run("{{#cd34}} Итого выданных ИР – 9.", "выведи структуру ИР", calls=(("map_inforesources", {}),),
                results=res)
    assert not twice["problems"] and "Итого" not in twice["text"]
    # список ИР запрошен только чтобы найти путь для записи – в ответе о записи его метка не нужна
    lst = {"block": "{{#ee11}}", "answer": "**Всего ИР: 9**"}
    save = run("Сохранение отменено.", "Сохрани кислород 5.0 в мою папку",
               calls=(("list_inforesources", {}), ("save_selection", {})), results=(("list_inforesources", lst),))
    assert not any("нет меток" in p for p in save["problems"])


def test_platform_numbers_are_listed_with_their_tool():
    """Под ответом по данным платформы видно, какое число с каким инструментом сверено (а не «… и ещё 2»)."""
    res = (("count_graph", {"nodes": 1451, "arcs": 1450}),)
    r = run("В ИР «Подборка газов» 1451 вершина и 1450 дуг.", "сколько там вершин и дуг",
            calls=(("count_graph", {}),), results=res)
    assert r["checked"] == 2
    assert r["matched"] == ["1451 – есть в ответе инструмента count_graph",
                            "1450 – есть в ответе инструмента count_graph"]


def test_write_confirmation_needs_no_mark_and_may_name_gases():
    res = (("save_selection", {"ok": True, "path": "Загрузки/Песочница ИИ/Подборка"}),)
    r = run("Подборка сохранена: Аргон газообразный высокой чистоты марка 6.0.", "Сохрани аргон 6.0",
            calls=(("save_selection", {}),), results=res, wrote=True)
    assert r["problems"] == []


def test_plan_must_match_question():
    def notes(q, *calls):
        return [m for m, _ in plan_remarks(q, list(calls))]
    assert notes("Какие аргоны без примеси метана?", {"op": "list", "gases": "аргон"})
    assert not notes("Какие аргоны без примеси метана?", {"op": "list", "gases": "аргон", "without": "метан"})
    # общие примеси – живой вопрос преподавателей
    q = "какие общие примеси есть в газах на основе водорода и на основе аргона"
    assert notes(q, {"op": "list", "gases": "водород; аргон"})
    assert not notes(q, {"op": "impurities", "gases": "водород; аргон"})
    # две крайние части – два запроса
    q = "Какой гелий хуже всего по неону и какой лучше всего по азоту?"
    assert notes(q, {"op": "max", "gases": "гелий", "by": "неон"})
    assert not notes(q, {"op": "max", "gases": "гелий", "by": "неон"}, {"op": "min", "gases": "гелий", "by": "азот"})
    # «сложи» – сумма, а не отношение (живой случай v3)
    q = "Сложи допустимые кислород и азот в аргоне 5.5"
    assert notes(q, {"op": "calc", "expr": "[Аргон 5.5 | кислород] / [Аргон 5.5 | азот]"})
    assert not notes(q, {"op": "calc", "expr": "[Аргон 5.5 | кислород] + [Аргон 5.5 | азот]"})
    # порог из вопроса – в where; ppm
    assert notes("Есть газы, где водорода в примесях больше 0,1 %?", {"op": "list", "where": "H > 0.01"})
    assert not notes("Есть газы, где водорода в примесях больше 0,1 %?", {"op": "list", "where": "H > 0.1"})
    assert notes("сколько ppm кислорода можно в азоте 6.0?", {"op": "card", "gases": "азот 6.0"})
    # «какой азот …» – запрос по азотам, а не по всей базе (живой Q1 v3)
    assert notes("Какие азоты хуже всего по кислороду?", {"op": "max", "by": "кислород"})
    assert not notes("Какие азоты хуже всего по кислороду?", {"op": "max", "gases": "азот", "by": "кислород"})


def test_no_false_remarks_on_plain_questions():
    def notes(q, *calls):
        return [m for m, _ in plan_remarks(q, list(calls))]
    assert not notes("Какие газы на основе аргона есть в базе?", {"op": "list", "gases": "аргон"})
    assert not notes("а есть азот чище чем 6.0?", {"op": "list", "gases": "азот",
                                                   "where": "чистота > Азот высокой чистоты (марка 6.0)"})
    assert not notes("Какой газ чище: криптон 5.5 или ксенон 5.5?", {"op": "max", "by": "чистота",
                                                                      "gases": "криптон 5.5; ксенон 5.5"})
    assert not notes("Сколько в базе газов, у которых только название, без состава?", {"op": "unfilled"})
    assert not notes("Отсортируй азоты по сумме примесей по убыванию", {"op": "sort", "gases": "азот",
                                                                         "by": "сумма примесей"})
    assert subjects("у какого кислорода больше всего воды") == ["кислород"]
    assert subjects("кислорода больше всего у каких газов") == []


def test_platform_numbers_are_exact():
    from iacpaas_agent.check import _numbers
    assert _numbers({"a": [1, "вершин 4536"], "b": 0.5}) == {Decimal("1"), Decimal("4536"), Decimal("0.5")}


def test_extreme_direction_by_and_scope():
    """«лучше по примеси» – min, «хуже» – max, по чистоте наоборот; по чему и среди каких газов (живой Q1)."""
    def notes(q, *calls):
        return [m for m, _ in plan_remarks(q, list(calls))]
    q = "Какие азоты лучше всего по водороду, а какие хуже всего по кислороду? Если значения равны – назови все."
    good = ({"op": "min", "gases": "азот", "by": "водород"}, {"op": "max", "gases": "азот", "by": "кислород"})
    assert not notes(q, *good)
    wrong = notes(q, {"op": "max", "gases": "азот", "by": "водород"}, {"op": "min", "by": "кислород"})
    assert len(wrong) == 2 and all("op=\"" in n for n in wrong)
    assert notes(q, {"op": "min", "gases": "азот", "by": "водород"}, {"op": "max", "by": "кислород"})   # не среди азотов
    cases = [
        ("Какой водород самый чистый?", {"op": "max", "gases": "водород", "by": "чистота"}),
        ("Где меньше всего метана среди аргонов?", {"op": "min", "gases": "аргон", "by": "метан"}),
        ("У какого газа в примесях больше всего кислорода?", {"op": "max", "by": "O"}),
        ("выдай три газа с наибольшими примесями аргона", {"op": "max", "by": "аргон", "top": 3}),
        ("Какой азот хуже всего по кислороду и лучше по водяным парам?",
         {"op": "max", "gases": "азот", "by": "кислород"}),
    ]
    for q, call in cases:
        extra = [{"op": "min", "gases": "азот", "by": "водяные пары"}] if "водяным" in q else []
        assert not notes(q, call, *extra), (q, notes(q, call, *extra))
    assert notes("Какой водород самый чистый?", {"op": "min", "gases": "водород", "by": "чистота"})
    q = "Какой гелий хуже всего по неону, какой лучше всего по азоту и у какого больше всего водяных паров?"
    calls = [{"op": "max", "gases": "гелий", "by": "неон"}, {"op": "min", "gases": "гелий", "by": "азот"},
             {"op": "max", "gases": "гелий", "by": "водяные пары"}]
    assert not notes(q, *calls) and notes(q, *calls[:2])
    q = "Сравни аргон 4.8, 5.5 и 6.0 по всем примесям таблицей и скажи, у какого сумма примесей меньше всего"
    assert not notes(q, {"op": "compare", "gases": "Аргон 4.8; Аргон 5.5; Аргон 6.0", "by": "сумма примесей"})


def test_scope_remark_names_the_part():
    """«вопрос о газах азот» – только про ту часть, где он назван (живой Q2: модель сузила и другую часть)."""
    q = "2. Какие газы имеют азота больше 0,01 %?\n3. Какой азот хуже всего по кислороду?"
    got = [m for m, _ in plan_remarks(q, [{"op": "list", "where": "N > 0.01"}, {"op": "max", "by": "кислород"}])]
    assert any("Какой азот хуже всего" in m and "другие части не меняй" in m for m in got)
    assert not any("Какие газы имеют" in m for m in got)


def test_meaning_is_checked_on_calls_whose_piece_is_shown():
    """Живой Q2: верный запрос («без метана») был, но в ответ вставлен кусок другого, без фильтров."""
    good = {"block": "{{#aaaa}}", "answer": "**3 газа:** – A", "understood": "без CH₄"}
    bad = {"block": "{{#bbbb}}", "answer": "**6 газов:** – B", "understood": "все"}
    calls = [("gas_query", {"op": "list", "gases": "аргон", "without": "метан"}),
             ("gas_query", {"op": "list", "gases": "аргон"})]
    res = [("gas_query", good), ("gas_query", bad)]
    args = {"{{#aaaa}}": calls[0][1], "{{#bbbb}}": calls[1][1]}
    q = "Какие аргоны без примеси метана?"
    shown_bad = check("Аргоны: {{#bbbb}}", q, res, res, calls, GASES, GRADES, block_args=args)
    assert shown_bad["remarks"]
    shown_good = check("Аргоны: {{#aaaa}}", q, res, res, calls, GASES, GRADES, block_args=args)
    assert not shown_good["remarks"]


def test_corrections_rebuild_query_from_question():
    """Живые Q2 и Q7: код чинит запрос модели по явным признакам вопроса."""
    from iacpaas_agent.check import corrections
    q = ("1. Перечисли все аргоны без примеси метана, кроме жидких, и отсортируй их по кислороду по убыванию. "
         "2. Какие газы имеют азота больше 0,01 % и при этом кислорода меньше 0,001 %? "
         "3. Какой азот хуже всего по кислороду и лучше по водяным парам?")
    used = [("{{#aaaa}}", {"op": "sort", "by": "кислород", "order": "desc"}),
            ("{{#bbbb}}", {"op": "list", "where": "N > 0.01; O < 0.001"}),
            ("{{#cccc}}", {"op": "max", "by": "кислород"}), ("{{#dddd}}", {"op": "min", "by": "водяные пары"})]
    got = {f["old"]: f["args"] for f in corrections(q, used)}
    assert got["{{#aaaa}}"] == {"op": "sort", "by": "кислород", "order": "desc", "gases": "аргон",
                                "without": "метана", "state": "газообразный"}
    assert "{{#bbbb}}" not in got                                # «какие газы» – все газы базы, не трогаем
    assert got["{{#cccc}}"]["gases"] == "азот" and got["{{#dddd}}"]["op"] == "min"
    q = "Какой гелий хуже всего по неону и какой лучше всего по азоту?"
    got = corrections(q, [("{{#aaaa}}", {"op": "max", "by": "неон", "gases": "Гелий"}),
                          ("{{#bbbb}}", {"op": "min", "by": "азот"})])
    assert [(f["old"], f["args"]["gases"], f["args"]["op"]) for f in got] == [("{{#bbbb}}", "гелий", "min")]
    # верный запрос – исправлять нечего
    assert not corrections("Какой водород самый чистый?", [("{{#aaaa}}", {"op": "max", "gases": "водород",
                                                                          "by": "чистота"})])


def test_full_bench_failures_are_closed():
    """Провалы полного стенда v4: каждое правило – на живом вопросе, верный запрос – без замечаний."""
    from iacpaas_agent.check import corrections

    def notes(q, *calls):
        return [m for m, _ in plan_remarks(q, list(calls))]

    def fixes(q, *used):
        return [f["args"] for f in corrections(q, [(f"{{{{#{i:04x}}}}}", a) for i, a in enumerate(used)])]
    # S10: цены – не о базе
    r = run("В базе цен нет – их уточняют у поставщика.", "сколько стоит баллон аргона?", calls=(), results=())
    assert r["problems"] == []
    # S12: «суммарно сколько примесей» – готовая сумма газа, не «+»
    q = "Суммарно сколько примесей допускается в кислороде марки 4.0?"
    assert not any("expr" in n for n in notes(q, {"op": "card", "gases": "Кислород 4.0", "show": "сумма примесей"}))
    assert notes(q, {"op": "calc", "expr": "[Кислород 4.0 | CH₄] + [Кислород 4.0 | CO₂]"})
    assert fixes(q, {"op": "calc", "expr": "[Кислород 4.0 | CH₄] + [Кислород 4.0 | CO₂]"}) == [
        {"op": "card", "gases": "Кислород 4.0", "show": "сумма примесей"}]
    # S13: топ-3 самых грязных
    q = "дай топ-3 самых грязных газа"
    assert not notes(q, {"op": "max", "by": "сумма примесей", "top": 3})
    assert fixes(q, {"op": "max", "by": "сумма примесей"})[0]["top"] == 3
    # S17, T49: «примесей больше, чем самого газа» – ошибки состава; модель их не вызвала – код добавит
    for q in ("у какого газа примесей больше чем самого газа?", "У каких газов сумма примесей больше, чем остаток до 100 %?"):
        assert notes(q, {"op": "list", "where": "сумма примесей > чистота"})
        assert {"op": "errors", "gases": ""} in fixes(q)
    # T18: «сколько газовых смесей» – не «сколько газов»
    assert not notes("Сколько газовых смесей в базе?", {"op": "mixtures"})
    # T27: «а у какого меньше всего» – по тем же водяным парам и среди кислородов
    q = "У какого кислорода больше всего водяных паров, а у какого меньше всего?"
    good = ({"op": "max", "gases": "кислород", "by": "водяные пары"}, {"op": "min", "gases": "кислород",
                                                                        "by": "водяные пары"})
    assert not notes(q, *good) and notes(q, good[0], {"op": "min", "by": "водяные пары"})
    # T55: «а есть азот чище, чем 6.0» – среди азотов
    q = "а есть азот чище чем 6.0?"
    assert notes(q, {"op": "list", "where": "чистота > Азот 6.0"})
    assert fixes(q, {"op": "list", "where": "чистота > Азот 6.0"})[0]["gases"] == "азот"
    # Q1, часть 2: модель потеряла порог – код возвращает его в where
    q = "Перечисли гелии, у которых азота не больше 0,0002 %, и отсортируй их по чистоте от самого чистого"
    got = fixes(q, {"op": "sort", "gases": "гелий", "by": "чистота", "order": "desc"})
    assert got and got[0]["where"] == "азота <= 0.0002"
    # T5: «сколько ppm неона в гелии 5.5» – только неон
    q = "сколько ppm неона в гелии 5.5?"
    assert fixes(q, {"op": "card", "gases": "Гелий 5.5", "unit": "ppm"})[0]["show"] == "неона"
    # общие примеси – живой вопрос преподавателей, код добавит запрос сам
    q = "какие общие примеси есть в газах на основе водорода и на основе аргона"
    assert {"op": "impurities", "gases": "водород; аргон"} in fixes(q, {"op": "list", "gases": "водород"})


def test_standard_from_piece_non_answer_and_reuse_of_good_call():
    """Живые T3 / D12: номер стандарта из куска в тексте модели – не выдумка; T63: «Понял условие» – не ответ;
    D3: «чище всего» – max по чистоте, верный запрос модели уже был – берётся его кусок."""
    from iacpaas_agent.check import corrections
    piece = {"block": "{{#aaaa}}", "answer": "**Гелий 4.6** (стандарт ТУ 0271-001-45905715-2016)", "understood": "card"}
    res = [("gas_query", piece)]
    r = check("Стандарт – ТУ 0271-001-45905715-2016: {{#aaaa}}", "По какому стандарту выпускается гелий 4.6?", res,
              res, [("gas_query", {"op": "card", "gases": "Гелий 4.6"})], GASES, GRADES)
    assert r["problems"] == []
    r = check("Стандарт – ТУ 1111-222: {{#aaaa}}", "По какому стандарту выпускается гелий 4.6?", res, res,
              [("gas_query", {"op": "card", "gases": "Гелий 4.6"})], GASES, GRADES)
    assert not r["problems"] and "1111-222" not in r["text"]      # выдумка модели не показывается
    r = run("Хорошо, понял условие. Готов отвечать на ваши вопросы.", "Что такое ppm?", calls=(), results=())
    assert any("не ответ" in p for p in r["problems"])
    good = ("{{#00aa}}", {"by": "чистота", "gases": "азот", "op": "max", "state": "жидкий"})
    bad = ("{{#00bb}}", {"by": "чистота", "gases": "азот", "op": "min"})
    got = corrections("Какой из них чище всего?", [bad], [good, bad])
    assert got == [{"old": "{{#00bb}}", "use": "{{#00aa}}", "args": good[1], "part": "Какой из них чище всего"}]


def test_correction_takes_ready_query_of_the_part():
    """Модель сделала верный запрос на часть, но вставила кусок другого (с порогами второй части): код берёт верный
    запрос, а не «чинит» чужой – иначе пороги второй части давали пустой список."""
    from iacpaas_agent.check import corrections
    q = ("1. Перечисли все аргоны без примеси метана, кроме жидких, и отсортируй их по кислороду по убыванию.\n"
         "2. Какие газы имеют азота больше 0,01 % и при этом кислорода меньше 0,001 %?")
    right = ("{{#00a1}}", {"by": "кислород", "gases": "Аргон", "op": "sort", "order": "desc", "state": "газообразный",
                           "without": "метан"})
    thr = ("{{#00a2}}", {"op": "list", "where": "N > 0.01; O < 0.001"})
    sort_all = ("{{#00a3}}", {"by": "кислород", "op": "sort", "order": "desc"})
    got = corrections(q, [sort_all, thr], [right, thr, sort_all])
    first = [f for f in got if f["part"].startswith("1.")]
    assert first and first[0].get("use") == "{{#00a1}}" and "where" not in first[0]["args"]


def test_until_100_percent_is_not_a_base_value():
    """«остаток до 100 %» в связке – целое, а не значение базы; «чистота 100 %» – по-прежнему выдумка."""
    assert glue_problems("Сумма примесей больше остатка до 100 %.", "", GASES, GRADES, set()) == []
    assert glue_problems("Чистота 100 %.", "", GASES, GRADES, set())


def test_last_full_bench_failures():
    """Живые S5, T20, T29 последнего полного стенда."""
    from iacpaas_agent.check import corrections

    def notes(q, *calls):
        return [m for m, _ in plan_remarks(q, list(calls))]
    # S5: вставлен кусок без ppm – код делает запрос только спрошенного и в ppm («можно» между словами – не мешает)
    q = "сколько ppm кислорода можно в азоте 6.0?"
    ppm = ("{{#00aa}}", {"gases": "Азот 6.0", "op": "card", "unit": "ppm"})
    plain = ("{{#00bb}}", {"gases": "Азот 6.0", "op": "card"})
    got = corrections(q, [plain], [ppm, plain])
    assert got[0]["old"] == "{{#00bb}}" and got[0]["args"]["unit"] == "ppm" and got[0]["args"]["show"] == "кислорода"
    assert any(f["args"].get("show") == "кислорода" for f in corrections(q, [ppm], [ppm]))   # «можно» между словами
    # T20: «12» калькулятором вместо куска classes – замечание и кусок classes
    q = "Сколько классов газов в базе?"
    classes, bare = ("{{#00cc}}", {"op": "classes"}), ("{{#00dd}}", {"op": "calc", "expr": "12"})
    assert notes(q, bare[1]) and not notes(q, classes[1])
    assert {"old": "{{#00dd}}", "use": "{{#00cc}}", "args": classes[1], "part": q} in corrections(q, [bare],
                                                                                                    [classes, bare])
    assert not notes("Сколько будет 17 умножить на 23?", {"op": "calc", "expr": "17 * 23"})
    # T29: «больше всего неона в примесях» – крайние, а не сумма примесей
    assert not notes("Где больше всего неона в примесях?", {"op": "max", "by": "неон"})
    assert notes("Сколько всего примесей в кислороде 4.0?", {"op": "calc", "expr": "[Кислород 4.0 | O] + 1"})


def test_grade_from_question_must_be_in_query_and_calc_result_may_be_repeated():
    """Живые D4 и T44: модель выдумала «Аргон газообразный марки А» вместо марки 4.8 – код ставит «Аргон 4.8»;
    итог расчёта «33.33» повторён в тексте – не выдумка (число кода)."""
    from iacpaas_agent.check import corrections
    q = "Сравни его с маркой 4.8 по всем примесям таблицей"
    used = [("{{#0001}}", {"op": "compare", "gases": "Аргон газообразный высокой чистоты марка 6.0; Аргон газообразный "
                                                    "марки А"})]
    assert plan_remarks(q, [used[0][1]])
    assert corrections(q, used, used)[0]["args"]["gases"] == "Аргон газообразный высокой чистоты марка 6.0; Аргон 4.8"
    assert not plan_remarks("Во сколько раз в аргоне 4.8 азота больше, чем в аргоне 6.0?",
                            [{"op": "calc", "expr": "[Аргон 4.8 | азот] / [Аргон 6.0 | азот]"}])
    piece = {"block": "{{#cccc}}", "answer": "**0.001 / 0.00003 = 33.33**", "understood": "расчёт", "result": "33.33"}
    r = check("Больше в 33.33 раза: {{#cccc}}", "Во сколько раз в аргоне 4.8 азота больше?", [("gas_query", piece)],
              [("gas_query", piece)], [("gas_query", {"op": "calc", "expr": "[Аргон 4.8 | N] / [Аргон 6.0 | N]"})],
              GASES, GRADES)
    assert r["problems"] == []
    r = check("Больше в 40 раз: {{#cccc}}", "Во сколько раз в аргоне 4.8 азота больше?", [("gas_query", piece)],
              [("gas_query", piece)], [("gas_query", {"op": "calc", "expr": "[Аргон 4.8 | N] / [Аргон 6.0 | N]"})],
              GASES, GRADES)
    assert not r["problems"] and "40" not in r["text"]           # своё число модели не показывается


def test_name_from_piece_with_numbers_is_not_a_value():
    """Живой S21: модель назвала смесь «Аргон-Гелий Ar-95% He-5%» – числа в названии из куска не выдумка."""
    piece = {"block": "{{#eeee}}", "answer": "**Аргон-Гелий Ar-95% He-5%** – компоненты:\n- Аргон – доля в смеси 95.0 %",
             "understood": "смеси"}
    res, calls = [("gas_query", piece)], [("gas_query", {"op": "mixtures"})]
    assert check("В смеси «Аргон-Гелий Ar-95% He-5%»: {{#eeee}}", "Какая доля аргона в смеси?", res, res, calls,
                 GASES, GRADES)["problems"] == []
    own = check("Доля аргона – 95 %: {{#eeee}}", "Какая доля аргона в смеси?", res, res, calls, GASES, GRADES)
    assert not own["problems"] and "Доля аргона – 95" not in own["text"]   # число своими словами не показывается


def test_sort_by_and_direction_and_single_value_repeat():
    """Живые S36 и S5: сортировка не по тому – замечание и исправление (by, order, has); число из карточки одного
    спрошенного значения повторять можно."""
    from iacpaas_agent.check import corrections
    q = "выдай газы с наиб примесеями аргона - отсортируй по увелечение примесей метана"
    wrong = {"by": "аргон", "op": "sort", "order": "desc", "show": "метан"}
    assert any("метана" in m for m, _ in plan_remarks(q, [wrong]))
    fixed = corrections(q, [("{{#0001}}", wrong)], [("{{#0001}}", wrong)])[0]["args"]
    assert (fixed["by"], fixed["order"], fixed["has"]) == ("метана", "asc", "аргона")
    assert not plan_remarks("Отсортируй аргоны по кислороду по возрастанию",
                            [{"op": "sort", "gases": "аргон", "by": "O", "order": "asc"}])
    piece = {"block": "{{#ffff}}", "answer": "**Азот 6.0**\n- O не более 0.000065 % (0.65 ppm)", "understood": "card"}
    args = {"op": "card", "gases": "Азот 6.0", "show": "кислорода", "unit": "ppm"}
    r = check("Кислорода не более 0.65 ppm: {{#ffff}}", "сколько ppm кислорода можно в азоте 6.0?",
              [("gas_query", piece)], [("gas_query", piece)], [("gas_query", args)], GASES, GRADES,
              block_args={"{{#ffff}}": args})
    assert r["problems"] == []


def test_follow_target_picks_the_question_it_refers_to():
    """«из них» – к последнему вопросу о газах; «вернёмся к …» – к вопросу с теми же словами; без отсылки – не
    уточнение; «вернёмся» ни к чему не подходит – не угадываем."""
    from iacpaas_agent.check import follow_target
    prev = [("Какие аргоны без метана?", {"op": "list", "gases": "аргон", "without": "метан"}, []),
            ("Какие газы на основе кислорода?", {"op": "list", "gases": "кислород"}, []),
            ("Какие гелии есть?", {"op": "list", "gases": "гелий"}, [])]
    assert follow_target("а какой из них хуже всего по водяным парам?", prev)[0] == "Какие гелии есть?"
    assert follow_target("вернёмся к аргонам без метана – какой из них самый чистый?", prev)[0] == prev[0][0]
    assert follow_target("вернёмся к кислородам: какие из них жидкие?", prev)[0] == prev[1][0]
    assert follow_target("Какой газ самый чистый?", prev) is None
    assert follow_target("вернёмся к неонам – какой из них чище?", prev) is None
    # совпадение по действию («самый чистый») – не ссылка: сравниваются только слова «к …»
    pure = [("Какой водород самый чистый?", {"op": "max", "gases": "водород", "by": "чистота"}, [])]
    assert follow_target("вернёмся к кислородам – какой из них самый чистый?", pure) is None
    # последний вопрос без запроса о газах (ответ по памяти, запись) – «из них» не к более раннему
    memo = prev + [("Напомни, сколько газов на основе аргона?", None, [])]
    assert follow_target("А какие из них жидкие?", memo) is None
    assert follow_target("вернёмся к кислородам: какие из них жидкие?", memo)[0] == prev[1][0]


def test_follow_part_drops_reference_to_earlier_question():
    """«вернёмся к …» – ссылка до тире, запятой или слова вопроса; по остатку сверяется запрос. Класс из ссылки –
    тоже предмет вопроса (если того прошлого вопроса нет)."""
    from iacpaas_agent.check import follow_part
    assert follow_part("вернёмся к аргонам без метана – какой из них самый чистый?") == "какой из них самый чистый?"
    assert follow_part("давай вернёмся к азотам, какие из них жидкие?") == "какие из них жидкие?"
    assert follow_part("вернёмся к аргонам без метана какой из них самый чистый") == "какой из них самый чистый"
    assert follow_part("Какой газ самый чистый?") == "Какой газ самый чистый?"
    assert subjects("вернёмся к кислородам – какой из них самый чистый?") == ["кислород"]


def test_follow_fix_keeps_scope_of_earlier_query():
    """Запрос уточнения – отбор прошлого запроса (класс, «без») + своё действие; свои условия не теряются."""
    from iacpaas_agent.check import follow_fix
    prev = {"op": "list", "gases": "аргон", "without": "метан"}
    assert follow_fix({"op": "max", "by": "чистота"}, prev) == {"op": "max", "by": "чистота", "gases": "аргон",
                                                                "without": "метан"}
    assert follow_fix({"op": "list", "without": "азот"}, prev)["without"] == "метан; азот"
    assert follow_fix({"op": "max", "by": "азот", "gases": "водород"}, prev)["gases"] == "аргон"
    # класс в path у модели – с gases сервер запрос не примет: path убирается, ИР базы – остаётся
    assert "path" not in follow_fix({"op": "max", "by": "чистота", "path": "Моногазы/Газы на основе Аргона"}, prev)
    base = "Загрузки/База технологических газов"
    assert follow_fix({"op": "max", "by": "чистота", "path": base}, prev)["path"] == base



def test_multi_part_question_takes_ready_queries_of_its_parts():
    """Живой Q1 (5 частей): модель сделала верные запросы на сортировку и «во сколько раз», но не вставила их метки –
    код берёт их куски, нового не придумывает; «самый чистый водород» не «чинится» на сравнении другой части (без
    by); ошибки состава – по газам своей части («у водородов»), а не первой (азоты). «Сколько газов» – список тоже
    ответ."""
    from iacpaas_agent.check import corrections
    q = ("1. Какие азоты лучше всего по водороду? 2. Перечисли гелии, у которых азота не больше 0,0002 %, и "
         "отсортируй их по чистоте. 3. Какой водород самый чистый, и есть ли у водородов ошибки в составе? "
         "4. Сравни кислород 3.0 и 4.0 по метану таблицей и скажи, во сколько раз метана в 3.0 больше, чем в 4.0.")
    best_n = ("{{#n1}}", {"op": "min", "gases": "азот", "by": "водород"})
    sort_he = ("{{#s1}}", {"op": "sort", "gases": "гелий", "by": "чистота", "where": "N <= 0.0002 %"})
    pure_h = ("{{#h1}}", {"op": "max", "gases": "водород", "by": "чистота"})
    cmp_o = ("{{#c1}}", {"op": "compare", "gases": "Кислород 3.0; Кислород 4.0", "show": "метан"})
    calc = ("{{#r1}}", {"op": "calc", "expr": "[Кислород 3.0 | метан] / [Кислород 4.0 | метан]"})
    used = [best_n, cmp_o]
    fixes = corrections(q, used, [best_n, sort_he, pure_h, cmp_o, calc])
    uses = {f.get("use") for f in fixes}
    assert {"{{#s1}}", "{{#r1}}", "{{#h1}}"} <= uses
    assert not any(f.get("old") == "{{#c1}}" for f in fixes)            # сравнение части 4 не трогается
    err = next(f for f in fixes if f["args"].get("op") == "errors")
    assert err["args"]["gases"] == "водород"
    assert not plan_remarks("Напомни, сколько газов на основе аргона?", [{"op": "list", "gases": "аргон"}])
