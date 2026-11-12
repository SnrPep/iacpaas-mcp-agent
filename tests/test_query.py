"""Запрос к таблице газов: каждый тип вопроса из docs/questions.md – на снимке настоящей базы (tests/snapshot).
Ожидаемые значения выписаны из самой базы (составы газов), а не из проверяемого кода."""
import json
from pathlib import Path

import pytest

from iacpaas_mcp import query
from iacpaas_mcp.answers import AnswerError
from iacpaas_mcp.config import load_synonyms

BASE = Path(__file__).resolve().parent / "snapshot" / "Загрузки" / "База технологических газов.json"


@pytest.fixture(scope="module")
def ir():
    return json.loads(BASE.read_text(encoding="utf-8"))


def q(ir, **kw):
    return query.run(ir, synonyms=load_synonyms(), **kw)


def names(r):
    return [x["name"] for x in r["rows"]]


def test_table_has_all_gases_and_mixture(ir):
    rows = query.gas_table(ir)
    assert len(rows) == 48
    assert sum(1 for g in rows if not g["filled"]) == 10
    mix = query.mixture_table(ir)
    assert [c["share"] for c in mix[0]["components"]] == ["95.0", "5.0"]


def test_list_and_count_of_class(ir):
    r = q(ir, op="list", gases="аргоны")
    assert r["total"] == 6 and "«Газы на основе Аргона»" in r["understood"]
    r = q(ir, op="count", gases="газы на основе водорода")
    assert r["answer"].startswith("**6 газов**")


def test_classes_most_and_unfilled(ir):
    r = q(ir, op="classes")
    assert "Всего газов: 48, классов: 12" in r["answer"]
    assert "Газы на основе Гелия – 10 газов – больше всего" in r["answer"]
    r = q(ir, op="unfilled")
    assert r["total"] == 10 and "Метан газообразный чистый" in names(r)


def test_card_value_and_ppm(ir):
    r = q(ir, op="card", gases="азот 6.0", unit="ppm")
    assert "O – не более 0.000065 % (0.65 ppm)" in r["answer"]
    assert "ТУ 20.11.11-009-45905715-2017" in r["answer"]
    r = q(ir, op="card", gases="неон 5.5")
    assert "Ne – не менее 99.999 %" in r["answer"]


def test_where_thresholds_and_ppm_threshold(ir):
    r = q(ir, op="list", gases="гелий", where="азот <= 0.0002")
    # N у гелиев: 7.0 – 0.0000015 … по данным не больше 0.0002 – только марки 5.0 и выше и 4.6 (0.0003 – нет)
    assert all("Гелий" in n for n in names(r)) and "Гелий газообразный высокой чистоты, марка 4.6" not in names(r)
    r2 = q(ir, op="list", gases="гелий", where="азот <= 2 ppm")
    assert names(r2) == names(r)


def test_where_relative_to_gas(ir):
    r = q(ir, op="list", gases="азот", where="чистота > Азот высокой чистоты (марка 6.0)")
    assert r["total"] == 0 and "99.9999" in r["understood"]
    r = q(ir, op="list", gases="азот", where="чистота >= Азот высокой чистоты (марка 6.0)")
    assert names(r) == ["Азот высокой чистоты (марка 6.0)"]


def test_has_and_without(ir):
    r = q(ir, op="list", gases="азот", has="водород")
    assert r["total"] == 6 and "Азот жидкий технический 1 сорт" not in names(r)
    r = q(ir, op="list", gases="аргон", without="метан")
    assert set(names(r)) == {"Аргон газообразный высший сорт", "Аргон газообразный первый сорт",
                             "Аргон жидкий высший сорт"}


def test_except_state_grade_standard(ir):
    r = q(ir, op="list", gases="аргон", name_not="жидк; первый сорт")
    assert r["total"] == 4
    assert names(q(ir, op="list", gases="аргон", state="жидкий")) == ["Аргон жидкий высший сорт"]
    assert q(ir, op="list", grade="5.5")["total"] == 4                 # ещё криптон и ксенон 5.5 – без состава
    assert q(ir, op="list", standard="ГОСТ 10157")["total"] == 3
    # «кроме … для шаров» моделью в standard – не стандарт: подсказка, а не «таких газов нет»
    with pytest.raises(AnswerError, match="не стандарт"):
        q(ir, op="card", gases="гелий", standard="не для шаров")
    assert "Азот жидкий технический 1 сорт" in names(q(ir, op="list", gases="азот", grade="первый"))


def test_extremes_with_all_ties(ir):
    r = q(ir, op="max", gases="водород", by="чистота")
    assert set(names(r)) == {"Водород газообразный особо чистый марки Б", "Водород газообразный чистый высший сорт"}
    r = q(ir, op="max", gases="азот", by="кислород")
    assert set(names(r)) == {"Азот жидкий технический 1 сорт", "Азот газообразный технический 1 сорт"}
    assert "поровну у 2" in r["understood"]
    assert names(q(ir, op="min", gases="азот", by="кислород")) == ["Азот высокой чистоты (марка 6.0)"]


def test_top_n_and_sort(ir):
    r = q(ir, op="max", by="аргон", top=3)
    assert names(r)[0] == "Водород газообразный чистый второй сорт" and r["total"] == 3
    r = q(ir, op="sort", gases="аргон", by="кислород", order="asc")
    assert names(r)[0] == "Аргон газообразный высокой чистоты марка 6.0"
    assert names(r)[-1] == "Аргон газообразный первый сорт"


def test_compare_table_and_named_without_composition(ir):
    r = q(ir, op="compare", gases="кислород 3.0; кислород 3.5; кислород 4.0", by="сумма примесей")
    assert "| Сумма примесей (все «не более»), % | 0.094 | 0.051 | 0.0104 |" in r["answer"]
    assert "Меньше всего сумма примесей: «Кислород газообразный высокой чистоты марка 4.0»" in r["answer"]
    r = q(ir, op="max", gases="криптон 5.5; ксенон 5.5", by="чистота")
    assert "только название" in r["answer"] and r["total"] == 0


def test_common_impurities_of_two_classes(ir):
    r = q(ir, op="impurities", gases="водород; аргон")
    assert "**Общие примеси (есть в каждой группе):** O, N, CH₄, Водяные пары." in r["answer"]
    assert "Только у «Газы на основе Водорода»: Ar, O + N (суммарно)." in r["answer"]
    assert "Только у «Газы на основе Аргона»: C, CO₂, H." in r["answer"]


def test_errors_in_composition(ir):
    r = q(ir, op="errors", gases="водород")
    assert "Водород газообразный особо чистый марки Б" in names(r)
    assert "вероятна ошибка ввода" in r["answer"]


def test_mixture_parts_and_shares(ir):
    r = q(ir, op="mixtures")
    assert "Аргон газообразный первый сорт – доля в смеси 95.0 %" in r["answer"]
    assert "Гелий газообразный марка А – доля в смеси 5.0 %" in r["answer"]


def test_absent_component_and_gas(ir):
    assert "«хлор» в составах базы нет" in q(ir, op="list", has="хлор")["answer"]
    assert "Газа «хлор» в базе нет" in q(ir, op="list", gases="хлор")["answer"]


def test_ambiguous_name_takes_all_and_says_so(ir):
    r = q(ir, op="card", gases="азот первого сорта")
    assert "взяты все" in r["answer"] and r["total"] >= 3


def test_unknown_op_and_missing_by_are_errors(ir):
    with pytest.raises(AnswerError, match="Умею"):
        q(ir, op="average")
    with pytest.raises(AnswerError, match="нужен by"):
        q(ir, op="max", gases="азот")


def test_calc_division_difference_sum_plain(ir):
    syn = load_synonyms()
    r = query.calc(ir, "[Аргон 4.8 | азот] / [Аргон 6.0 | азот]", synonyms=syn)
    assert r["result"] == "33.33" and "больше в **33.33** раза" in r["answer"]
    r = query.calc(ir, "[Аргон 6.0 | чистота] - [Аргон газообразный первый сорт | чистота]", synonyms=syn)
    assert "= 0.0129 %" in r["answer"]
    r = query.calc(ir, "[Аргон 5.5 | кислород] + [Аргон 5.5 | азот]", synonyms=syn)
    assert "= 0.00035 %" in r["answer"]
    assert query.calc(ir, "17 * 23")["result"] == "391"


def test_calc_refuses_ambiguous_and_text(ir):
    with pytest.raises(AnswerError, match="несколько"):
        query.calc(ir, "[азот первого сорта | кислород] * 2", synonyms=load_synonyms())
    with pytest.raises(AnswerError, match="только числа"):
        query.calc(ir, "__import__('os')")


def test_same_plan_same_block_and_answer(ir):
    a = q(ir, op="max", gases="водород", by="чистота")
    b = q(ir, op="max", gases="водород", by="чистота")
    assert a == b
    assert query.block_id("gas_query", {"op": "max"}) == query.block_id("gas_query", {"op": "max"})


def test_calc_accepts_quoted_expression_and_extreme_has_own_header(ir):
    r = query.calc(ir, '"[Аргон 4.8 | азот] / [Аргон 6.0 | азот]"', synonyms=load_synonyms())
    assert r["result"] == "33.33"
    a = q(ir, op="max", gases="азот", by="кислород")["answer"]
    assert a.startswith("**Больше всего O (хуже всего) – поровну у 2:**")


def test_card_only_asked_odd_note_mixture_count_errors_hint(ir):
    """Живые: «сколько ppm неона в гелии 5.5» – только неон; сортировка водородов по воде – пометка об ошибке ввода
    (2 % и 10 % у особо чистых); «сколько смесей» – число; «примесей больше, чем самого газа» – подсказка op=errors."""
    r = q(ir, op="card", gases="гелий 5.5", show="неон", unit="ppm")
    assert "Ne не более 0.0001 % (1 ppm)" in r["answer"] and "O + Ar" not in r["answer"] and "остаток" not in r["answer"]
    r = q(ir, op="sort", gases="водород", by="водяные пары", order="asc")
    assert "вероятна ошибка ввода" in r["answer"] and "марки Б" in r["answer"].split("_")[-2]
    assert "Газовых смесей в базе: 1." in q(ir, op="mixtures")["answer"]
    with pytest.raises(AnswerError, match="op=\"errors\""):
        q(ir, op="list", where="сумма примесей > чистота")


def test_where_zero_is_without_and_state_in_where_is_hinted(ir):
    """Живой Q2: модель писала where «метан = 0; state != жидкий»."""
    a = q(ir, op="list", gases="аргон", where="метан = 0")
    assert names(a) == names(q(ir, op="list", gases="аргон", without="метан")) and "без CH₄" in a["understood"]
    with pytest.raises(AnswerError, match="state"):
        q(ir, op="list", gases="аргон", where="метан = 0; state != жидкий")


def test_calc_takes_expression_out_of_call_garbage(ir):
    """Живой Q3: GigaChat дописал в expr служебный мусор вызова – 15 одинаковых вызовов до предела шагов."""
    junk = '"[Аргон 4.8 | азот] / [Аргон 6.0 | азот]"}}![]() ассистент function callrecipient{'
    assert query.calc(ir, junk, synonyms=load_synonyms())["result"] == "33.33"
    with pytest.raises(AnswerError):
        query.calc(ir, "ассистент [Аргон 4.8 | азот]")              # мусор в начале – не угадываем


def test_whole_list_says_how_many_gases_in_selection(ir):
    """Список всей базы: в куске 38 газов с составом, а газов в базе 48 – сноска называет, сколько всего в отборе
    («сколько их» – 48, а не 38). Отбор по составу – прежняя сноска «сравнить не с чем»."""
    r = q(ir, op="list")
    assert r["total"] == 38
    assert "Ещё 10 газов только с названием (состав не заполнен) – всего в отборе 48 газов" in r["answer"]
    assert "сравнить не с чем" in q(ir, op="list", without="метан")["answer"]
