import json
import time

import pytest

from conftest import FIXTURES, IR
from iacpaas_mcp import graph
from iacpaas_mcp.paths import PathError, full_ir_path, normalize_concept_path, normalize_ir_path, split_labels
from iacpaas_mcp.policy import WritePolicy

IR_JSON = json.loads((FIXTURES / "Учебные" / "База наблюдений.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("raw", [
    "Учебные/База наблюдений",
    " Учебные / База наблюдений ",
    "student@example.ru/Мой Фонд/Учебные/База наблюдений$;",
    "student@example.ru / Мой Фонд / Учебные / База наблюдений$;",
    "Мой Фонд/Учебные/База наблюдений",
])
def test_ir_path_forms(raw):
    assert normalize_ir_path(raw) == "Учебные/База наблюдений"


def test_triple_prime_and_concept_paths():
    assert split_labels("начало / ‴г/л‴ / конец") == ["начало", "г/л", "конец"]
    assert normalize_concept_path("/Основные жалобы/Светобоязнь") == "Основные жалобы/Светобоязнь"
    assert normalize_concept_path("u@x/Мой Фонд/Учебные/База$/Группа/‴a/b‴;") == "Группа/‴a/b‴"
    assert full_ir_path("Учебные/База", "u@x.ru") == "u@x.ru/Мой Фонд/Учебные/База$;"
    with pytest.raises(PathError):
        split_labels("a/‴незакрыто")
    with pytest.raises(PathError):
        normalize_ir_path("   ")


def test_count_with_link():
    assert graph.count(IR_JSON) == {"vertices": 11, "arcs": 11, "links": 1, "terminals": 3}


def test_flatten_paths_are_reusable():
    nodes, truncated = graph.flatten(IR_JSON, start="Дополнительные жалобы", max_nodes=100)
    assert not truncated
    temp = next(n for n in nodes if n["label"] == "Температура")
    assert temp["path"] == "Дополнительные жалобы/Температура" and temp["children"] == 1
    assert graph.find_node(IR_JSON, temp["path"])["name"] == "Температура"
    link = next(n for n in nodes if "link" in n)
    assert link["label"] == "Светобоязнь"
    _, cut = graph.flatten(IR_JSON, max_nodes=3)
    assert cut


def test_find_and_validate():
    m, total = graph.find(IR_JSON, meta="Наблюдение")
    assert total == 5 and len(m) == 5          # 4 вершины + 1 ссылка
    assert graph.validate(IR_JSON) == []
    bad = {"title": "X", "type": "НЕТЕРМИНАЛ", "json_type": "universal", "ontology": "o$;", "meta": "m",
           "successors": [{"type": "ТЕРМИНАЛ-ЗНАЧЕНИЕ", "meta": "v"}]}
    errs = graph.validate(bad)
    assert any("КОРЕНЬ" in e for e in errs) and any("valtype" in e for e in errs) and any("value" in e for e in errs)


def test_policy_token_bound_single_use_and_expiry():
    p = WritePolicy(enabled=True, ttl_s=60)
    args = {"a": 1}
    t = p.issue("op", args)
    assert p.check("op", {"a": 2}, t) is not None          # другие аргументы
    assert p.check("other", args, t) is not None           # другая операция
    assert p.check("op", args, t) is None                  # годен
    assert "использован" in p.check("op", args, t)         # одноразовый
    assert p.issue("op", args) != p.issue("op", args)       # в одну секунду – разные (раньше совпадали)
    old = f"{int(time.time()) - 120}.ab12.{p._sign('op', args, int(time.time()) - 120, 'ab12')}"
    assert "просрочен" in p.check("op", args, old)
    assert "формат" in p.check("op", args, "мусор")
