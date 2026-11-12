"""Разметка ответов на странице веб-чата: блоки кода, дерево, нумерация списков.

Функции берутся прямо из index.html (разделы «Утилиты» и «Markdown … Блоки кода») и выполняются в Node.js –
так проверяется тот же код, что работает в браузере. Без Node.js тесты пропускаются.
"""
import json
import shutil
import subprocess

import pytest

from conftest import ROOT

PAGE = ROOT / "src" / "iacpaas_agent" / "web" / "static" / "index.html"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="нужен Node.js")


def _section(html: str, start: str, end: str) -> str:
    i = html.index(start)
    return html[i:html.index(end, i)]


def render(*samples: str) -> list[str]:
    html = PAGE.read_text(encoding="utf-8")
    script = html[html.index("<script>", html.index("</svg>")):]
    js = ("globalThis.sessionStorage={getItem:()=>null};globalThis.window=globalThis;\n"
          + _section(script, "/* ===== Утилиты", "/* ===== Тема")
          + _section(script, "/* ===== Markdown", "/* ===== Состояние")
          + "\nconst out=JSON.parse(require('fs').readFileSync(0,'utf8')).map(md);"
          + "process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([NODE, "-e", js], input=json.dumps(samples), capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_code_block_language_highlight_and_line_numbers():
    code = '```python\ndef add(a, b=0):\n    # сумма\n    return a + b\nprint(f"x {add(1)}")\n```'
    [h] = render(code)
    assert 'class="lang"' in h and ">Python<" in h and "Копировать" in h
    assert '<span class="t-kw">def</span>' in h and '<span class="t-def">add</span>' in h
    assert '<span class="t-com"># сумма</span>' in h and '<span class="t-num">0</span>' in h
    assert '<span class="t-bi">print</span>' in h and 't-str' in h
    assert '<span class="gut" aria-hidden="true">1\n2\n3\n4</span>' in h      # номера строк не попадают в копию


def test_code_block_guess_language_and_escape():
    sh, js = render("```\npython sum.py --fast\n```", "```js\nconst s = '<b>' // x\n```")
    assert ">Консоль<" in sh and '<span class="t-fn">python</span>' in sh and '<span class="t-attr">--fast</span>' in sh
    assert "class=\"gut\"" not in sh                                                 # одна строка – без номеров
    assert "&lt;b&gt;" in js and "'<b>'" not in js                                  # HTML из кода экранирован


def test_tree_from_box_drawing():
    tree = "```\n[Корень]\n├── Раздел\n│   ├── Аксиомы\n│   └── Теоремы – с доказательством\n└── Метазнания\n```"
    [h] = render(tree)
    assert 'class="code tree"' in h and ">Дерево<" in h
    assert h.count("<details open>") == 2                                            # Корень, Раздел – ветки
    assert h.count('class="leaf"') == 3                                              # Аксиомы, Теоремы, Метазнания
    assert '<span class="tr">– с доказательством</span>' in h
    assert "├──" in h.split("<code hidden>")[1]                                      # копируется исходный текст


def test_numbered_list_keeps_numbers_across_bullets_and_blank_lines():
    lazy = "1. **Типы**\n- а\n- б\n\n1. **Карточность**\n\nТекст\n- в\n\n1. **Создание**\n   - новая"
    explicit = "1. один\n\n2. два\n\n3. три"
    restart = "1. а\n2. б\n\n### Заголовок\n\n1. снова"
    a, b, c = render(lazy, explicit, restart)
    assert [a.count(f'data-n="{n}"') for n in (1, 2, 3)] == [1, 1, 1]
    assert "<li>новая" in a                                                           # подпункт вложен
    assert [b.count(f'data-n="{n}"') for n in (1, 2, 3)] == [1, 1, 1]
    assert c.count('data-n="1"') == 2 and 'data-n="3"' not in c                       # после заголовка – заново


def test_tables_alignment_and_table_inside_code_fence():
    fenced = ("```\n| Раздел | Вершины | Дуг |\n|---|---|---|\n| Арифметика | 1850 | 2177 |\n"
              "| Пределы функций | 354 | 415 |\n```")
    plain = "| Имя | Тип |\n|:--|:-:|\n| a | b |"
    pandas = "|   epsilon |    N | ok   |\n|----------:|-----:|:-----|\n|     0.1   |   11 | True |"   # как to_markdown
    code = "```python\n| не таблица |\n|---|\n```"
    f, p, d, c = render(fenced, plain, pandas, code)
    assert 'class="tbl"' in f and 'class="code' not in f                      # таблица в ``` – рисуется таблицей
    assert '<td class="num nw">1850</td>' in f and '<th class="num nw">Вершины</th>' in f   # числа – одной ширины
    # всё по левому краю: :-: и ---: из разделителя не применяются (модели ставят их вразнобой)
    assert '<th class="nw">Имя</th>' in p and '<td class="nw">b</td>' in p
    assert '<td class="num nw">0.1</td>' in d and '<td class="nw">True</td>' in d and "right" not in d
    assert 'class="code' in c                                                   # с языком – это код
    # короткая ячейка – в одну строку (nw), длинная – переносится по словам (wrap), а не по буквам
    [w] = render("| Газ | Сумма |\n|---|---|\n| Гелий газообразный (сжатый) высокой чистоты, марка 6.0 | 0.0003 |")
    assert '<td class="wrap">Гелий' in w and '<td class="num nw">0.0003</td>' in w


def test_strike_tasks_and_autolinks():
    [h] = render("~~старое~~ новое\n\n- [x] сделано\n- [ ] осталось\n\nСм. https://iacpaas.dvo.ru/docs.")
    assert "<s>старое</s>" in h
    assert 'class="task done">сделано' in h and 'class="task">осталось' in h
    assert '<a href="https://iacpaas.dvo.ru/docs"' in h                        # точка в конце – не часть ссылки


def test_dangling_or_empty_code_fence_is_hidden():
    """«…командой: ```python», а дальше модель вызвала инструмент (или текст ещё печатается) – кавычки не видны."""
    dangling, empty, ok = render("Посмотреть можно командой: ```python", "Текст\n\n```\n\n```\nдальше",
                                 "```python\nprint(1)\n```")
    assert "```" not in dangling and "командой:" in dangling
    assert "```" not in empty and 'class="code' not in empty and "дальше" in empty
    assert 'class="code' in ok


def wrong_nums(reports: list[dict]) -> list[list[str]]:
    """Какие числа страница подчеркнёт по замечаниям проверки (wrongNums из index.html)."""
    html = PAGE.read_text(encoding="utf-8")
    js = (_section(html, "const WRONG_NUM", "/* Где ошибка")
          + "const out=JSON.parse(require('fs').readFileSync(0,'utf8')).map(v=>wrongNums(v).map(x=>x[0]));"
          + "process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([NODE, "-e", js], input=json.dumps(reports), capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_wrong_number_underlined_for_every_kind_of_problem():
    """Неверное число ИИ подчёркивается при любом виде замечания; число из данных – нет."""
    g = "Азот газообразный высокой чистоты (марка 5.5)"
    probs = [f"«{g}»: O в данных – не более 0.0002 %, в ответе 0.002",
             "число 0.0129 % не найдено в данных инструментов",
             f"«{g}»: числа 0.07 в данных этого газа нет",
             f"«{g}»: Водяные пары 0.0002 – состава этого газа в данных хода нет, число не из данных",
             f"«{g}»: 0.0003 в данных – это H, а не O",
             "«в 33 раз» не сходится с данными: 0.001 и 0.00003 – в 33.33",
             "«в 5 000 000 раз» не сходится с данными: 0.4 и 0.000001 – в 400000.00",
             "подсчёт «17 газов…»: в данных – 15, 49",
             "итог 9 не сходится с данными: 7 + 1 = 8",
             "расчёт неверен: 0.001 / 0.00003 = 3.33 (по коду – 33.33)",
             "расчёт неверен: 0.0002 + 0.0001 = 0.0004 = 0.0005 (по коду – 0.0003)",
             f"«{g}»: O в данных есть – не более 0.0002 %, а в ответе «нет»",
             "в ответе нет 2 из 6 газов результата: «Гелий»"]
    got = wrong_nums([{"problems": [p]} for p in probs])
    assert got == [["0.002"], ["0.0129"], ["0.07"], ["0.0002"], ["0.0003"], ["33"], ["5 000 000"], ["17"], ["9"],
                   ["3.33"], ["0.0005"], [], []]


def test_wrong_number_only_where_its_gas_is_named():
    """Замечание про гелий 6.0 («Водяные пары 0.0002 – …»): то же 0.0002 у аргона 4.8 верно и не подчёркивается;
    в абзаце с гелием 6.0 – подчёркивается; замечание без газа – везде."""
    html = PAGE.read_text(encoding="utf-8")
    why = ("«Гелий газообразный (сжатый) высокой чистоты, марка 6.0»: Водяные пары 0.0002 – состава этого газа в "
           "данных хода нет, число не из данных")
    places = ["Аргон газообразный высокой чистоты марка 4.8 — не более 0.0002 %",
              "Среди газов с минимальным содержанием водяных паров (0.0002 % у ряда газов, включая «Гелий газообразный "
              "(сжатый) высокой чистоты, марка 6.0»)", "У гелия 6.0 водяных паров 0.0002 %", "Гелий 5.5 – 0.0002 %"]
    js = (_section(html, "const WRONG_NUM", "/* Где ошибка")
          + f"const why={json.dumps(why, ensure_ascii=False)};"
          + "const out=JSON.parse(require('fs').readFileSync(0,'utf8')).map(t=>gasHere(t,gasKey(why)));"
          + "out.push(gasHere('Аргон 4.8 – 0.0002 %',gasKey('число 0.0002 % не найдено')));"
          + "process.stdout.write(JSON.stringify(out));")
    r = subprocess.run([NODE, "-e", js], input=json.dumps(places, ensure_ascii=False), capture_output=True, text=True,
                       encoding="utf-8", timeout=30)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [False, True, True, False, True]


def test_dollar_formula_on_own_line_is_block_any_length():
    long = "$" + r"\text{" + "а" * 250 + "} = 1" + "$"
    block, inline_ = render(long, "текст $a + b$ текст")
    assert 'class="math' in block and "\text" not in block.split("data-tex")[0]   # не сырой LaTeX абзацем
    assert 'class="mi"' in inline_ and 'class="math' not in inline_
