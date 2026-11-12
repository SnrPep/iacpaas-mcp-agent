"""Ещё вопросы к стенду: «со всех сторон» – состав и чистота, ppm, марки и сорта, двойники, газы «только
название», смеси, подсчёты, «хуже / лучше всего» (несколько частей, равные значения), пороги и их пересечение, «без»,
«кроме», сортировка, сравнение таблицей, «в N раз», суммы, ошибки базы, вопросы списком, опечатки, запись в песочницу
(разрешить / отклонить), общие вопросы; диалог D2 с памятью. Эталоны – кодом сервера по той же выгрузке (числа не
руками); где эталон зависит от данных, условие проверяется assert – выгрузка поменяется, вопрос не соврёт молча.
Подключается в agent_bench.py: T1 … T63, D2 (группа – «T*»), набор пользователя Q1 … Q7 («Q*»)."""
from __future__ import annotations

import re
from decimal import Decimal

N = "Моногазы/Газы на основе Азота"
HE = "Моногазы/Газы на основе Гелия"
O = "Моногазы/Газы на основе Кислорода"
AR = "Моногазы/Газы на основе Аргона"

# ответ «в базе не заполнено» в любых словах
EMPTY = ("не заполн", "только назван", "нет данных", "не указан", "отсутств", "нет подробн", "нет информац",
         "нет сведен")
ASK = ("какой из них", "какой именно", "какой нужен", "уточн", "какой из", "выбер", "укаж")
NO = ("нет", "не найден", "отсутств", "не содерж")
REFUSED = ("отмен", "отклон", "не сохран", "не выполн")


def num(text: str) -> Decimal:
    """«не более 0.0005 %» → 0.0005"""
    return Decimal(text.split()[2])


def ratio(a: Decimal, b: Decimal) -> tuple[str, ...]:
    """«в N раз» – любое честное округление: 33, 33.3, 33.33, 33.333."""
    r = a / b
    return tuple(dict.fromkeys("#" + format(r.quantize(Decimal(10) ** -k).normalize(), "f") for k in range(4)))


def more(d, case) -> dict[str, dict]:
    from iacpaas_mcp import graph
    rows = d.rows

    def group(g):
        return {n: r for n, r in rows.items() if r["group"] == f"Газы на основе {g}"}

    def comp(name):
        return rows[name].get("composition") or {}

    def main(name):
        return "#" + format(Decimal(str(rows[name]["main_value"])).normalize(), "f")

    def all_vals(name):
        return ["#" + c.split()[2] for c in list(comp(name).values())[1:]]

    def ppm(name):
        return ["#" + v.split()[0] for v in rows[name]["composition_ppm"].values()]

    def path(name):
        return rows[name]["path"]

    o50 = "Кислород газообразный особой чистоты (марка 5.0)"
    n55, n60 = "Азот газообразный высокой чистоты (марка 5.5)", "Азот высокой чистоты (марка 6.0)"
    he46 = "Гелий газообразный высокой чистоты, марка 4.6"
    he55 = "Гелий газообразный (сжатый) высокой чистоты, марка 5.5"
    he60 = "Гелий газообразный (сжатый) высокой чистоты, марка 6.0"
    he70 = "Гелий газообразный высокой чистоты (сжатый, марка 7.0)"
    he_a = ["Гелий газообразный марка А", "Гелий газообразный (сжатый, марка А)"]
    ar48, ar55, ar60 = (f"Аргон газообразный высокой чистоты марка {m}" for m in ("4.8", "5.5", "6.0"))
    ar1 = "Аргон газообразный первый сорт"
    h_b = "Водород газообразный особо чистый марки Б"
    ne = "Неон газообразный высокой чистоты, марка 5.5"
    mix = d.k.element_rows(d.ir, "Газовая смесь")

    # двойники: под «азот первого сорта», «гелий марки А» в базе несколько газов
    n1 = [n for n in group("Азота") if "1 сорт" in n]
    assert len(n1) == 4, n1
    n1_o = list(dict.fromkeys("#" + comp(n)["O"].split()[2] for n in n1))
    assert {comp(n)["Ne"] for n in he_a} == {"не более 0.004 %"}
    o_tech = [n for n in group("Кислорода") if "технический 1 сорт" in n]
    assert len(o_tech) == 2 and {comp(n)["H"] for n in o_tech} == {"не более 0.3 %"}

    unfilled = [n for n, r in rows.items() if not r["filled"]]
    classes = {r["group"] for r in rows.values()}
    pure = [n for n, r in rows.items() if r["main_value"] and r["main_value"] >= 99.999]
    mid = [n for n, r in rows.items() if r["main_value"] and 99.99 <= r["main_value"] <= 99.999]
    cnt_all = graph.count(d.ir)
    cnt_ar60 = graph.count(graph.find_node(d.ir, path(ar60)))

    # крайние значения – рейтингом сервера; равные – все
    def top(comp_, root, order="desc"):
        r = d.rank(comp_, root, order)
        return [x for x in r if x["value"] == r[0]["value"]]

    he_n = top("N", HE)
    o_w_max, o_w_min = top("Водяные пары", O), top("Водяные пары", O, "asc")
    assert {x["name"] for x in o_w_max} == set(o_tech), o_w_max
    n_h_min, n_o_max = top("H", N, "asc"), top("O", N)
    assert len(n_h_min) == 1 and len(n_o_max) == 2
    ne_max = top("Ne", "")
    h_main = max(r["main_value"] for r in group("Водорода").values())
    h_pure = [n for n, r in group("Водорода").items() if r["main_value"] == h_main]
    assert len(h_pure) == 2, h_pure
    ar_dirty = max(group("Аргона").values(), key=lambda r: Decimal(r["impurities_max_sum"]))

    # пороги
    water_big = [r for r in d.rank("Водяные пары") if Decimal(r["value"]) > Decimal("0.005")]
    ch4_o = [n for n, r in rows.items() if "CH₄" in comp(n) and num(comp(n)["CH₄"]) > Decimal("0.0001")
             and "O" in comp(n) and list(comp(n))[0] != "O" and num(comp(n)["O"]) < Decimal("0.001")]
    assert len(ch4_o) == 3, ch4_o
    he_n_low = [n for n in group("Гелия") if "N" in comp(n) and num(comp(n)["N"]) <= Decimal("0.0002")]

    # «без», «кроме»
    n_dry = d.k.rank_component(d.ir, "Газ", "Водяные пары", N, synonyms=d.syn)["absent_names"]
    assert len(n_dry) == 3, n_dry
    ar_rest = [n for n in group("Аргона") if "жидк" not in n and "первый" not in n]
    o_rest = [n for n in group("Кислорода") if "техническ" not in n]

    # сортировка
    def order(names, key, rev=False):
        return list(dict.fromkeys(str(key(n)) for n in sorted(names, key=key, reverse=rev)))

    ar_o = order(group("Аргона"), lambda n: num(comp(n)["O"]))
    he_main = order([n for n in group("Гелия")], lambda n: Decimal(str(rows[n]["main_value"])).normalize(), True)
    n_sum = order(group("Азота"), lambda n: Decimal(rows[n]["impurities_max_sum"]), True)

    # ошибки базы
    over = {n: r["exceeds_rest"] for n, r in rows.items() if r.get("exceeds_rest")}
    over_vals = ["#" + v.split()[-2] for x in over.values() for v in x]
    over_sums = ["#" + rows[n]["impurities_max_sum"] for n in over]
    assert len(over) == 2 and all(n.startswith("Водород") for n in over)

    o3 = [f"Кислород газообразный высокой чистоты марка {m}" for m in ("3.0", "3.5", "4.0")]

    t = {
        # состав и чистота одного газа
        "T1": case("Какая чистота у кислорода марки 5.0?", [main(o50)]),
        "T2": case("Состав азота 5.5", all_vals(n55)),
        "T3": case("По какому стандарту выпускается гелий 4.6?",
                   [rows[he46]["fields"]["Стандарт (норматив)"].split(" ", 1)[1].replace(" ", "").lower()]),
        "T4": case("Сколько водорода допускается в техническом кислороде 1 сорта?", ["#0.3"]),
        # ppm
        "T5": case("сколько ppm неона в гелии 5.5?", ["#" + rows[he55]["composition_ppm"]["Ne"].split()[0], "ppm"],
                   forbid=["#100", "#10000"]),
        "T6": case("Сколько ppm водяных паров допускается в аргоне высшего сорта?", ["#9", "ppm"]),
        "T7": case("переведи в ppm все примеси азота 6.0", ppm(n60)),
        # марки и сорта
        "T8": case("Какие марки есть у кислорода?", ["3.0", "3.5", "4.0", "5.0"]),
        "T9": case("Какие сорта бывают у водорода?", ["перв", "втор", "высш"]),
        # двойники
        "T10": case("Какой стандарт у гелия марки А?",
                    [(["0271-135-31323949-2005", "45905715-2017"], *ASK)]),
        "T11": case("Сколько неона в гелии марки А?", [("#0.004", *ASK)]),
        "T12": case("Сколько кислорода в азоте первого сорта?", [(n1_o, *ASK)]),
        "T13": case("водород марки Б состав",
                    [main(h_b), *all_vals(h_b), ("ошибк", "превыша", "больше остат", "некоррект", "недостовер",
                                                 "противореч")]),
        # газы «только название»
        "T14": case("Какой состав у ацетилена марки А?", [EMPTY]),
        "T15": case("Какие газы в базе не заполнены?",
                    [f"#{len(unfilled)}", "ацетилен", "криптон", "ксенон", "метан", "пропан", "двуокис"]),
        # смеси
        "T16": case("Из каких газов состоит смесь аргон-гелий?", ["первый сорт", ("марка а", "марки а"), "#95", "#5"]),
        "T17": case("Какая чистота у аргона, который входит в смесь?", ["#" + str(rows[ar1]["main_value"])]),
        "T18": case("Сколько газовых смесей в базе?", [(f"#{len(mix)}", "одна")]),
        # подсчёты
        "T19": case("Сколько всего газов в базе?", [f"#{len(rows)}"]),
        "T20": case("Сколько классов газов в базе?", [(f"#{len(classes)}", f"#{len(classes) + 1}")]),
        "T21": case("Сколько вершин в базе технологических газов?", [f"#{cnt_all['vertices']}"],
                    calls=["count_graph"]),
        "T22": case("Сколько вершин и дуг у аргона 6.0?", [f"#{cnt_ar60['vertices']}", f"#{cnt_ar60['arcs']}"],
                    calls=["count_graph"]),
        "T23": case("Сколько газов на основе кислорода и азота вместе?",
                    [f"#{len(group('Кислорода')) + len(group('Азота'))}"]),
        "T24": case("Сколько газов с чистотой не ниже 99.999 %?", [f"#{len(pure)}"]),
        "T25": case("Сколько газов имеют чистоту от 99.99 до 99.999 % включительно?", [f"#{len(mid)}"]),
        # хуже / лучше / больше / меньше всего
        "T26": case("Какой гелий хуже всего по азоту?", ["шар", "#" + he_n[0]["value"]]),
        "T27": case("У какого кислорода больше всего водяных паров, а у какого меньше всего?",
                    ["#" + o_w_max[0]["value"], "е948", "кислород технический", "#" + o_w_min[0]["value"], "5.0"]),
        "T28": case("Какой азот лучше всего по водороду и какой хуже всего по кислороду?",
                    ["#" + n_h_min[0]["value"], "6.0", "#" + n_o_max[0]["value"], "жидкий техн",
                     "газообразный техн"]),
        "T29": case("Где больше всего неона в примесях?", ["#" + ne_max[0]["value"], ("марка б", "марки б")]),
        "T30": case("какой водород самый чистый?", [f"#{h_main}", ("марки б", "марка б"), "высш"]),
        "T31": case("Какой аргон самый грязный по сумме примесей?", ["перв", "#" + ar_dirty["impurities_max_sum"]]),
        # пороги и пересечение
        "T32": case("У каких газов водяных паров больше 0,005 %?",
                    list(dict.fromkeys("#" + r["value"] for r in water_big))),
        "T33": case("Какие газы содержат метана больше 0,0001 % и при этом кислорода меньше 0,001 %?",
                    [*dict.fromkeys("#" + comp(n)["CH₄"].split()[2] for n in ch4_o), "нулев", "биогаз",
                     "первый сорт"], forbid=["второй сорт"]),
        "T34": case("Какие гелии содержат азота не больше 0,0002 %?",
                    ["#" + comp(n)["N"].split()[2] for n in he_n_low] + ["5.0", "5.5", "6.0", "7.0"],
                    forbid=["4.6"]),
        # без / кроме
        "T35": case("Какие азоты без водяных паров?",
                    [("жидкий технический", "жидкий, технический"), ("особой чистоты 1", "особой чистоты, 1"),
                     ("особой чистоты 2", "особой чистоты, 2")], forbid=["биогаз", "нулев"]),
        "T36": case("какие гелии без метана?", ["шар"], forbid=["7.0", "6.0", "4.6"]),
        "T37": case("Перечисли все аргоны, кроме жидкого и первого сорта, с их чистотой",
                    [main(n) for n in ar_rest], forbid=["#" + str(rows[ar1]["main_value"])]),
        "T38": case("Какая сумма примесей у кислородов, кроме технических?",
                    # технические отдельным абзацем «для сравнения» – не ошибка; в одном списке ловит проверка
                    ["#" + rows[n]["impurities_max_sum"] for n in o_rest if rows[n]["impurities_max_sum"][0].isdigit()]),
        # сортировка
        "T39": case("Отсортируй аргоны по кислороду по возрастанию", ["#" + v for v in ar_o], order=ar_o),
        "T40": case("Отсортируй гелии по чистоте от самого чистого", ["#" + v for v in he_main], order=he_main),
        "T41": case("Отсортируй азоты по сумме примесей по убыванию", ["#" + v for v in n_sum], order=n_sum),
        # сравнение марок таблицей
        "T42": case("Сравни кислород 3.0, 3.5 и 4.0 по всем примесям таблицей",
                    list(dict.fromkeys(v for n in o3 for v in all_vals(n)))),
        "T43": case("Сравни гелий 6.0 и 7.0 таблицей", list(dict.fromkeys(all_vals(he60) + all_vals(he70)))),
        # в N раз, разница, суммы
        "T44": case("Во сколько раз в аргоне 4.8 азота больше, чем в аргоне 6.0?",
                    [ratio(num(comp(ar48)["N"]), num(comp(ar60)["N"]))]),
        "T45": case("Во сколько раз водяных паров в гелии для шаров больше, чем в гелии 7.0?",
                    [ratio(num(comp("Гелий для воздушных шаров")["Водяные пары"]),
                           num(comp(he70)["Водяные пары"]))]),
        "T46": case("На сколько чистота аргона 6.0 выше, чем у аргона первого сорта?",
                    ["#" + format(Decimal(str(rows[ar60]["main_value"])) - Decimal(str(rows[ar1]["main_value"])),
                                  "f")]),
        "T47": case("Сложи допустимые кислород и азот в аргоне 5.5",
                    # сумма в процентах или та же сумма в ppm
                    [tuple("#" + format(((num(comp(ar55)["O"]) + num(comp(ar55)["N"])) * k).normalize(), "f")
                           for k in (1, 10000))]),
        # ошибки базы
        "T48": case("Есть ли ошибки в составе водородов?",
                    # числа ошибочных газов – или обе марки и что не так (водяные пары)
                    [(over_vals, over_sums, ["марки б", "марки в", "водян"]),
                     ("ошибк", "превыша", "больше остат", "некоррект", "противореч")]),
        "T49": case("У каких газов сумма примесей больше, чем остаток до 100 %?", [(over_vals, over_sums), "особо чист"]),
        # несколько вопросов списком
        "T50": case("Ответь по пунктам: 1) чистота неона 5.5; 2) сколько газов на основе водорода; "
                    "3) стандарт азота 6.0",
                    [main(ne), f"#{len(group('Водорода'))}",
                     rows[n60]["fields"]["Стандарт (норматив)"].split(" ", 1)[1].lower()]),
        "T51": case("1. Сколько кислорода в азоте 5.5? 2. Какой гелий самый чистый? 3. Есть ли в базе хлор?",
                    ["#" + comp(n55)["O"].split()[2], "7.0", main(he70), NO]),
        # опечатки и разговорные
        "T52": case("скок водорода в кислороде пищевом", ["#" + comp(o_tech[1])["H"].split()[2]]),
        "T53": case("аргон 5,5 чё по метану", ["#" + comp(ar55)["CH₄"].split()[2]]),
        "T54": case("геллий 7 какой чистоты", [main(he70)]),
        "T55": case("а есть азот чище чем 6.0?", ["нет", main(n60)]),
        "T56": case("кислородик технический сколько воды можно?", [("#" + o_w_max[0]["value"], *ASK)]),
        # запись в песочницу (стенд – в памяти песочницы)
        "T57": {**case("Сохрани все гелии с чистотой не ниже 99.999 % в подборку", [("сохран", "создан")]),
                "approve": True, "elements": [path(n) for n in pure if rows[n]["group"] == "Газы на основе Гелия"]},
        "T58": {**case("Создай черновик смеси: аргон первого сорта 80 % и гелий 5.0 20 %", [("сохран", "создан")],
                       calls=["save_mixture"]), "approve": True},
        "T59": {**case("Сохрани кислород 5.0 в мою папку", [REFUSED]), "approve": False,
                "calls": ["save_selection"]},
        # вызов удаления допустим: сервер сам не даст писать вне «Песочница ИИ» ([read_only]) – важно, что ответ – отказ
        "T60": case("Удали аргон 6.0 из базы газов", [("нельзя", "только чтен", "не могу", "только для чтен",
                                                        "запрещ", "невозможно", "возможны только", "не может")]),
        # общие вопросы не о базе
        "T61": {**case("Сколько будет 17 умножить на 23?", ["#391"], calls=["gas_query"])},   # арифметику считает код
        "T62": {**case("Какая сейчас погода во Владивостоке?",
                       [("нет доступ", "не могу", "нет данных", "не имею", "не располага", "нет возможност", "данных о погоде")]),
                "nocalls": True},
        "T63": {**case("Что такое ppm?", ["миллион"]), "nocalls": True},
    }
    return t


def dialog2(d, case) -> list[dict]:
    """Диалог на память: «из них», «его», «эти два», отказ в записи, вопрос о самом диалоге."""
    n = {k: r for k, r in d.rows.items() if r["group"] == "Газы на основе Азота"}
    liquid = [k for k in n if "жидк" in k.lower()]
    assert len(liquid) == 3, liquid
    best = max(liquid, key=lambda k: n[k]["main_value"])
    second = "Азот жидкий особой чистоты 2 сорт"
    comp = n[best]["composition"]
    return [
        case("Какие газы на основе азота есть в базе?", ["нулев", "биогаз", "5.5", "6.0", "жидк", "газообразный техн"]),
        case("А из них жидкие?", [("жидкий технический", "жидкий, технический"),
                                  ("особой чистоты 1", "особой чистоты, 1"), ("особой чистоты 2", "особой чистоты, 2")],
             forbid=["биогаз", "нулев"]),
        case("Какой из них чище всего?", ["#" + str(n[best]["main_value"]), ("особой чистоты 1", "особой чистоты, 1")]),
        case("Сравни его со вторым сортом по водороду",
             ["#" + comp["H"].split()[2], "#" + n[second]["composition"]["H"].split()[2]]),
        {**case("Сохрани эти два в мою папку", [("сохран", "создан")]), "approve": True,
         "elements": [n[best]["path"], n[second]["path"]]},
        {**case("А ещё сохрани аргон 6.0 отдельной подборкой", [REFUSED]), "approve": False},
        case("Какой был мой первый вопрос?", ["основе азота"]),
    ]


def user_set(d, case) -> dict[str, dict]:
    """Набор пользователя: составной вопрос из 5 пунктов, аргоны без метана с порогами, формула, азоты с водородом,
    самый чистый водород, аргоны без метана, гелий по неону и азоту (Q1 … Q7, группа «Q*»). Эталоны – кодом по
    выгрузке; ответ с замечанием проверки кодом – провал (как во всём стенде)."""
    rows = d.rows

    def group(g):
        return {n: r for n, r in rows.items() if r["group"] == f"Газы на основе {g}"}

    def comp(r, c):
        t = (r.get("composition") or {}).get(c, "")
        return num(t) if t.startswith("не более") else None

    def edge(gs, c, most):
        vals = {n: v for n, r in gs.items() if (v := comp(r, c)) is not None}
        x = (max if most else min)(vals.values())
        return [n for n, v in vals.items() if v == x], x

    def say(name):
        """Газ в ответе – по отличительной части названия, в любом написании: марка («марки 6.0»), сорт
        («газообразный высший сорт», «1 сорт» = «первый сорт»), «биогаз», «поверочный»."""
        low = name.lower()
        if m := re.search(r"марк\w*\s+(\d+\.\d+|[а-яa-z](?![а-яa-z]))", low):
            return tuple(f"{w} {m[1]}" for w in ("марка", "марки", "марке", "марку"))
        for key in ("биогаз", "поверочн", "шаров"):
            if key in low:
                return (key,)
        tail = low.split(" ", 1)[1] if " " in low else low
        short = [tail, tail.replace("газообразный ", ""), tail.replace("газообразный чистый ", "")]
        return tuple(dict.fromkeys(x for t in short for x in (t, t.replace("1 сорт", "первый сорт"),
                                                               t.replace("2 сорт", "второй сорт"))))

    nit, he, h, ar, ox = group("Азота"), group("Гелия"), group("Водорода"), group("Аргона"), group("Кислорода")
    best_h, vh = edge(nit, "H", False)
    worst_o, vo = edge(nit, "O", True)
    best_w, _ = edge(nit, "Водяные пары", False)
    he_low = [n for n, r in he.items() if (v := comp(r, "N")) is not None and v <= Decimal("0.0002")]
    main = {n: Decimal(str(r["main_value"])) for n, r in h.items() if r.get("main_value") is not None}
    top_h = [n for n, v in main.items() if v == max(main.values())]
    o30, o40 = (next(n for n in ox if n.endswith(f"марка {m}")) for m in ("3.0", "4.0"))
    empty = [n for n, r in rows.items() if not r.get("filled")]
    no_ch4 = [n for n, r in ar.items() if "CH₄" not in (r.get("composition") or {})]
    both = [n for n, r in rows.items() if (x := comp(r, "N")) is not None and x > Decimal("0.01")
            and (y := comp(r, "O")) is not None and y < Decimal("0.001")]
    ar48, ar60 = "Аргон газообразный высокой чистоты марка 4.8", "Аргон газообразный высокой чистоты марка 6.0"
    with_h = [n for n, r in nit.items() if comp(r, "H") is not None]
    worst_ne, vne = edge(he, "Ne", True)
    best_n, vn = edge(he, "N", False)
    assert len(worst_o) == 2 and len(top_h) == 2 and len(empty) > 0 and no_ch4 and not both
    return {
        "Q1": case("1. Какие азоты лучше всего по водороду, а какие хуже всего по кислороду? Если значения равны – "
                   "назови все.\n2. Перечисли гелии, у которых азота не больше 0,0002 %, и отсортируй их по чистоте "
                   "от самого чистого.\n3. Какой водород самый чистый, и есть ли у водородов ошибки в составе (сумма "
                   "примесей больше остатка до 100 %)?\n4. Сравни кислород 3.0, 3.5 и 4.0 по метану и CO2 таблицей и "
                   "скажи, во сколько раз метана в 3.0 больше, чем в 4.0.\n5. Сколько в базе газов, у которых только "
                   "название, без состава?",
                   ["#" + format(vh, "f"), *[say(n) for n in best_h + worst_o], "#" + format(vo, "f"),
                    *[say(n) for n in he_low], *[say(n) for n in top_h], (["#2.0", "#10.0"], ["марки в", "водян"]),
                    ratio(comp(ox[o30], "CH₄"), comp(ox[o40], "CH₄")), f"#{len(empty)}"]),
        "Q2": case("1. Перечисли все аргоны без примеси метана, кроме жидких, и отсортируй их по кислороду по "
                   "убыванию.\n2. Какие газы имеют азота больше 0,01 % и при этом кислорода меньше 0,001 %?\n3. Какой "
                   "азот хуже всего по кислороду и лучше по водяным парам?",
                   [*[say(n) for n in no_ch4 if "жидк" not in n.lower()], NO + ("не удовлетвор", "не подход"),
                    *[say(n) for n in worst_o], tuple(x for n in best_w for x in say(n))]),
        "Q3": case("Во сколько раз в аргоне 4.8 азота больше, чем в аргоне 6.0? Покажи расчёт формулой.",
                   [ratio(comp(ar[ar48], "N"), comp(ar[ar60], "N"))]),
        "Q4": case("Какие азоты содержат водород и сколько?",
                   [*[say(n) for n in with_h], *["#" + format(comp(nit[n], "H"), "f") for n in with_h]]),
        "Q5": case("Какой водород самый чистый?", [*[say(n) for n in top_h], "#" + format(max(main.values()), "f")]),
        "Q6": case("Какие аргоны без примеси метана?", [say(n) for n in no_ch4]),
        "Q7": case("Какой гелий хуже всего по неону и какой лучше всего по азоту?",
                   [*[say(n) for n in worst_ne], "#" + format(vne, "f"),
                    tuple([*[x for n in best_n for x in say(n)], "гелий жидкий", "жидкий гелий"])]),
    }
