"""Стенд точности: «неудобные» вопросы и диалог на память – настоящая модель, оценка – кодом.

Данные – полная выгрузка базы газов (48 газов, 1478 вершин) в песочнице (IACPAAS_BACKEND=mock): ответы те же, что на
платформе, а платформа не нужна и не трогается; запись – только в памяти песочницы. Эталоны считаются кодом сервера
по той же выгрузке (числа – не руками). Каждый одиночный вопрос – отдельная сессия; диалог D – одна сессия с
памятью 10, «сжать весь чат» и правкой вопроса.
    .venv\\Scripts\\python scripts\\agent_bench.py --base <папка выгрузки> --model gigachat            всё
    .venv\\Scripts\\python scripts\\agent_bench.py --base <папка> --model gigachat S3 S7 D           выбранное
<папка выгрузки> – где лежат «Загрузки/База технологических газов.json» и онтологии (как fixtures/).
Результат: logs/agent_bench_<модель>.jsonl и итог в консоли: ВЕРНО / ОШИБКА – чего не хватает, что лишнее,
что сказала проверка ответа кодом. История – data/bench.sqlite (диалоги пользователя не засоряются)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AR = "Моногазы/Газы на основе Аргона"
H = "Моногазы/Газы на основе Водорода"


def norm(text: str) -> str:
    """Ответ для сравнения: без разметки, «0,002» = «0.002», «ё» = «е», один пробел."""
    t = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)          # номера пунктов «2.» – не числа ответа
    t = t.replace("{,}", ",")                             # LaTeX «$0{,}0002\,\%$» – десятичная запятая
    t = re.sub(r"[*_`|$\\]+", " ", t.lower().replace("ё", "е").replace("\u00a0", " "))
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    return re.sub(r"\s+", " ", t)


def has_num(t: str, v: str) -> bool:
    """Число как отдельное: «0.0002» не находится внутри «0.00002»."""
    d = Decimal(v)
    return any(Decimal(m) == d for m in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])", t))


class Data:
    """Эталоны – кодом сервера по той же выгрузке."""

    def __init__(self, base: Path):
        from iacpaas_mcp import knowledge
        from iacpaas_mcp.config import load_synonyms
        self.k, self.syn = knowledge, load_synonyms()
        self.ir = json.loads((base / "Загрузки" / "База технологических газов.json").read_text(encoding="utf-8"))
        self.rows = {r["name"]: r for r in knowledge.element_rows(self.ir, "Газ", with_composition=True)}

    def rank(self, comp, root="", order="desc"):
        return self.k.rank_component(self.ir, "Газ", comp, root, order, synonyms=self.syn)["rows"]

    def val(self, gas, comp):
        return self.rows[gas]["composition"][comp].split()[2]

    def paths(self, value) -> list[str]:
        """Газы из аргумента elements (строка «газ 1; газ 2» или список) – путями базы, как их находит сервер."""
        r = self.k.find_elements(self.ir, self.k.split_items(value), ("Газ", "Газовая смесь"), groups=True,
                                 synonyms=self.syn)
        return r["paths"] + [f"? {q}" for q in [*r["ambiguous"], *r["missing"]]]


DATA: Data | None = None                                   # для оценки: те же пути, что найдёт сервер


# Вопрос: текст и проверка ответа (список замечаний; пусто – верно). need – все варианты должны быть (каждый пункт –
# строка, число "#…" или кортеж «любой из»), forbid – не должно быть.
def case(q, need=(), forbid=(), order=(), calls=()):
    return {"q": q, "need": list(need), "forbid": list(forbid), "order": list(order), "calls": list(calls)}


def singles(d: Data) -> dict[str, dict]:
    o = d.rank("O")
    h_big = [r for r in d.rank("H") if Decimal(r["value"]) > Decimal("0.1")]
    water = d.rank("водяные пары", H, "asc")
    co = d.rank("CO")
    co2 = d.rank("CO2")
    sums = sorted((r for r in d.rows.values() if r.get("impurities_max_sum", "")[:1].isdigit()
                   and not r.get("exceeds_rest")), key=lambda r: Decimal(r["impurities_max_sum"]), reverse=True)
    he = [n for n, r in d.rows.items() if r["group"] == "Газы на основе Гелия"]
    h_top = "Водород газообразный чистый высший сорт"
    # новые: эталоны – кодом по той же выгрузке
    ar = {n: r for n, r in d.rows.items() if r["group"] == "Газы на основе Аргона"}
    no_ch4 = [n for n, r in ar.items() if "CH₄" not in r["composition"]]
    he_rest = [r for n, r in d.rows.items() if r["group"] == "Газы на основе Гелия" and n not in
               ("Гелий жидкий", "Гелий для воздушных шаров")]
    he_main = list(dict.fromkeys(r["composition"]["He"].split()[2] for r in he_rest))
    n_big = [r for r in d.rank("N") if Decimal(r["value"]) > Decimal("0.01")]
    ch4_he = d.rank("CH4", "Моногазы/Газы на основе Гелия", "asc")[0]
    he6 = "Гелий газообразный (сжатый) высокой чистоты, марка 6.0"
    water6 = str((Decimal(d.val(he6, "Водяные пары")) * 10000).normalize())
    a48 = "Аргон газообразный высокой чистоты марка 4.8"
    return {
        "S1": case("У какого газа в примесях больше всего кислорода?",
                   ["#" + o[0]["value"], "азот", "техн"]),
        "S2": case("Где меньше всего метана среди аргонов?",
                   ["6.0", "#" + d.rank("CH4", AR, "asc")[0]["value"]]),
        "S3": case("Есть газы, где водорода в примесях больше 0,1 %?",
                   ["#" + h_big[0]["value"], "кислород", ("е948", "пищев")], calls=["gas_query"]),
        "S4": case("сравни гелий 5.0 и 7.0 по неону",
                   ["#" + d.val("Гелий газообразный (сжатый) высокой чистоты, марка 5.0", "Ne"),
                    "#" + d.val("Гелий газообразный высокой чистоты (сжатый, марка 7.0)", "Ne")]),
        "S5": case("сколько ppm кислорода можно в азоте 6.0?",
                   [("#0.65", "#" + d.val("Азот высокой чистоты (марка 6.0)", "O"))], forbid=["#65"]),
        "S6": case("отсортируй водороды по водяным парам, от меньшего",
                   [("ошибк", "больше остатка", "превыша", "в сотни раз", "недостовер")],
                   order=[r["value"] for r in water]),
        "S7": case("какой аргон хуже всего по азоту?",
                   [("первый сорт", "первого сорта", "первом сорте"), "#" + d.rank("N", AR)[0]["value"]],
                   forbid=["марка 6.0 – хуже"]),
        "S8": case("в каких газах есть угарный газ?", ["#" + r["value"] for r in co]),
        "S9": case("Чем отличается гелий А от гелия Б?", ["#99.99", "#99.995"]),
        "S10": case("сколько стоит баллон аргона?", [("нет", "не содерж", "отсутств")], forbid=["руб"]),
        "S11": case("какой газ чище: криптон 5.5 или ксенон 5.5?",
                    [("не заполн", "только назван", "нет данных", "не указан", "отсутств", "нет подробн",
                      "нет информац", "нет сведен")], forbid=["#99.999"]),
        "S12": case("Суммарно сколько примесей допускается в кислороде марки 4.0?",
                    ["#" + d.rows["Кислород газообразный высокой чистоты марка 4.0"]["impurities_max_sum"]]),
        "S13": case("дай топ-3 самых грязных газа",
                    [("#" + sums[0]["impurities_max_sum"], "#99.6")]),
        "S14": case("сколько гелиев в базе?", [f"#{len(he)}"]),
        "S15": case("какие марки бывают у азота?", ["5.5", "6.0"]),
        "S16": case("Аргон 6.0 – это сколько девяток?", ["#99.9999"]),
        "S17": case("у какого газа примесей больше чем самого газа?",
                    ["водород", ("ошибк", "ввод", "#2.0", "#10.0", "#10")]),
        "S18": case("Какие газы содержат ксенон?", ["кислород", ("сумм", "+")]),
        "S19": case("водорд высшего сорта состав",
                    ["#99.9999", "#" + d.val(h_top, "O"), "#" + d.val(h_top, "N"), "#" + d.val(h_top, "Водяные пары")]),
        "S20": case("Сколько всего примесей в смеси аргон-гелий?",
                    [("нет", "не вычисл", "не указ", "не хран", "отсутств", "не представлен")], forbid=["#0.0192"]),
        "S21": case("Какая доля аргона в смеси?", ["#95"]),
        "S22": case("Перечисли все газы с примесью CO2", ["#" + v for v in dict.fromkeys(r["value"] for r in co2)]),
        "S24": case("Есть ли неон марки 7.0?", [("нет", "не найден", "отсутств"), "5.5"], forbid=["#99.99999"]),
        "S25": case("Какие аргоны без примеси метана?",
                    [w for n in no_ch4 for w in (("перв",) if "перв" in n else ("жидк",) if "жидк" in n else ())]),
        "S26": case("Какая чистота у всех гелиев, кроме жидкого и для шаров?", ["#" + v for v in he_main],
                    forbid=["#99.8"]),
        "S27": case("сколько ppm влаги в гелии 6.0", ["#" + water6, "ppm"], forbid=["#20", "#200"]),
        "S28": case("сравни аргон 4.8, 5.5 и 6.0 по кислороду",
                    ["#" + d.val(f"Аргон газообразный высокой чистоты марка {m}", "O") for m in ("4.8", "5.5", "6.0")]),
        # «гелий марки А» – в базе два газа: назвать оба (у одного O отдельно, у другого только «O + Ar»)
        "S29": case("Сколько кислорода в гелии марки А?",
                    # верно и число по обоим, и переспрос «какой из них» с обоими названиями
                    [("сжат", "два", "оба", "2 газа"), ("#" + d.val("Гелий газообразный (сжатый, марка А)", "O"),
                                                         "какой из них", "какой именно", "какой нужен")]),
        "S30": case("У каких газов азота в примесях больше 0,01 %?",
                    [w for r in n_big for w in ("#" + r["value"], r["name"].split()[-1].lower())]),
        "S31": case("Какой гелий лучше всего по метану?", ["7.0", "#" + ch4_he["value"]]),
        "S32": case("Есть ли в базе фтор?", [("нет", "не найден", "отсутств", "не содерж")]),
        "S33": case("Чем криптон отличается от ксенона по составу?",
                    [("не заполн", "только назван", "нет данных", "не указан", "отсутств", "нет информац",
                      "нет сведен")], forbid=["#99.999", "#99.9995"]),
        "S34": case("состав аргона 4,8", ["#" + d.val(a48, c) for c in ("O", "N", "CH₄")]),
        # исходные вопросы 21.10 дословно (с опечатками), на которых GigaChat выдумывал аргон у гелиев и азоты
        "S35": case("выдай три газа с наибольш примесями аргона", ["#" + r["value"] for r in d.rank("Ar")[:3]],
                    forbid=["гелий газообразный (сжатый, марка а) – ar", "технической очистки"]),
        "S36": case("выдай газы с наиб примесеями аргона - отсортируй по увелечение примесей метана",
                    ["#" + r["value"] for r in d.rank("Ar")],
                    order=[r["value"] for r in sorted(d.rank("Ar"), key=lambda r: Decimal(
                        d.rows[r["name"]]["composition"]["CH₄"].split()[2]))]),
        # запись в песочнице стенда: два газа в разные подборки – по одному газу в каждой
        "S23": {**case("Сохрани аргон 4.8 и аргон 6.0 в мою папку в разные подборки", [("сохран", "создан")]),
                "approve": True, "separate": [AR + "/Аргон газообразный высокой чистоты марка 4.8",
                                              AR + "/Аргон газообразный высокой чистоты марка 6.0"]},
    }


def dialog(d: Data) -> list[dict]:
    a48 = AR + "/Аргон газообразный высокой чистоты марка 4.8"
    a60 = AR + "/Аргон газообразный высокой чистоты марка 6.0"
    r48 = d.rows["Аргон газообразный высокой чистоты марка 4.8"]["composition"]
    return [
        case("Какие газы на основе аргона есть в базе?", ["высш", "перв", "4.8", "5.5", "6.0", "жидк"]),
        case("А какой из них самый чистый?", ["6.0", "#99.9999"]),
        case("Сколько в нём метана?", ["#" + d.val("Аргон газообразный высокой чистоты марка 6.0", "CH₄")]),
        case("Сравни его с маркой 4.8 по всем примесям таблицей",
             ["#" + r48[c].split()[2] for c in ("O", "N", "CO₂", "CH₄", "H")]),
        {**case("Сохрани эти два газа в мою папку", [("сохран", "создан")]), "approve": True,
         "elements": [a48, a60]},
        case("Какой был мой второй вопрос?", ["самый чистый"]),
        case("Что ты ответил на третий вопрос?", ["#" + d.val("Аргон газообразный высокой чистоты марка 6.0", "CH₄")]),
        {"compact": True},
        case("Напомни, сколько газов на основе аргона?", [("#6", "шест")]),
        case("А какие из них жидкие?", ["жидк", "высш"]),
        case("Какой у него стандарт?", ["10157"]),
        {**case("А какой стандарт у марки 6.0?", ["20.11.11-006"]), "edit": "last"},   # правка последнего вопроса
    ]


def grade(c: dict, answer: str, calls: list[dict]) -> list[str]:
    t = norm(answer)
    bad = []

    def ok(item) -> bool:
        if isinstance(item, tuple):                        # любой из
            return any(ok(x) for x in item)
        if isinstance(item, list):                         # все (вариант внутри «любого из»)
            return all(ok(x) for x in item)
        return has_num(t, item[1:]) if item.startswith("#") else item in t
    for item in c["need"]:
        if not ok(item):
            bad.append("нет: " + (" / ".join(" + ".join(x) if isinstance(x, list) else x for x in item)
                                  if isinstance(item, tuple) else item))
    for item in c["forbid"]:
        if ok(item):
            bad.append("лишнее: " + item)
    if c["order"]:
        pos = []
        for v in c["order"]:
            m = [i.start() for i in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])", t) if Decimal(i.group()) == Decimal(v)]
            pos.append(m[0] if m else -1)
        if -1 in pos or pos != sorted(pos):
            bad.append(f"порядок не тот: {c['order']}")
    for name in c["calls"]:
        if name not in [x["tool"] for x in calls]:
            bad.append(f"не вызван {name}")
    for name in c.get("forbid_calls", ()):
        if name in [x["tool"] for x in calls]:
            bad.append(f"вызван {name}")
    if c.get("nocalls") and calls:                         # общий вопрос – без инструментов
        bad.append("инструменты не нужны: " + ", ".join(x["tool"] for x in calls))
    if c.get("elements"):                                  # газы – названиями или путями: пути находит код сервера
        saved = [x for x in calls if x["tool"] == "save_selection"]
        got = DATA.paths(saved[-1]["args"].get("elements")) if saved else []
        if sorted(got) != sorted(c["elements"]):
            bad.append(f"сохранены не те газы: {got}")
    if c.get("separate"):                                  # каждый газ – в своей подборке
        last = {}
        for x in calls:
            if x["tool"] == "save_selection" and not x["args"].get("confirm_token"):
                last[tuple(DATA.paths(x["args"].get("elements")))] = True
        singles = sorted(e[0] for e in last if len(e) == 1)
        if singles != sorted(c["separate"]):
            bad.append(f"не по одному газу в подборке: {list(last)}")
    return bad


async def run_turn(rt, sid, text, model, keep, edit=None, approve=False):
    t0 = time.perf_counter()
    r = await rt.ask(sid, text, model=model, keep=keep, edit=edit)
    events = list(r.events)
    while r.status == "needs_approval":
        r = await rt.decide(sid, approve, model=model, keep=keep)
        events += r.events
    calls = [{"tool": e["tool"], "args": e.get("args") or {}} for e in events if e["event"] == "tool_call"]
    tin = sum((e.get("tokens_in") or 0) - (e.get("tokens_cached") or 0) for e in events if e["event"] == "model_end")
    return r, calls, round(time.perf_counter() - t0, 1), tin


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True, help="папка выгрузки (как fixtures/)")
    ap.add_argument("--keep", type=int, default=10, help="память диалога D (сообщений)")
    ap.add_argument("ids", nargs="*", help="S1 … S36, T1 … T63, D, D2; группа – «T*» (по умолчанию всё)")
    a = ap.parse_args()
    base = Path(a.base).resolve()
    os.environ.update({"IACPAAS_BACKEND": "mock", "IACPAAS_FIXTURES": str(base), "IACPAAS_ALLOW_WRITE": "true",
                       "IACPAAS_REGISTRY_FILE": str(ROOT / "config" / "inforesources.json"),
                       "LOG_FILE": str(ROOT / "logs" / "agent_bench_server.jsonl"), "LOG_LEVEL": "WARNING"})
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)            # строки «HTTP/1.1 200 OK» в выводе не нужны
    from iacpaas_agent.core import AgentRuntime
    from iacpaas_agent.settings import AgentSettings
    s = AgentSettings.from_env()
    s.mcp, s.model = "inproc", a.model
    # data/ – общая очередь к модели с веб-чатом и консолью (файл блокировки рядом с БД)
    s.db_file, s.sessions_file = ROOT / "data" / "bench.sqlite", ROOT / "data" / "bench_sessions.sqlite"
    s.trace_file = ROOT / "logs" / "agent_bench_trace.jsonl"
    global DATA
    d = DATA = Data(base)
    import bench_more
    cases = {**singles(d), **bench_more.more(d, case), **bench_more.user_set(d, case)}
    dialogs = {"D": dialog(d), "D2": bench_more.dialog2(d, case)}
    ids = [k for x in a.ids for k in ([*cases, *dialogs] if x == "*" else [k for k in [*cases, *dialogs]
                                      if k.startswith(x[:-1])] if x.endswith("*") else [x])] or [*cases, *dialogs]
    out = ROOT / "logs" / f"agent_bench_{a.model}.jsonl"
    stamp = time.strftime("%m%d-%H%M")
    total, good = 0, 0
    async with AgentRuntime(s) as rt:
        async def one(qid, sid, c, **kw):
            nonlocal total, good
            r, calls, sec, tin = await run_turn(rt, sid, c["q"], a.model, kw.get("keep"), kw.get("edit"),
                                                c.get("approve", False))
            bad = grade(c, r.answer or r.error, calls) if r.status == "done" else [f"статус {r.status}: {r.error}"]
            v = r.verify or {}
            if v.get("problems"):                                # строго: замечание проверки кодом – тоже провал
                bad.append("проверка кодом: " + ("ответ заменён данными из базы – " if v.get("strict") else "")
                           + "; ".join(v["problems"]))
            if v.get("unverified"):                              # запрос не по вопросу – тоже провал
                bad.append("возможно, понял вопрос не так: " + "; ".join(v["unverified"]))
            total += 1
            good += not bad
            mark = "ВЕРНО " if not bad else "ОШИБКА"
            print(f"\n=== {qid} {mark} ({sec} с, ↑{tin}) {c['q']}")
            print("  вызовы:", ", ".join(f"{x['tool']}({json.dumps(x['args'], ensure_ascii=False)[:90]})" for x in calls))
            print("  ответ:", (r.answer or r.error).replace("\n", " ⏎ ")[:900])
            print("  проверка кодом:", "не из базы" if v.get("skipped") else
                  f"✓{v.get('checked', 0)}" + (f" ⚠ {v['problems']}" if v.get("problems") else "")
                  + (" (переписан)" if v.get("retried") else ""))
            for b in bad:
                print("  ✗", b)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run": stamp, "id": qid, "q": c["q"], "ok": not bad, "problems": bad, "sec": sec,
                                    "tokens": tin, "calls": calls, "answer": r.answer, "verify": v,
                                    "status": r.status}, ensure_ascii=False) + "\n")
        for qid in ids:
            if qid in cases:
                await one(qid, f"bench-{qid}-{stamp}", cases[qid])
        for dname in [x for x in ids if x in dialogs]:
            sid = f"bench-{dname}-{stamp}"
            for n, step in enumerate(dialogs[dname], 1):
                if step.get("compact"):
                    res = await rt.compact(sid)
                    print(f"\n=== D{n} сжать весь чат: {res.get('questions')} вопр., ~{res.get('before')} → "
                          f"~{res.get('after')} ток.\n  выжимка: {res.get('digest', res.get('text'))[:700]}")
                    continue
                edit = step.get("edit")
                if edit == "last":
                    edit = sum(h["role"] == "user" and not h.get("steer") for h in await rt.history(sid)) - 1
                await one(f"D{n}", sid, step, keep=a.keep, edit=edit)
            users = [h["text"] for h in await rt.history(sid) if h["role"] == "user"]
            print(f"\n  вопросы диалога после правки ({len(users)}): {users[-3:]}")
    print(f"\nИТОГ: верно {good} из {total}. Журнал – {out}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
