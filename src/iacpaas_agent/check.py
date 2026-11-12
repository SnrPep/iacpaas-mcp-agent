"""Проверка ответа: всё о газах базы пишет код – куски gas_query вставляются по меткам {{#id}}, модель пишет только
связку. Проверяется кодом, без угадывания по тексту ответа:
1) метки – из результатов инструментов, куски подставляет код;
2) в связке нет газов базы и чисел (кроме номеров марок, чисел из вопроса и чисел из ответов инструментов
   платформы – их можно сверить точно);
3) вопрос о газах базы – есть запрос gas_query; без запроса – ответ модели «не из базы», и в нём не должно быть
   фактов базы;
4) запрос соответствует вопросу – короткий список явных признаков («без», «самый», «общие», «во сколько раз»…),
   а не разбор ответа. Как код понял запрос – строка understood у каждого куска, её видит человек."""
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any

MARK = re.compile(r"\{\{\s*(#[0-9a-f]{4})\s*\}\}")
GAS_TOOL = "gas_query"
WRITE_TOOLS = ("save_selection", "save_mixture", "delete_from_fund", "update_concept", "import_inforesource",
               "run_service", "stop_service")
# любое число: «0.0002», «99,999», «57», «1 234» не склеиваем – каждое отдельно
NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?![\w]|[.,]\d)")
# номер пункта: «1. …», «### 2. …», «3) …» в начале строки или после «;»
ITEM = re.compile(r"(?m)(?:^|;)\s*(?:#{1,6}\s*)?(?:\*\*)?\d{1,2}[.)](?=\s|\*)")
UNIT_AFTER = re.compile(r"\s*(?:%|ppm|раз)", re.I)
STD = re.compile(r"(?:ГОСТ(?:\s*Р)?|ТУ)\s*\d[\d\s.\-–]*\d")
# «Понял условие. Готов отвечать на вопросы» – GigaChat вместо ответа на общий вопрос
NON_ANSWER = re.compile(r"понял\s+(?:условие|задачу|инструкц)|готов\s+отвечать|жду\s+(?:ваш|вопрос)", re.I)

# вещества классов базы («какие аргоны», «у какого кислорода») – о газах какого класса вопрос
SUBSTANCES = ("аргон", "азот", "гелий", "водород", "кислород", "неон", "криптон", "ксенон", "метан", "ацетилен",
              "двуокис", "пропан")
# слова вопроса о газах базы
DOMAIN = re.compile(r"\bгаз\w*|примес\w*|чист\w*|\bмарк\w*|\bсорт\w*|стандарт\w*|\bгост\b|\bсмес\w*|\bсостав\w*|"
                    r"\bppm\b|" + "|".join(rf"\b{s}\w*" for s in SUBSTANCES), re.I)
TREE_ASK = re.compile(r"\bдерев", re.I)
# уточнение к прошлому вопросу о газах: «из них», «среди них», «у них», «из этих», «вернёмся к …»
FOLLOW_REF = re.compile(r"из\s+них|среди\s+них|у\s+них|из\s+этих|эт(?:и|их)\s+газ|из\s+перечисленн|верн[её]м|возвраща",
                        re.I)
# отбор газов прошлого запроса: эти ограничения переносятся в запрос уточнения
SCOPE = ("gases", "has", "without", "where", "state", "name_not", "name_has", "grade", "standard")
_PLAIN = {"какой", "какие", "какая", "котор", "всего", "самый", "самые", "самая", "лучше", "больш", "меньш", "верне",
          "вернё", "есть", "базе", "газов", "газы", "основ", "можно", "покаж", "скажи", "давай", "этих", "среди", "сколь"}


def _content(q: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]+", q.lower()) if len(w) >= 4 and w[:5] not in _PLAIN}


def follow_target(question: str, prev: list[tuple[str, dict | None, list[str]]]) -> tuple[str, dict, list[str]] | None:
    """К какому прошлому вопросу уточнение: prev – все прошлые вопросы [(вопрос, аргументы его запроса gas_query или
    None, газы ответа)]. «из них / среди них / у них» – к последнему (он без запроса о газах – ответ по памяти,
    запись, вопрос о диалоге – не угадываем); «вернёмся к аргонам без метана» – к вопросу о газах с теми же словами
    (больше всего совпадений, при равенстве – позднейший); совпадений нет – не угадываем."""
    if not FOLLOW_REF.search(question) or not prev:
        return None
    if not re.search(r"верн[её]м|возвраща", question, re.I):
        return prev[-1] if prev[-1][1] is not None else None
    prev = [p for p in prev if p[1] is not None]
    if not prev:
        return None
    # слова самой ссылки («к аргонам без метана»), не действия: «самый чистый» есть во многих вопросах
    ref = BACK_REF.search(question)
    words = _content(ref[0] if ref else question)
    best = max(range(len(prev)), key=lambda i: (len(words & _content(prev[i][0])), i))
    return prev[best] if words & _content(prev[best][0]) else None


# «вернёмся к аргонам без метана – …»: ссылка на прошлый вопрос до тире / двоеточия / запятой или до слова вопроса
BACK_REF = re.compile(r"(?:давай(?:те)?\s+)?(?:верн[её]мся|возвраща\w*)\s+(?:снова\s+|ещё\s+раз\s+|опять\s+)?к\s+"
                      r".+?(?:\s*[–—:,.;]\s*|(?=\s+(?:как\w*|котор\w*|сколько|что|где|чем|а|и|из\s+них|среди\s+них|"
                      r"у\s+них)\b)|$)", re.I)


def follow_part(question: str) -> str:
    """Вопрос без ссылки «вернёмся к …»: по нему сверяется запрос (отбор газов – от того прошлого вопроса)."""
    return BACK_REF.sub(" ", question).strip() or question


def follow_fix(cur: dict, prev_args: dict) -> dict:
    """Запрос уточнения в рамках газов прошлого вопроса: его отбор (класс, «без», условия…) + своё действие."""
    new = dict(cur)
    for k in SCOPE:
        v = str(prev_args.get(k) or "").strip()
        if not v:
            continue
        if k == "gases":
            new[k] = v
            # класс в path («Моногазы/Газы на основе Аргона») вместе с gases сервер не примет – отбор теперь в gases
            if not str(new.get("path") or "").startswith("Загрузки/"):
                new.pop("path", None)
        elif new.get(k) and k in ("where", "without", "has", "name_not"):
            new[k] = new[k] if v in str(new[k]) else f"{v}; {new[k]}"
        else:
            new.setdefault(k, v)
    return new


# просьба изменить базу: «удали», «переименуй», «измени», «сотри», «замени», «исправь в базе», «пометь»
MODIFY = re.compile(r"\b(?:удал|измен|переимен|сотр|замен|исправь|пометь)\w*", re.I)
GENERAL = re.compile(r"^\s*(?:что\s+такое|что\s+значит|кто\s+такой|расскажи\s+про|объясни,?\s+что)\b", re.I)
# цены, наличие, поставщики, сроки – о газах, но в базе этого нет: отвечает модель («в базе таких данных нет»)
NOT_IN_BASE = re.compile(r"\bстоит\b|\bцен[аыуе]\b|стоимост|купить|продаж|поставщик|наличи|доставк|\bсрок", re.I)
# ошибка состава: «примесей больше, чем самого газа», «сумма примесей больше остатка»
ERRORS = re.compile(r"ошибк|примес\w*\s+больше,?\s+чем\s+(?:сам\w*\s+)?газ|больше,?\s+чем\s+остат|больше\s+остатка",
                    re.I)
# «топ-3», «три газа с наибольшими …» – сколько крайних
TOP_N = re.compile(r"\bтоп[-\s]*(\d+)|\b(\d+|два|две|три|четыре|пять)\s+(?:газ\w*|марк\w*)\s+с\s+(?:наибол|наимен|сам)",
                   re.I)
NUMBERS = {"два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5}
# «суммарно сколько примесей», «сумма примесей», «всего примесей» – готовая сумма примесей газа
# «всего» – только вплотную («всего примесей»): «больше всего неона в примесях» – крайние, а не сумма
SUM_OF = re.compile(r"(?:суммарн\w*|в\s+сумме|сумм\w*)(?:\s+\w+){0,3}?\s+примес|всего\s+примес|"
                    r"примес\w*\s+(?:суммарн|в\s+сумме)", re.I)
# «сколько неона в гелии 5.5», «сколько ppm кислорода в азоте 6.0» – одно значение газа
ASKED_COMP = re.compile(r"сколько\s+(?:ppm\s+|процент\w*\s+)?([а-яё]+(?:\s+пар\w*)?)\s+"
                        r"(?:(?:можно|допускается|допустимо|содержится|есть|бывает)\s+)?(?:в|у)\s", re.I)
ARITH = re.compile(r"\d\s*[-+*/×÷]\s*\d|умнож|подели|раздели|прибав|отним", re.I)

# крайние – превосходная степень: «хуже всего», «самый чистый», «наибольшая»; «чище, чем 6.0» – не крайние
SUPER = re.compile(r"(?:хуже|лучше|больше|меньше|чище|грязнее)\s+(?:всего|всех)|(?:хуже|лучше)\s+по\s|"
                   r"\bсам\w+\s+(?:чист|грязн|больш|мал|высок|низк|плох|хорош)\w*|наибол\w+|наимен\w+|"
                   r"\bлучш(?:ий|ая|ие|ее)\b|\bхудш\w+|\bтоп[-\s]*\d*", re.I)
SORTED = re.compile(r"отсортир|сортир|по\s+возраст|по\s+убыв|упорядоч|от\s+сам\w+", re.I)


# ------------------------------------------------------------------ данные хода
def tool_results(messages: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """[(инструмент, текст результата)] → [(инструмент, JSON-объект)]; не JSON – пропуск."""
    out = []
    for name, text in messages:
        try:
            d = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict):
            out.append((name, d))
    return out


def is_piece(d: dict) -> bool:
    """Результат с готовым куском ответа: gas_query и инструменты платформы (список ИР, подсчёт, схема, дерево…)."""
    return isinstance(d.get("block"), str) and bool(d["block"]) and isinstance(d.get("answer"), str) \
        and bool(d["answer"])


def pieces_of(results: list[tuple[str, dict]]) -> dict[str, dict]:
    """Метка → {answer, understood} из результатов инструментов с кусками (последний с этой меткой)."""
    return {d["block"]: {"answer": d["answer"], "understood": d.get("understood", "")}
            for name, d in results if is_piece(d)}


def _numbers(value: Any) -> set[Decimal]:
    """Все числа во вложенных данных (для чисел платформы: «Всего ИР: 9», «вершин 4536»)."""
    if isinstance(value, dict):
        return set().union(*(_numbers(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_numbers(v) for v in value), set())
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {Decimal(str(value))}
    if isinstance(value, str):
        return {_dec(x) for x in NUM.findall(value)}
    return set()


def _dec(s: str) -> Decimal:
    return Decimal(s.replace(",", ".")).normalize()


# ------------------------------------------------------------------ связка
def glue_problems(glue: str, question: str, gas_names: set[str], grades: set[str],
                  platform_numbers: set[Decimal], wrote: bool = False, only_units: bool = False,
                  seen: str = "", matched: list | None = None, pieces: str = "",
                  general: bool = False) -> list[str]:
    """Что в связке (тексте модели вне кусков) не так: газ базы, число. Связка проверяется там, где модель пишет
    сама: ответ без кусков, ответ о записи. Можно: числа из вопросов пользователя (question – все его вопросы
    диалога), номера марок, числа ответов инструментов платформы (их – в matched), числа и газы прошлых ответов
    (seen). only_units – ответ без инструментов: ошибка только числа с % / ppm."""
    out = []
    text = ITEM.sub(" ", glue)
    # номер стандарта («ТУ 0271-001-45905715-2016») из куска кода – повтор того, что написал код, а не выдумка
    have = {re.sub(r"\D", "", m) for m in STD.findall(pieces)}
    text = STD.sub(lambda m: " " * len(m[0]) if re.sub(r"\D", "", m[0]) in have else m[0], text)
    # название из куска («Аргон-Гелий Ar-95% He-5%») дословно – числа внутри названия не значения
    for name in sorted({a or b for a, b in re.findall(r"\*\*([^*\n]{8,}?)\*\*|«([^»\n]{8,}?)»", pieces)},
                       key=len, reverse=True):
        text = text.replace(name, " " * len(name))
    asked = {_dec(x) for x in NUM.findall(question + " " + seen)}
    bad = []
    for m in NUM.finditer(text):
        raw, v = m[1], _dec(m[1])
        unit = bool(UNIT_AFTER.match(text, m.end()))
        if v in platform_numbers:                     # совпало с ответом инструмента – сверено
            if matched is not None:
                matched.append(raw)
            continue
        if v in asked:
            continue
        # «остаток до 100 %» – целое, а не значение базы
        if v == 100 and unit and re.search(r"\bдо\s*$", text[:m.start()]):
            continue
        if not unit and (only_units or raw.replace(",", ".") in grades):   # «аргон 6.0», «1 сорт» – номер марки
            continue
        bad.append(raw)
    # общий вопрос («что такое ppm» – «1 ppm = 0.0001 %») без запросов к базе: числа – знания, а не данные базы
    if bad and not general:
        out.append(f"в тексте вне кусков числа {', '.join(dict.fromkeys(bad))} – числа пишет код: вставь метку "
                   "куска gas_query (расчёт – op=calc), сам чисел не пиши")
    low, before = glue.lower(), seen.lower()
    # название, которое человек сам написал в вопросе, повторить можно – это не выдумка
    asked_low = question.lower()
    if not wrote and (named := sorted(n for n in gas_names if len(n) > 12 and n.lower() in low
                                      and n.lower() not in before and n.lower() not in asked_low)):
        out.append("в тексте вне кусков названы газы " + "; ".join(f"«{n}»" for n in named[:4])
                   + " – газы называет код в куске: вставь метку, сам газы не перечисляй")
    return out


# ------------------------------------------------------------------ запрос ↔ вопрос
def subjects(question: str) -> list[str]:
    """О газах какого вещества вопрос: «какой азот», «какие аргоны», «среди гелиев», «у какого кислорода»,
    «вернёмся к кислородам»."""
    out = []
    for m in re.finditer(r"\b(?:как\w+|среди|все|у\s+как\w+|есть(?:\s+ли)?|(?:верн[её]мся|возвраща\w*)\s+к)\s+"
                         r"(?=([а-яё]+))", question.lower()):
        w = m[1]
        for s in SUBSTANCES:
            if w[:4] == s[:4] and not w.endswith(("у", "ом")) and s not in out:
                out.append(s)
    return out


def _thresholds(question: str) -> list[str]:
    """Числа-пороги вопроса: «больше 0,1 %», «чище 99.999», «не более 0,0002 %» (не номера марок)."""
    out = []
    for m in re.finditer(r"(?:больше|меньше|более|менее|выше|ниже|чище|грязнее|>|<|≥|≤)\s*(?:чем\s+)?"
                         r"(\d+(?:[.,]\d+)?)\s*(%|ppm)?", question, re.I):
        nxt = question[m.end():m.end() + 1]
        if m[2] or not re.match(r"[\d.]", nxt):
            # «чище, чем 6.0» без % – газ (марка), а не порог
            if not m[2] and re.fullmatch(r"\d\.\d", m[1]) and re.search(r"чище|грязнее", m[0], re.I):
                continue
            out.append(m[1].replace(",", "."))
    return out


_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_PURE, _SUM = "чист", "сумм"


def _stems() -> dict[str, str]:
    """Основа слова (5 букв) → компонент: «кислород» → O, «водяные пары» → водяные пары (из config/components.toml)."""
    from iacpaas_mcp.config import load_synonyms
    out = {}
    for k, v in load_synonyms().items():
        val = v.translate(_SUB).upper() if re.fullmatch(r"[A-Za-z₀-₉0-9+ ]+", v) else v.lower()
        words = [k.lower()] + ([v.lower()] if not re.fullmatch(r"[A-Za-z₀-₉0-9+ ]+", v) else [])
        for w in words:
            out.setdefault(w[:5], val)
            out.setdefault(w[:4], val)                 # «неон» – 4 буквы: «неону» найдётся по «неон»
    return out


def comp_of(word: str) -> str:
    """Слово «по чему» → один вид: «водороду» / «H» → H, «чистота» → чист, «сумма примесей» → сумм; иначе основа."""
    w = re.sub(r"^(?:примес\w*|содержани\w*|количеств\w*)\s+", "", word.strip().lower()).strip()
    if not w:
        return ""
    if w.startswith(("чист", "основн", "дол")):
        return _PURE
    if w.startswith("сумм"):
        return _SUM
    if re.fullmatch(r"[a-z₀-₉0-9]+", w):
        return w.translate(_SUB).upper()
    stems = _stems()
    return stems.get(w[:5]) or stems.get(w[:4]) or w[:5]


_BY_AFTER = re.compile(r"\s*(?:по\s+|у\s+)?((?:примес\w*\s+)?[а-яёa-z₀-₉0-9]+(?:\s+пар\w*|\s+примес\w*)?)")
_BY_BEFORE = re.compile(r"((?:сумм\w*\s+примес\w*)|[а-яё]+)\s+$")


def extremes(question: str) -> list[dict]:
    """Части вопроса «самый / хуже / лучше всего …»: какое действие нужно (ops), по чему (by), среди каких газов."""
    out = []
    for shown in re.split(r"[\n?;]", question):
        part = shown.lower()
        if not part.strip() or SORTED.search(part):
            continue
        subj = next(iter(subjects(part)), "")
        top = 0
        if t := TOP_N.search(part):
            raw = t[1] or t[2]
            top = int(raw) if raw.isdigit() else NUMBERS.get(raw, 0)
        matches = list(SUPER.finditer(part))
        prev_by = ""
        for j, m in enumerate(matches):
            phrase = m[0].strip()
            # «топ-3 самых грязных» – одна часть: «топ» лишь число, крайние – по следующему слову
            if phrase.startswith("топ") and j + 1 < len(matches) and matches[j + 1].start() - m.end() < 15:
                continue
            by_word = ""
            if re.search(r"чист|чищ|чище|грязн", phrase):
                by_word = "чистота"
            else:
                if (a := _BY_AFTER.match(part[m.end():])) and not re.fullmatch(r"(?:и|а|у|в|из|среди|газ\w*|как\w*|ли)",
                                                                     a[1].split()[0]):
                    by_word = a[1]
                elif b := _BY_BEFORE.search(part[:m.start()]):
                    if not re.fullmatch(r"как\w*|газ\w*|у|в|и|а|где|из|них|ему|него|дай|выдай|покажи|назови", b[1]):
                        by_word = b[1]
            # «у какого больше всего водяных паров, а у какого меньше всего» – по тому же, что предыдущая часть
            by_word = by_word or prev_by
            prev_by = by_word
            by = comp_of(by_word) if by_word else ""
            if "грязн" in phrase:
                ops = {("min", _PURE), ("max", _SUM)}
            elif re.search(r"лучш", phrase):
                ops = {("max", by)} if by == _PURE else {("min", by)}
            elif re.search(r"худш|хуже", phrase):
                ops = {("min", by)} if by == _PURE else {("max", by)}
            elif re.search(r"чист|чищ|чище|больш|наибол|высок|топ", phrase):
                ops = {("max", by)}
            else:
                ops = {("min", by)}
            out.append({"part": shown, "phrase": phrase, "subject": subj, "by_word": by_word, "by": by,
                        "ops": {o for o, _ in ops}, "pairs": ops, "top": top})
            top = 0                                    # «топ-N» – к первой крайней части
    return out


def _extreme_call(a: dict, x: dict) -> bool:
    """Запрос отвечает на эту часть: нужное действие, по тому же, среди тех же газов (или таблица с by)."""
    op, by = str(a.get("op") or ""), comp_of(str(a.get("by") or ""))
    if op == "compare" and a.get("by"):
        ok = not x["by"] or by == x["by"] or any(b == by for _, b in x["pairs"])
    else:
        ok = any(op == o and (not b or by == b) for o, b in x["pairs"])
    gases = str(a.get("gases") or "").lower()
    top_ok = not x.get("top") or str(a.get("top") or "") == str(x["top"])
    return ok and top_ok and (not x["subject"] or x["subject"][:4] in gases)


def plan_remarks(question: str, calls: list[dict]) -> list[tuple[str, str]]:
    """Явные признаки вопроса, которых нет в запросах gas_query: [(замечание модели, пометка человеку)]."""
    q = question.lower()
    ops = [str(a.get("op") or "") for a in calls]
    out: list[tuple[str, str]] = []

    def need(cond: bool, model: str, human: str) -> None:
        if cond:
            out.append((model, human))
    for x in extremes(question):
        hit = any(_extreme_call(a, x) for a in calls)
        scope = f", gases=\"{x['subject']}\"" if x["subject"] else ""
        by = f", by=\"{x['by_word']}\"" if x["by_word"] else ", by=…"
        ops_txt = " или ".join(f"op=\"{o}\"" for o in sorted(x["ops"]))
        need(not hit, f"в части «{x['part'].strip()}» на «{x['phrase']}» нужен gas_query({ops_txt}{by}{scope}) – "
             "лучше по примеси – min, хуже – max; по чистоте наоборот",
             f"«{x['phrase']}» – код выбирал не это (не те крайние, не по тому или не среди тех газов)")
    need(bool(SORTED.search(q)) and "sort" not in ops, "в вопросе сортировка – нужен gas_query(op=\"sort\", by=…, "
         "order=…)", "в вопросе сортировка, а код не сортировал")
    # по чему и куда сортировать: «отсортируй по увеличению примесей метана» – by=метан, order=asc
    for word, order in sort_specs(q):
        c = comp_of(word)
        need("sort" in ops and not any(a.get("op") == "sort" and comp_of(str(a.get("by") or "")) == c
                                       and (not order or str(a.get("order") or "desc") == order) for a in calls),
             f"в вопросе сортировка по «{word}»" + (" по возрастанию" if order == "asc" else " по убыванию" if order
                                                  else "") + f" – gas_query(op=\"sort\", by=\"{word}\""
             + (f", order=\"{order}\"" if order else "") + ")", f"в вопросе сортировка по «{word}», а код сортировал не так")
    without = re.search(r"\bбез\s+(?:примес\w*\s+)?(?!состав|ошиб|чего|них|указ|числ|жидк|назван)[а-яёa-z]", q)
    need(bool(without) and not any(a.get("without") for a in calls),
         "в вопросе «без …» – укажи without=<компонент>", "в вопросе «без …», а в запросе этого нет")
    need("кроме" in q and not any(a.get("name_not") or a.get("state") or a.get("grade") for a in calls)
         and not any(";" in str(a.get("gases") or "") for a in calls),
         "в вопросе «кроме …» – укажи name_not (или state / grade)", "в вопросе «кроме …», а в запросе этого нет")
    need(bool(re.search(r"общ\w*", q)) and "примес" in q and "impurities" not in ops,
         "в вопросе общие примеси групп – gas_query(op=\"impurities\", gases=\"<группа 1>; <группа 2>\")",
         "в вопросе «общие примеси», а код их не искал")
    need(bool(ERRORS.search(q)) and "errors" not in ops, "в вопросе ошибки состава (сумма примесей больше остатка до "
         "100 %) – gas_query(op=\"errors\")", "в вопросе «ошибки в составе», а код их не искал")
    need(bool(re.search(r"только\s+(?:с\s+)?назван|без\s+состав|не\s+заполн", q)) and "unfilled" not in ops,
         "в вопросе газы только с названием – gas_query(op=\"unfilled\")", "в вопросе «только с названием», а код "
         "таких не отбирал")
    need(bool(re.search(r"\bсмес(?!ь\s+пропан)", q)) and "mixtures" not in ops
         and not any(re.search(r"пропан|бутан|спбт", str(a.get("gases") or ""), re.I) for a in calls),
         "в вопросе газовая смесь – gas_query(op=\"mixtures\")", "в вопросе смесь, а код смеси не смотрел")
    need(bool(re.search(r"\bсравни", q)) and not any(o in ("compare", "max", "min") for o in ops),
         "в вопросе «сравни» – gas_query(op=\"compare\", gases=\"<газ 1>; <газ 2>\")", "в вопросе «сравни», а код не "
         "сравнивал")
    need(bool(re.search(r"сколько\s+(?:всего\s+)?(?:в\s+базе\s+)?(?:\w+\s+)?газ(?!\w*\s+смес)", q))
         # список – тоже ответ: заголовок «6 газов», газы только с названием – в сноске («всего в отборе 48»)
         and not any(o in ("count", "classes", "unfilled", "list") for o in ops),
         "в вопросе «сколько газов» – gas_query(op=\"count\") или op=\"classes\"", "в вопросе «сколько газов», а "
         "код не считал")
    exprs = " ".join(str(a.get("expr") or "") for a in calls if a.get("op") == "calc")
    # «op=calc, expr=12» – модель печатает число из другого куска калькулятором (числа ей писать нельзя)
    need(any(a.get("op") == "calc" and "[" not in str(a.get("expr") or "") for a in calls) and not ARITH.search(q),
         "op=calc – только расчёт по базе ([газ | что]) или арифметика из вопроса; число из ответа запроса не печатай "
         "калькулятором – вставь метку куска того запроса", "число напечатано калькулятором, а не взято из куска")
    need("во сколько раз" in q and "/" not in exprs, "в вопросе «во сколько раз» – gas_query(op=\"calc\", expr=\"[газ "
         "| что] / [газ | что]\")", "в вопросе «во сколько раз», а код этого не считал")
    need(bool(re.search(r"\bна\s*сколько\b", q)) and "-" not in exprs, "в вопросе «на сколько» – gas_query(op="
         "\"calc\", expr=\"[газ | что] - [газ | что]\")", "в вопросе «на сколько», а код разницу не считал")
    # «суммарно сколько примесей», «сумма примесей» – готовая сумма газа (card), а не расчёт «+»
    total = bool(SUM_OF.search(q))
    need(bool(re.search(r"\bслож\w*|\bв\s+сумме\b|\bсуммарн\w+", q)) and not total and "+" not in exprs,
         "в вопросе сумма значений – gas_query(op=\"calc\", expr=\"[газ | что] + [газ | что]\")",
         "в вопросе «сложи», а код сумму не считал")
    need(total and not any(a.get("op") in ("card", "compare", "errors", "mixtures") or re.search(
        r"сумм", str(a.get("by") or "") + str(a.get("show") or "") + str(a.get("where") or "")) for a in calls),
         "в вопросе сумма примесей газа – готовое значение: gas_query(op=\"card\", gases=<газ>, show=\"сумма "
         "примесей\"), не складывай сам", "в вопросе «сумма примесей», а код её не показал")
    for word in ASKED_COMP.findall(q):
        c = comp_of(word)
        need(c in set(_stems().values()) and not any(
            comp_of(str(a.get("by") or "")) == c or any(comp_of(w) == c for w in re.split(r"[;,]", str(
                a.get("show") or ""))) or a.get("op") != "card" for a in calls),
             f"в вопросе «сколько {word}» – укажи show=\"{word}\" (только спрошенное)",
             f"в вопросе «сколько {word}», а код показал весь состав")
    need(bool(re.search(r"\bppm\b", q)) and not GENERAL.search(q) and not any(
        a.get("unit") == "ppm" or "ppm" in str(a.get("where") or "") + str(a.get("expr") or "") for a in calls),
         "в вопросе ppm – укажи unit=\"ppm\"", "в вопросе ppm, а код в ppm не переводил")
    plan = " ".join(str(a.get(k) or "") for a in calls for k in ("where", "expr")).replace(",", ".")
    for x in _thresholds(question):
        need(x not in plan, f"в вопросе порог {x} – укажи его в where (например «кислород < {x}»)",
             f"в вопросе порог {x}, а в запросе его нет")
    gases_all = " ".join(str(a.get("gases") or "") + " " + str(a.get("expr") or "") for a in calls)
    for g in dict.fromkeys(GRADE_ASK.findall(question)):
        need(bool(calls) and g not in gases_all, f"в вопросе марка {g} – в gases укажи газ с этой маркой («<вещество> "
             f"{g}»), названия не придумывай", f"в вопросе марка {g}, а в запросе её нет")
    scoped = " ".join(str(a.get("gases") or "") for a in calls).lower()
    for part in re.split(r"[\n?;]", question):
        for s in subjects(part):
            # только эта часть: в других частях вопроса могут быть все газы базы
            need(bool(calls) and s[:4] not in scoped, f"в части «{part.strip()}» вопрос о газах «{s}», а запрос по "
                 f"всей базе: в запросе этой части укажи gases=\"{s}\" (другие части не меняй)",
                 f"вопрос о газах «{s}», а код смотрел все газы базы")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ итог
def check(text: str, question: str, results_all: list[tuple[str, dict]], results_turn: list[tuple[str, dict]],
          calls_turn: list[tuple[str, dict]], gas_names: set[str], grades: set[str], wrote: bool = False,
          asked_before: str = "", answers_before: str = "", dialog: bool = False,
          block_args: dict[str, dict] | None = None, plan_question: str | None = None) -> dict[str, Any]:
    """Ответ модели → итог: text (куски подставлены), problems (повтор / красная строка), remarks (запрос не
    по вопросу: [(модели, человеку)]), pieces и understood (для страницы), source: code – собрал код из базы, data –
    ИИ по данным инструментов платформы, model – написал ИИ без данных базы. asked_before – прошлые вопросы
    пользователя (его числа можно повторять), answers_before – прошлые ответы (их числа и газы можно повторять, если
    в этом ходе нет запроса к базе), dialog – вопрос о самом диалоге («что ты ответил»), block_args – метка куска →
    аргументы его запроса: смысл сверяется по запросам, чьи куски в ответе (запрос без куска – не ответ);
    plan_question – по чему сверять смысл запроса (уточнение «вернёмся к …» – без ссылки на прошлый вопрос)."""
    blocks = pieces_of(results_all)
    used: list[str] = []
    problems: list[str] = []

    def put(m: re.Match) -> str:
        key = "{{" + m[1] + "}}"
        if key not in blocks and wrote:
            return ""                                  # подтверждение записи: лишняя метка – просто убрать
        if key not in blocks:
            problems.append(f"метка {key} – такого куска в результатах инструментов нет: вставь метку block из ответа "
                            "gas_query")
            return ""
        used.append(key)
        return "\n\n" + blocks[key]["answer"] + "\n\n"
    filled = re.sub(r"\n{3,}", "\n\n", MARK.sub(put, text)).strip()
    glue = MARK.sub(" ", text)
    gas_calls = [a for n, a in calls_turn if n == GAS_TOOL]
    turn_blocks = [d["block"] for n, d in results_turn if is_piece(d)]
    # числа ответов платформы в связке можно – только у инструментов без готового куска (запись и т. п.): кусок пишет
    # код, пересказ его чисел моделью – лишний и непроверяемый по смыслу
    platform = set().union(*(_numbers(d) for n, d in results_turn if n != GAS_TOOL and not is_piece(d)), set())
    matched: list[str] = []
    seen = answers_before if not gas_calls else ""
    # ход с записью (даже отклонённой): ответ – о записи, а запросы (список ИР, газы) были служебными – метки не нужны
    writing = wrote or any(n in WRITE_TOOLS for n, _ in calls_turn)
    # ответ по данным (база газов, ИР) – только куски кода: текст модели вокруг них не показывается и не проверяется
    # (в нём модель пересказывала газы и числа – отсюда были почти все повторы); свой текст – там, где кусков нет
    if used and not writing:
        filled = "\n\n".join(blocks[k]["answer"].strip() for k in dict.fromkeys(used))
    else:
        problems += glue_problems(glue, question + " " + asked_before, gas_names, grades, platform, wrote,
                                  only_units=not calls_turn, seen=seen, matched=matched,
                                  general=not calls_turn and bool(GENERAL.search(question)),
                                  pieces="\n".join(blocks[k]["answer"] for k in dict.fromkeys(used)))
    if not calls_turn and NON_ANSWER.search(text) and len(text) < 300:
        problems.append("это не ответ: ответь на вопрос пользователя по существу (общий вопрос – своими знаниями, "
                        "2–4 предложения)")
    if turn_blocks and not used and not writing:
        problems.append("в ответе нет меток кусков: вставь метки block из результатов инструментов ("
                        + ", ".join(dict.fromkeys(turn_blocks)) + ") – данные подставит код, сам их не переписывай")
    # «удали аргон 6.0 из базы» – просьба изменить, а не вопрос о данных: отказ без запроса к базе – верный ответ
    about = bool(DOMAIN.search(question)) and not GENERAL.search(question) and not dialog \
        and not NOT_IN_BASE.search(question) and not MODIFY.search(question)
    if not calls_turn and not used and (about or (ARITH.search(question) and not dialog)):
        problems.append("вопрос о газах базы (или расчёт) – сначала gas_query: ответ пишет код из базы, сам не отвечай"
                        if about else "расчёт делает код: gas_query(op=\"calc\", expr=\"17 * 23\")")
    # запрос был, но не удался (ошибка инструмента), а модель ответила словами («не могу») – без куска не ответ
    if about and gas_calls and not used and not turn_blocks and not wrote and not platform:
        problems.append("запрос gas_query не удался – исправь его по подсказке ошибки (или другой op) и вставь метку "
                        "куска; ответ о газах без куска кода не показывается")
    shown_calls = [block_args[k] for k in dict.fromkeys(used) if block_args and k in block_args] or gas_calls
    # ход с записью – просьба «сохрани / создай», а не вопрос о данных: смысл запросов не сверяется
    remarks = plan_remarks(plan_question or question, shown_calls) if gas_calls and not wrote else []
    # «покажи дерево газа …» – нужно дерево (get_subgraph с as_tree), а не карточка или список
    if TREE_ASK.search(question) and calls_turn and not any(
            (n == "get_subgraph" and a.get("as_tree")) or n == "map_inforesources" for n, a in calls_turn):
        remarks.append(("в вопросе «дерево» – вызови get_subgraph(path=\"Загрузки/База технологических газов\", "
                        "root=<название газа или класса>, as_tree=true) и вставь метку его куска",
                        "просили дерево, а код показал не дерево"))
    uniq = list(dict.fromkeys(used))
    pieces = [blocks[k]["answer"].strip() for k in uniq]
    source = "code" if uniq else "data" if calls_turn else "model"
    # что именно сверено: число связки и инструмент, в ответе которого оно есть – человеку под ответом
    where = [(n, _numbers(d)) for n, d in results_turn if n != GAS_TOOL]
    sources = [f"{raw} – есть в ответе инструмента {next((n for n, nums in where if _dec(raw) in nums), 'gas_query')}"
               for raw in matched]
    return {"text": filled, "problems": problems, "remarks": remarks, "pieces": pieces,
            "understood": [blocks[k]["understood"] for k in uniq], "blocks": len(uniq), "source": source,
            "checked": sum(len(NUM.findall(p)) for p in pieces) + len(matched), "used": uniq, "matched": sources}


# ------------------------------------------------------------------ исправленный запрос
_WITHOUT = re.compile(r"\bбез\s+(?:примес\w*\s+)?(?!состав|ошиб|чего|них|указ|числ|жидк|назван)([а-яёa-z₀-₉0-9]+)",
                      re.I)
_EXCEPT = re.compile(r"\bкроме\s+([^,.;?]+?)(?=\s*(?:,|\.|;|\?|$|\sи\s+(?:отсорт|сорт|упоряд)))", re.I)
_SORT_BY = re.compile(r"(?:отсортир\w*|сортир\w*|упорядоч\w*)(?:\s+\w+)?\s+по\s+((?:сумм\w*\s+примес\w*)|[а-яё]+)", re.I)
# «отсортируй по увеличению примесей метана», «… по кислороду по убыванию», «по чистоте от самого чистого»
_SORT_SPEC = re.compile(r"(?:отсортир\w*|сортир\w*|упорядоч\w*)(?:\s+(?!по\b)[а-яё]+)?\s+по\s+"
                        r"(?:(?:увел\w*|возраст\w*|уменьш\w*|убыв\w*)\s+)?(?:примес\w*\s+|содержани\w*\s+)?"
                        r"((?:сумм\w*\s+примес\w*)|[а-яё]+(?:\s+пар\w*)?)", re.I)
_DIRECTION = re.compile(r"по\s+(?:увел\w*|возраст\w*)|от\s+меньш|от\s+наимен|(по\s+(?:уменьш\w*|убыв\w*)|от\s+больш|"
                        r"от\s+наибол|от\s+сам\w+\s+чист)", re.I)
_GAS_WITH = re.compile(r"газ\w*\s+с\s+(?:наиб\w*\s+|наибольш\w*\s+|самыми\s+\w+\s+)?примес\w*\s+([а-яё]+)", re.I)


def sort_specs(question: str) -> list[tuple[str, str]]:
    """По чему и куда сортировать: [(слово, asc | desc | '')]; слово – компонент, «чистота» или «сумма примесей»."""
    out = []
    for part in _parts(question):
        if not (m := _SORT_SPEC.search(part)):
            continue
        word = m[1]
        if re.fullmatch(r"(?:увел|возраст|уменьш|убыв|чист)\w*", word) and not word.startswith("чист"):
            continue
        if comp_of(word) not in set(_stems().values()) | {_PURE, _SUM}:
            continue
        d = _DIRECTION.search(part)
        out.append((word, "" if not d else "desc" if d[1] else "asc"))
    return out


def _parts(question: str) -> list[str]:
    """Части вопроса: строки, «?», «;», пункты «1.», «2)»."""
    out = re.split(r"[\n?;]|(?<=\S)\s+(?=\d[.)]\s)", question)
    return [p for p in (x.strip() for x in out) if p]


def _words(a: dict) -> list[str]:
    return [w[:4] for k in ("by", "without", "has", "gases", "name_not", "where", "show")
            for w in re.findall(r"[а-яёa-z]{3,}", str(a.get(k) or "").lower())]


def _named(c: str) -> str:
    return {_PURE: "чистота", _SUM: "сумма примесей"}.get(c, "")


def corrections(question: str, used: list[tuple[str, dict]],
                pool: list[tuple[str, dict]] | None = None) -> list[dict]:
    """Запросы, исправленные по явным признакам вопроса: [{old: метка куска или None, args, part, use}]; use – метка
    уже сделанного верного запроса из pool (все запросы хода), тогда нового запроса не нужно. Признаки – те
    же, что в plan_remarks: крайние (действие, по чему, среди каких газов), «какие аргоны», «без X», «кроме жидких /
    кроме X», сортировка. Код не придумывает новое – только чинит запрос модели (или добавляет запрос на часть с
    «самый / хуже / лучше», которую модель пропустила)."""
    out: list[dict] = []
    taken: set[str] = set()
    parts = _parts(question)
    # сначала части с крайними (их запрос узнаётся по by), потом остальные – из оставшихся запросов
    order = sorted(range(len(parts)), key=lambda j: not extremes(parts[j]))
    for i in order:
        part = parts[i]
        low = part.lower()
        xs = extremes(part)
        for x in xs:
            # сравнение без by («сравни … по метану таблицей») – другая часть вопроса, а не «самый / хуже»
            cands = [(k, a) for k, a in used if k not in taken and (a.get("op") in ("max", "min")
                                                                     or (a.get("op") == "compare" and a.get("by")))]
            target = next(((k, a) for k, a in cands if x["by"] and comp_of(str(a.get("by") or "")) == x["by"]),
                          None) or (cands[0] if len(cands) == 1 and len(xs) == 1 else None)
            if target and _extreme_call(target[1], x):
                taken.add(target[0])
                continue
            # модель уже сделала верный запрос, но вставила кусок другого – взять кусок верного
            if ready := next(((k, a) for k, a in (pool or []) if _extreme_call(a, x)), None):
                out.append({"old": target[0] if target else None, "use": ready[0], "args": ready[1], "part": part})
                if target:
                    taken.add(target[0])
                continue
            new = dict(target[1]) if target else {}
            op, by = sorted(x["pairs"])[0]
            new["op"] = op if new.get("op") != "compare" else "compare"
            if x["by_word"]:
                new["by"] = _named(by) or x["by_word"]
            if x["subject"]:
                new["gases"] = x["subject"]
            if x.get("top"):
                new["top"] = x["top"]
            if new.get("by") and _extreme_call(new, x):
                out.append({"old": target[0] if target else None, "args": new, "part": part})
                if target:
                    taken.add(target[0])
        if xs:
            continue
        # части без крайних: «какие аргоны», «без метана», «кроме жидких», «отсортируй по …»
        subj = next(iter(subjects(low)), "")
        rest = [(k, a) for k, a in used if k not in taken]
        # чей запрос: совпали слова (по чему, без чего, какие газы) и вид операции (сортировка – к «отсортируй», порог –
        # к части с порогом); «кислород» в двух частях сам по себе не решает
        sort_part, thr_part = bool(SORTED.search(low)), bool(_THRESH.search(low))

        def score(a: dict) -> int:
            s = sum(1 for w in set(_words(a)) if w in low)
            if s:
                s += 2 if sort_part == (a.get("op") == "sort") else -2
                s += 1 if thr_part == bool(a.get("where")) else 0
            return s
        scored = [(s, k, a) for k, a in rest if (s := score(a))]
        best = max((s for s, _, _ in scored), default=0)
        mine = [(k, a) for s, k, a in scored if s == best and s > 0]
        if not mine and len(parts) == len(used) and i < len(used) and used[i][0] not in taken:
            mine = [used[i]]                          # части и куски – по порядку
        if len(mine) != 1:
            continue
        k, a = mine[0]
        new = _fix_part(a, low, subj, part)
        if new != a:
            # модель уже сделала верный запрос на эту часть, но вставила кусок другого – взять кусок верного,
            # а не «чинить» чужой запрос (запрос второй части с порогами давал пустой список первой)
            ready = next(((pk, pa) for pk, pa in (pool or []) if pk != k and score(pa) >= best
                          and _fix_part(pa, low, subj, part) == pa), None)
            if ready:
                out.append({"old": k, "use": ready[0], "args": ready[1], "part": part})
                taken.add(k)
                continue
            out.append({"old": k, "args": new, "part": part})
            taken.add(k)
    out += _missing_ops(question, used, taken, pool)
    out += [f for f in grade_fixes(question, used) if f["old"] not in {x["old"] for x in out}]
    return out


def _fix_part(a: dict, low: str, subj: str, part: str) -> dict:
    """Запрос части без крайних, исправленный по явным признакам части: класс газов, «без», «кроме», сортировка,
    «газы с примесью X», пороги, «сколько X в газе». Совпал с исходным – запрос уже верный."""
    new = dict(a)
    # «grade=азот» – не марка, а класс газов
    if any(str(a.get("grade") or "").lower().startswith(s[:4]) for s in SUBSTANCES):
        new.pop("grade")
    if subj and subj[:4] not in str(a.get("gases") or "").lower():
        new["gases"] = subj
    if (m := _WITHOUT.search(low)) and not a.get("without"):
        new["without"] = m[1]
    if (m := _EXCEPT.search(low)) and not (a.get("name_not") or a.get("state")):
        what = m[1].strip()
        if re.match(r"жидк", what):
            new["state"] = "газообразный"
        elif re.match(r"газообр", what):
            new["state"] = "жидкий"
        else:
            new["name_not"] = what
    if SORTED.search(low) and a.get("op") != "sort" and (m := _SORT_BY.search(low)):
        new.update(op="sort", by=m[1], order="asc" if re.search(r"возраст|от\s+меньш", low) else "desc")
    # сортировка не по тому или не туда: «по увеличению примесей метана» – by=метан, order=asc
    for word, order in sort_specs(part):
        if new.get("op") == "sort" and (comp_of(str(new.get("by") or "")) != comp_of(word)
                                        or (order and str(new.get("order") or "desc") != order)):
            new.update(by=word, **({"order": order} if order else {}))
    # «газы с (наибольшими) примесями аргона» – только те, где аргон есть
    if (m := _GAS_WITH.search(low)) and not new.get("has") and comp_of(m[1]) in set(_stems().values()):
        new["has"] = m[1]
    # порог из вопроса: «азота не больше 0,0002 %» – в where
    where = str(new.get("where") or "")
    for m in _THRESH.finditer(low):
        x = m[3].replace(",", ".")
        if comp_of(m[1]) in set(_stems().values()) | {_PURE, _SUM} and x not in where.replace(",", "."):
            where = (where + "; " if where else "") + f"{m[1]} {_SIGNS[re.sub(r'\s+', ' ', m[2])]} {x}" \
                + (" ppm" if m[4] == "ppm" else "")
    if where != str(new.get("where") or ""):
        new["where"] = where
    # «сколько неона в гелии 5.5» – только спрошенное
    if new.get("op") == "card" and not new.get("show") and (m := ASKED_COMP.search(low)) \
            and comp_of(m[1]) in set(_stems().values()):
        new["show"] = m[1]
        if "ppm" in low:
            new["unit"] = "ppm"
    return new


_THRESH = re.compile(r"([а-яё]+(?:\s+пар\w*)?)\s+(не\s+больше|не\s+более|не\s+меньше|не\s+менее|не\s+ниже|не\s+выше|"
                     r"больше|меньше|выше|ниже)\s+(\d+(?:[.,]\d+)?)\s*(%|ppm)?", re.I)
_SIGNS = {"не больше": "<=", "не более": "<=", "не выше": "<=", "не меньше": ">=", "не менее": ">=", "не ниже": ">=",
          "больше": ">", "выше": ">", "меньше": "<", "ниже": "<"}


# «сравни его с маркой 4.8», «аргон 6.0» – номер марки из вопроса должен быть в запросе
GRADE_ASK = re.compile(r"(?:марк\w*|" + "|".join(rf"{s[:5]}\w*" for s in SUBSTANCES) + r")\s+(\d\.\d)", re.I)


def grade_fixes(question: str, used: list[tuple[str, dict]]) -> list[dict]:
    """Марки из вопроса нет ни в одном запросе (модель выдумала «марки А» вместо 4.8) – последний газ запроса
    заменяется на «<вещество> <марка>»: вещество – первое слово первого газа запроса."""
    gases_all = " ".join(str(a.get("gases") or "") + " " + str(a.get("expr") or "") for _, a in used)
    out = []
    for g in dict.fromkeys(GRADE_ASK.findall(question)):
        if g in gases_all:
            continue
        for k, a in used:
            items = split_gases(str(a.get("gases") or ""))
            if items and a.get("op") in ("compare", "card", "max", "min", "list", "sort"):
                subst = items[0].split()[0]
                items = (items[:-1] if len(items) > 1 else items) + [f"{subst} {g}"]
                out.append({"old": k, "args": {**a, "gases": "; ".join(items)}, "part": question})
                break
    return out


def split_gases(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;\n]", text) if x.strip()]


def _missing_ops(question: str, used: list[tuple[str, dict]], taken: set[str],
                 pool: list[tuple[str, dict]] | None = None) -> list[dict]:
    """Нужной операции в запросах нет вовсе – запрос на неё берётся из уже сделанных моделью (pool) или добавляет
    код: ошибки состава, газы только с названием, смеси, общие примеси групп, классы, сумма примесей газа (вместо
    сложения модели), ppm; «op=calc» с голым числом («12») вместо куска – кусок другого запроса хода."""
    q = question.lower()
    ops = {str(a.get("op") or "") for _, a in used}
    subj = next(iter(subjects(q)), "")
    out: list[dict] = []
    calc_bare = [(k, a) for k, a in used if a.get("op") == "calc" and "[" not in str(a.get("expr") or "")
                 and not ARITH.search(q)]

    def add(op: str, args: dict) -> None:
        """Нужная операция: запрос модели из pool, если был; иначе новый. Вместо голого calc – если он есть."""
        old = calc_bare.pop(0)[0] if calc_bare else None
        ready = next(((k, a) for k, a in (pool or []) if a.get("op") == op), None)
        out.append({"old": old, "use": ready[0], "args": ready[1], "part": question} if ready
                   else {"old": old, "args": args, "part": question})
    def own_subj(rx) -> str:
        """Газы той части вопроса, где этот признак («есть ли у водородов ошибки»), а не первой части."""
        part = next((p for p in _parts(question) if rx.search(p.lower())), q)
        return next(iter(subjects(part.lower())), "") if part is not q else subj
    if ERRORS.search(q) and "errors" not in ops:
        add("errors", {"op": "errors", "gases": own_subj(ERRORS)})
    if (unf := re.compile(r"только\s+(?:с\s+)?назван|без\s+состав|не\s+заполн")).search(q) and "unfilled" not in ops:
        add("unfilled", {"op": "unfilled", "gases": own_subj(unf)})
    if re.search(r"\bсмес(?!ь\s+пропан)", q) and "mixtures" not in ops:
        add("mixtures", {"op": "mixtures"})
    if re.search(r"сколько\s+(?:\w+\s+)?класс", q) and "classes" not in ops:
        add("classes", {"op": "classes"})
    if re.search(r"общ\w*", q) and "примес" in q and "impurities" not in ops:
        found = sorted((m.start(), s) for s in SUBSTANCES if (m := re.search(rf"\b{s[:5]}", q)))
        if found:
            add("impurities", {"op": "impurities", "gases": "; ".join(s for _, s in found)})
    # «сколько ppm …» – модель сделала запрос в ppm, а вставила кусок без ppm (или не перевела вовсе)
    if re.search(r"\bppm\b", q) and not GENERAL.search(q) and not any(
            a.get("unit") == "ppm" or "ppm" in str(a.get("where") or "") + str(a.get("expr") or "") for _, a in used):
        for k, a in used:
            if a.get("op") in ("card", "list", "sort", "max", "min", "compare") and k not in taken:
                same = next(((pk, pa) for pk, pa in (pool or []) if pa.get("unit") == "ppm"
                             and {x: v for x, v in pa.items() if x not in ("unit", "path")}
                             == {x: v for x, v in a.items() if x not in ("unit", "path")}), None)
                out.append({"old": k, "use": same[0], "args": same[1], "part": question} if same
                           else {"old": k, "args": {**a, "unit": "ppm"}, "part": question})
                taken.add(k)
                break
    if SUM_OF.search(q) and not any(a.get("op") in ("card", "compare", "errors", "mixtures") or re.search(
            r"сумм", str(a.get("by") or "") + str(a.get("show") or "") + str(a.get("where") or "")) for _, a in used):
        calc = [(k, a) for k, a in used if a.get("op") == "calc" and k not in taken]
        gases = list(dict.fromkeys(g.strip() for _, a in calc for g in re.findall(r"\[([^|\]]+)\|", str(a.get("expr")))))
        if gases:
            out.append({"old": calc[0][0], "args": {"op": "card", "gases": "; ".join(gases),
                                                    "show": "сумма примесей"}, "part": question})
    # сортировка, сравнение, расчёт по базе: модель запрос сделала, а метку не вставила – берётся её кусок (нового
    # запроса код здесь не придумывает)
    have = {f.get("use") for f in out}
    for need, op in ((SORTED.search(q), "sort"), (re.search(r"\bсравни", q), "compare"),
                     (re.search(r"во\s+сколько\s+раз|\bна\s*сколько\b", q), "calc")):
        if need and op not in ops and (ready := next(((k, a) for k, a in (pool or []) if a.get("op") == op
                                                       and k not in have and k not in taken
                                                       and (op != "calc" or "[" in str(a.get("expr") or ""))),
                                                      None)):
            out.append({"old": None, "use": ready[0], "args": ready[1], "part": question})
    # «19 = 19» калькулятором, а на его место ставить нечего – кусок убрать: число есть в настоящем куске
    out += [{"old": k, "drop": True, "args": a, "part": question} for k, a in calc_bare]
    return out


def call_text(args: dict) -> str:
    """Запрос строкой для замечания модели: gas_query(op="max", by="кислород", gases="азот")."""
    keep = [(k, v) for k, v in args.items() if v not in ("", None, 0) and k != "path"]
    return "gas_query(" + ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in keep) + ")"
