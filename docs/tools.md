# Инструменты MCP-сервера и соответствие операциям IACPaaS

Названия инструментов — проектное решение. Ниже — какой метод REST API IACPaaS стоит за каждым
(https://iacpaas.dvo.ru/docs/RemoteCalls/RemoteCalls.html, OpenAPI `/v3/api-docs.yaml`).

## Чтение (всегда доступны, аннотация `readOnlyHint: true`)

| Инструмент | Аргументы | Операция IACPaaS | Результат |
|---|---|---|---|
| `iacpaas_status` | — | (локально) | режим, вход, запись, лимиты, статистика |
| `list_inforesources` | filter, limit | метода нет → реестр `config/inforesources.json` + свои ИР (`data/own_inforesources.json`) | пути, названия, описания ИР; `origin` — выдан / свой, `given` и `own` — сколько каких, `hint` — итог «всего N: M выданных и K своих в «Песочница ИИ»» |
| `map_inforesources` | path (пусто — весь набор) | полный экспорт каждого ИР (кэш) + разбор в адаптере | схема набора ИР: онтология каждого ИР, ссылки (`link`) и клоны (`original`) на другие ИР с числом и примерами, ИР вне набора; `scheme` — готовая схема деревом |
| `get_inforesource` | path | `GET /api/data/export/user-item`, `export-depth=1` | название, онтология, даты, 1-й уровень |
| `get_subgraph` | path, root, depth ≤ 6, max_nodes ≤ 400, as_tree | полный экспорт (кэш), глубина режется в адаптере* | плоский список вершин с путями, типами, meta, children; `by_meta`; `truncated` + подсказка; чисел вершин/дуг нет — их считает `count_graph`. `as_tree` — готовое дерево; больше 300 вершин — только верхние уровни (до 60 строк, `levels`, «(+N внутри)») |
| `count_graph` | path, root | полный экспорт + подсчёт в адаптере (аналог `IConcept.getSubGraphAmountsInfo` Java API) | вершины, дуги, ссылки, терминалы |
| `query_graph` | path, text / meta / vertex_type, limit (20) | экспорт + поиск в адаптере | совпадения с путями, общее число |
| `summarize_inforesource` | path | полный экспорт | числа, сколько вершин каждого метапонятия, 1-й уровень |
| `get_ontology` | path ИР или онтологии, root, depth (4) | экспорт ИР (`depth=0`) → путь `ontology` → экспорт онтологии | метапонятия до depth уровней от root: тип, valtype, спецификатор, ограничитель, `choice` (одно из вариантов) |
| `gas_query` | op (list, count, max, min, sort, card, compare, impurities, errors, unfilled, classes, mixtures, calc), gases (класс или названия через «;»), where, has, without, name_has, name_not, state, grade, standard, by, order, top, show, unit (% / ppm), expr (для calc), path | полный экспорт ИР + таблица газов в адаптере (`query.py`) | готовый кусок ответа `answer` (Markdown), строка `understood` («понял вопрос так»), метка `block`, `rows`, `data_version` (дата изменения ИР); числа – как в базе (Decimal), крайние – все равные; нет операции или компонента – ошибка с подсказкой |
| `check_inforesource` | path, root | экспорт ИР + его онтологии и связанных онтологий | нарушения ограничений онтологии: нет обязательного, лишнее, не выбран вариант, неверный тип; по вершинам |
| `export_json` | path, json_type (universal/meta/simple), root, depth | экспорт universal (+ `start-target-concept-path`, `export-depth`); simple/meta строятся из него** | JSON (≤ `IACPAAS_MAX_RESULT_CHARS`) |
| `validate_resource` | ir_json | (локально) | ошибки структуры universal JSON |
| `service_status` | service_id | `GET /api/service/{id}/running` | запущен ли сервис |

\* Во фрагменте (`start-target-concept-path`) платформа разворачивает ссылки в копии вершин из других частей ИР,
а на границе `export-depth` не видно потомков — поэтому подграф и подсчёт берутся из полного экспорта.
\*\* Платформа отдаёт `simple`/`meta` некорректным JSON и без учёта начальной вершины (проверено 13.10.2026).

Ответ модели — компактный JSON без пустых полей (метка вершины опускается, если равна концу `path`);
`structuredContent` — полный, по схеме инструмента.

У всех инструментов чтения и у `gas_query` в ответе есть готовый кусок для человека: `answer` (текст пишет код –
`src/iacpaas_mcp/pieces.py`, у схемы и дерева – блок ```) и метка `block`. Как его показать – в `hint`: наш агент
вставляет метку, код подставляет кусок (`IACPAAS_PIECES=marks`); VS Code и другие клиенты пересказывают `answer`
дословно (`text`, по умолчанию).

## Изменение (`readOnlyHint: false`, `destructiveHint: true`)

Появляются при `IACPAAS_ALLOW_WRITE=true` (с `mock` — в песочнице). На реальной платформе — при
`IACPAAS_REMOTE_WRITE=true` и заданных папках `IACPAAS_WRITE_PATHS` (сейчас `Загрузки/Песочница ИИ`): цель вне
папок — `[read_only]` до предпросмотра. Каждый — в два шага (`confirm_token`).

| Инструмент | Аргументы | Операция IACPaaS | Где |
|---|---|---|---|
| `save_selection` | source_path, title, elements (строка «газ 1; газ 2» – названия или пути газов и классов), notes[{path, text}], comment, replace, folder_path | экспорт исходного ИР → сборка подборки → `POST /api/data/import` | платформа, песочница |
| `save_mixture` | source_path, title, name, symbol, components[{gas, percent}], group, standard, comment, replace | экспорт базы → смесь по онтологии + копии газов-компонентов → `POST /api/data/import` | платформа, песочница |
| `import_inforesource` | folder_path, ir_json, clear_if_exists | `POST /api/data/import` (json — строкой) | платформа, песочница |
| `delete_from_fund` | path, mode (deleteIR/trashIR/clearIR/deleteSubGraph/clearSubGraph), concept_path, metaconcept_path | `DELETE /api/data/delete` | платформа, песочница |
| `update_concept` | path, concept_path, metaconcept_path, mark, comment | `PUT /api/data/update` | только песочница* |
| `run_service` / `stop_service` | service_id, imports[], exports[] | `POST /api/service/import-run-export`, `PUT /api/service/{id}/stop` | только песочница** |

\* На платформе метод отвечает «ошибка» без пояснения (проверено 18.10.2026 на своём ИР, 8 вариантов путей);
пометки делаются через `save_selection(notes)` — комментарии вершин при импорте.
\*\* Сервис платформы может изменить что угодно — папкой это не ограничить.

`save_selection`: новый ИР на той же онтологии из выбранных элементов (газов или классов); разделы верхнего
уровня остаются всегда (в онтологии они «копия» — обязательны); газы, на которые ссылается смесь, добавляются
сами; служебные `id`/`link_id` убираются, ссылки переписываются на новый ИР. Платформа принимает такой ИР на
чужой онтологии «только чтение» (проверено 18.10.2026).

Газы в `elements` – одной строкой через «;» (старый список тоже принимается): GigaChat в массиве путей передавал
только первый путь (проба 26.10: 0 из 6), строкой – все. Пути по названиям находит сервер (`knowledge.find_elements`):
слова без окончаний, числа марки и буквы – точно, обозначения («Ar 6.0»); несколько подходящих – при сохранении
`[ambiguous]` (спросить человека), газа нет – `[not_found]` с похожими того же вещества.

## Промпт и ресурс
- Промпт `analyze_inforesource(path)` — пошаговый разбор ИР (в VS Code — слэш-команда).
- Ресурс `iacpaas://guide` — краткая справка по лимитам и порядку работы.

## Формат ошибок
Ошибки выполнения приходят модели как `isError: true` с текстом `[вид] описание. подсказка`:
`not_found`, `invalid_path`, `invalid_request`, `too_large`, `auth`, `timeout`, `network`, `unavailable`,
`confirmation`, `read_only`, `platform_error`.
