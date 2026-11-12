# Когда появится доступ к платформе IACPaaS

Всё ниже — только чтение. Запись остаётся выключенной.

1. **Вход.** В `.env`: `IACPAAS_API_KEY=` (ключ от администратора) **или** `IACPAAS_USERNAME=`/`IACPAAS_PASSWORD=`.
2. **Реестр ИР.** В `config/inforesources.json` — пути ваших ИР внутри «Мой Фонд» (без `<e-mail>/Мой Фонд/`) и
   описания «что это за база». Агент выбирает ИР по описаниям — пишите их понятно.
3. **Проверка платформы:** `.venv\Scripts\python scripts\check_platform.py`
   Отчёт `logs/platform_check.json`: ответ экспорта (сам ИР или обёртка), время ответа, работает ли
   `start-target-concept-path` в виде меток от корня, форматы meta/simple, подсчёт вершин/дуг.
   Если что-то отличается от ожиданий — правка в одном месте: `src/iacpaas_mcp/backend/http.py`.
4. **Реальные данные в песочницу:** `check_platform.py --save` сохранит экспорт в `fixtures/` — тесты и демо
   будут на настоящих структурах.
5. **Переключение:** `IACPAAS_BACKEND=http` в `.env`. Перезапустить клиента (`MCP: List Servers → Restart` в VS Code,
   `/mcp` в Claude Code). `iacpaas_status` покажет `backend: http`.
6. **Инструкция агента.** В `config/agent_prompt.md` → «Термины предметной области»: 5–10 строк о смысле ключевых
   понятий вашей онтологии (структуру агент берёт сам через `get_ontology`) и 2–3 примера вопросов. То же — в
   `.github/agents/iacpaas.agent.md` для VS Code.
7. **Прогон сценариев** из `docs/scenarios.md` на реальных ИР; смотреть `logs/server.jsonl` и `logs/agent_trace.jsonl`.
8. **Сверить подсчёт** вершин/дуг с данными платформы. Учесть: ссылки на вершины других ИР считаются дугами, но не
   вершинами.

## Если когда-нибудь понадобится запись (только после согласования)
`IACPAAS_ALLOW_WRITE=true` и `IACPAAS_REMOTE_WRITE=true`. Начинать с копии ИР. Проверить формат
`metaconcept_path` для `/api/data/update` на копии. Каждое изменение — предпросмотр → подтверждение → `audit_write`.
