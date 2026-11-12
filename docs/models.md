# Модели: как подключить любую

Агент работает с любой моделью, которая умеет вызывать инструменты (tool calling). Профили —
`config/models.toml`, ключи — `.env`. Выбор: `--model <имя>`, список в веб-чате, `/model <имя>` в CLI или
`AGENT_MODEL` в `.env` (`auto` — первый доступный из списка `auto`). Контекст сессии общий для всех моделей.

| Профиль | Что нужно в `.env` | Заметки |
|---|---|---|
| `ollama-qwen`, `ollama-small` | ничего; установить Ollama, `ollama pull qwen3:8b` | бесплатно, локально, данные не уходят; 8B — от 16 ГБ ОЗУ, 4B — 8 ГБ |
| `gigachat` (GigaChat-2-Max), `gigachat-pro` (GigaChat-2-Pro), `gigachat-lite` (GigaChat-2) | `GIGACHAT_CREDENTIALS`, сертификат НУЦ Минцифры в `GIGACHAT_CA_BUNDLE_FILE` | Freemium для физлиц; один ключ и общая очередь (`parallel_group`) — один запрос одновременно на все три; в веб-чате — группа «GigaChat» (`group`, подпись — `label`) |
| `yandex` | `YANDEX_API_KEY`, `YANDEX_FOLDER_ID` | OpenAI-совместимый API AI Studio; стартовый грант Yandex Cloud |
| `deepseek` | `DEEPSEEK_API_KEY` | платно; thinking выключен (иначе ошибка 400 с инструментами) |
| `openrouter` | `OPENROUTER_API_KEY` | `:free`-модели с лимитами |
| `openai`, `claude` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | только там, где сервис официально доступен; из РФ — нет (риск блокировки аккаунта) |
| `custom` | `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | любой OpenAI-совместимый сервер (LM Studio, vLLM, шлюз) |
| `demo` | ничего | сценарий без LLM — проверка цепочки агент → MCP → IACPaaS |

## Новая модель
Добавьте секцию в `config/models.toml`:
```toml
[my-model]
provider = "openai"                    # или anthropic, ollama, deepseek, openrouter, gigachat, google_genai, …
model = "имя-модели"
base_url = "https://…/v1"              # для OpenAI-совместимых
api_key = "${MY_MODEL_KEY}"            # значение — в .env
```
Проверка, что профиль принимает все инструменты: `pytest tests/test_transports_models.py -k profile`.

## Память под модель и железо
`context_tokens` в профиле – сколько токенов прошлого диалога модель видит на каждом шаге (без инструкции и схем
инструментов, ~7 тыс.). Не задано – 16 000. Сейчас: GigaChat-2-Max 24 000, Pro 16 000, Lite 12 000, `ollama-qwen`
4 000 (окно 16 тыс.), `ollama-small` 3 000 (процессор). Что не влезло – выжимка кодом: вопросы, числа и строки таблиц
из ответов дословно, пути газов (`ContextMiddleware`, `docs/architecture.md`).

## В VS Code и Claude Code
Там модель выбирается в самом клиенте (список моделей Copilot или BYOK; в Claude Code — модель Claude).
MCP-сервер один и тот же.

## Что уходит провайдеру
Всё, что возвращают инструменты (фрагменты ИР), попадает в контекст модели. Для облачных моделей это значит
передачу данных провайдеру. Для локальной Ollama — нет.
