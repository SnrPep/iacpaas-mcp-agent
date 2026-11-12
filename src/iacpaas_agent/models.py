"""Реестр моделей: профиль из config/models.toml -> чат-модель LangChain.
Подключить новую модель = добавить секцию в models.toml; код менять не нужно.
Строки вида ${VAR} подставляются из окружения (.env); секреты в models.toml не хранятся."""
from __future__ import annotations

import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from iacpaas_mcp.config import project_path

_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ModelConfigError(RuntimeError):
    pass


def load_profiles(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelConfigError(f"Нет файла профилей моделей: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def profiles(path: Path) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in load_profiles(path).items() if isinstance(v, dict)}


def missing_vars(profile: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def scan(v: Any) -> None:
        if isinstance(v, str):
            found.extend(m for m in _VAR.findall(v) if not os.getenv(m))
        elif isinstance(v, dict):
            for x in v.values():
                scan(x)
        elif isinstance(v, list):
            for x in v:
                scan(x)
    scan(profile)
    return sorted(set(found))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ[m.group(1)], value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


_TAGS: dict[str, tuple[float, set[str] | None]] = {}     # base_url -> (когда проверяли, скачанные модели)


def _ollama_tags(base_url: str) -> set[str] | None:
    """Модели Ollama или None, если она не запущена. Ответ помним 30 с: на Windows отказ соединения с localhost
    ждёт ~2 с, а проверяется каждый локальный профиль – список моделей в веб-чате грузился 6 с."""
    hit = _TAGS.get(base_url)
    if hit and time.monotonic() - hit[0] < 30:
        return hit[1]
    try:
        r = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=1.0, trust_env=False)
        names = {m.get("name", "") for m in r.json().get("models", [])} if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError, AttributeError):
        names = None
    _TAGS[base_url] = (time.monotonic(), names)
    return names


def _ollama_up(base_url: str, model: str = "") -> bool:
    """Ollama запущена и (если задано) модель скачана: сервер без модели для auto не годится."""
    names = _ollama_tags(base_url)
    if names is None:
        return False
    return not model or model in names or (":" not in model and f"{model}:latest" in names)


def resolve_name(name: str, path: Path) -> str:
    """'auto' -> первый профиль из списка auto, для которого заданы все переменные (для Ollama – сервер запущен
    и модель скачана)."""
    if name != "auto":
        return name
    data = load_profiles(path)
    for candidate in data.get("auto", []):
        p = data.get(candidate)
        if not isinstance(p, dict) or missing_vars(p):
            continue
        if p.get("provider") == "ollama" and not _ollama_up(p.get("base_url", "http://localhost:11434"),
                                                            p.get("model", "")):
            continue
        return candidate
    raise ModelConfigError(
        "Не найдено ни одной доступной модели. Варианты: установите Ollama и скачайте модель профиля "
        "(`ollama pull qwen3:4b-instruct-2507-q4_K_M`), "
        "или задайте ключ в .env (GIGACHAT_CREDENTIALS, YANDEX_API_KEY+YANDEX_FOLDER_ID, DEEPSEEK_API_KEY, "
        "OPENROUTER_API_KEY …), или укажите профиль явно: --model <имя>.")


def describe(path: Path) -> list[dict[str, Any]]:
    out = []
    for name, p in profiles(path).items():
        demo, local = p.get("provider") == "fake", p.get("provider") == "ollama"
        ready = not missing_vars(p)
        if local and ready:                      # локальная модель готова, только если Ollama запущена и модель скачана
            ready = _ollama_up(p.get("base_url", "http://localhost:11434"), p.get("model", ""))
        out.append({"name": name, "provider": "demo" if demo else p.get("provider"),
                    "model": "сценарий без LLM" if demo else p.get("model"),
                    "label": p.get("label", ""), "group": p.get("group", ""),    # подпись и поставщик в выборе
                    "kind": "demo" if demo else "local" if local else "cloud",
                    "ready": ready, "missing": missing_vars(p), "note": p.get("note", "")})
    return out


def parallel_limit(name: str, path: Path) -> tuple[str, int]:
    """Сколько запросов к модели можно слать одновременно (ключ parallel; 0 – без ограничения) и общая группа
    (parallel_group, по умолчанию провайдер): у GigaChat Freemium – один запрос за раз на весь ключ."""
    p = profiles(path).get(resolve_name(name, path), {})
    return str(p.get("parallel_group") or p.get("provider", name)), int(p.get("parallel", 0) or 0)


DEFAULT_CONTEXT = 16000                              # токенов прошлого диалога, если в профиле не задано


def context_budget(name: str, path: Path) -> int:
    """Сколько токенов прошлого диалога модель видит на каждом шаге (context_tokens в профиле): у GigaChat окно
    большое, у локальной модели на слабом железе – маленькое. Что не влезло – выжимкой кодом (ContextMiddleware)."""
    try:
        p = profiles(path).get(resolve_name(name, path), {})
    except ModelConfigError:
        p = {}
    return int(p.get("context_tokens") or DEFAULT_CONTEXT)


def context_window(name: str, path: Path) -> int:
    """Окно модели целиком (window_tokens в профиле, у Ollama – num_ctx): инструкция, схемы инструментов, история и
    текущий вопрос вместе. В шапке веб-чата – какую долю окна займёт следующий запрос; не задано – 0 (доля не
    показывается)."""
    try:
        p = profiles(path).get(resolve_name(name, path), {})
    except ModelConfigError:
        p = {}
    return int(p.get("window_tokens") or p.get("num_ctx") or 0)          # 0 – окно неизвестно


def chars_per_token(name: str, path: Path) -> float:
    """Сколько символов текста в среднем на токен у этой модели (chars_per_token в профиле) – для показа контекста в
    шапке веб-чата, пока модель в этом чате не отвечала (дальше – замер по её ответам): у GigaChat по трассе ~4
    (инструкция и схемы ~42 тыс. символов = 10,4 тыс. токенов). Не задано – 0 (неизвестно)."""
    try:
        p = profiles(path).get(resolve_name(name, path), {})
    except ModelConfigError:
        p = {}
    return float(p.get("chars_per_token") or 0)


def make_model(name: str, path: Path) -> tuple[str, BaseChatModel]:
    name = resolve_name(name, path)
    data = profiles(path)
    if name not in data:
        raise ModelConfigError(f"Нет профиля '{name}'. Доступны: {', '.join(data)}")
    raw = dict(data[name])
    if miss := missing_vars(raw):
        raise ModelConfigError(f"Профиль '{name}': задайте в .env {', '.join(miss)}")
    p = _expand(raw)
    for k in ("note", "parallel", "parallel_group", "label", "group", "context_tokens", "window_tokens",
              "chars_per_token"):   # служебные
        p.pop(k, None)
    provider, model = p.pop("provider"), p.pop("model")
    if provider == "gigachat":                      # нет во встроенном списке init_chat_model
        from langchain_gigachat import GigaChat
        if ca := os.getenv("GIGACHAT_CA_BUNDLE_FILE"):  # относительный путь – от корня проекта, а не от папки запуска
            ca_path = project_path(ca)
            if not ca_path.is_file():
                raise ModelConfigError(
                    f"GigaChat: нет файла сертификата Минцифры {ca_path}. Скачайте russian_trusted_root_ca.cer "
                    "с https://www.gosuslugi.ru/crt в папку certs/ или поправьте GIGACHAT_CA_BUNDLE_FILE в .env.")
            p.setdefault("ca_bundle_file", str(ca_path))
        return name, GigaChat(model=model, **p)
    if provider == "fake":                          # сценарная модель для тестов и демо без сети
        from .testing import ScriptedModel
        return name, ScriptedModel(script=p["script"], cycle=bool(p.get("cycle", False)),
                                   delay=float(p.get("delay", 0)))
    return name, init_chat_model(model, model_provider=provider, **p)
