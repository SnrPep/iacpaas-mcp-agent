"""Настройки сервера из переменных окружения (.env). Все относительные пути считаются от корня проекта,
поэтому сервер ведёт себя одинаково, из какой бы папки его ни запустил клиент (VS Code, Claude Code, Desktop)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | os.PathLike) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_env(path: str | os.PathLike | None = None) -> Path | None:
    """Мини-загрузчик .env: KEY=VALUE, # – комментарий. Уже заданные переменные окружения не перезаписываются.
    Порядок поиска: IACPAAS_ENV_FILE, ./.env (рабочая папка), <корень проекта>/.env."""
    explicit = path or os.getenv("IACPAAS_ENV_FILE")
    # Явно заданный файл – только он (даже если его нет): тесты и CI не должны подхватывать личный .env.
    candidates = [explicit] if explicit else [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    for c in candidates:
        if not c:
            continue
        file = Path(c)
        if file.is_file():
            for raw in file.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.split(" #", 1)[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
            return file
    return None


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in {"1", "true", "yes", "on", "да"}


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if not v else int(v)


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if not v else float(v)


@dataclass
class Settings:
    backend: str = "mock"                    # mock | http
    base_url: str = "https://iacpaas.dvo.ru"
    api_key: str | None = None               # X-API-KEY (выдаёт администратор платформы)
    jwt: str | None = None                   # готовый JWT
    username: str | None = None              # или логин/пароль -> /api/signin (JWT получается и обновляется сам)
    password: str | None = None
    user_email: str | None = None            # для нормализации полных путей "<e-mail>/Мой Фонд/..."
    timeout_s: float = 60.0                  # платформа может обрабатывать запрос до 30 с
    retries: int = 2                         # повторы ТОЛЬКО для чтения и только на временные ошибки
    max_concurrency: int = 2                 # одновременных запросов чтения к платформе
    cache_ttl_s: float = 0.0                 # кэш экспорта, с; 0 – каждый запрос к платформе (данные всегда свежие)
    allow_write: bool = False                # политика сервера: есть ли инструменты записи вообще
    remote_write: bool = False               # отдельное разрешение писать на РЕАЛЬНУЮ платформу
    write_paths: list[str] = field(default_factory=list)  # куда можно писать на платформе: папки «Мой Фонд»
    confirm_ttl_s: int = 300
    # запись – только если клиент сам спросит человека окном (MCP elicitation); клиент без окна – запись выключена.
    # Включается для VS Code / Claude Code (env в .mcp.json): там токен видит модель и могла бы подтвердить сама
    require_dialog: bool = False
    confirm_mode: str = ""                # chat – подтверждение в чате (VS Code), окна сервера не используются
    # как клиент показывает кусок gas_query: text – модель пересказывает answer дословно (VS Code, любой клиент);
    # marks – вставляет метку {{#id}}, кусок подставляет наш агент (веб-чат, консоль)
    pieces: str = "text"
    max_depth: int = 6
    max_nodes: int = 400                     # сколько вершин подграфа отдавать модели за раз
    max_result_chars: int = 60_000           # предел размера сырого JSON в ответе инструмента
    registry_file: Path = field(default_factory=lambda: project_path("config/inforesources.json"))
    known_irs: list[str] = field(default_factory=list)
    # ИР, созданные агентом в папках для записи (API фонд не перечисляет – помним сами; только для платформы)
    own_registry_file: Path = field(default_factory=lambda: project_path("data/own_inforesources.json"))
    fixtures_dir: Path = field(default_factory=lambda: project_path("fixtures"))
    log_level: str = "INFO"
    log_file: Path | None = None
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
    mcp_token: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        log_file = os.getenv("LOG_FILE", "logs/server.jsonl")
        backend = os.getenv("IACPAAS_BACKEND", "mock").strip().lower()
        # реестр ИР: свой для платформы и для песочницы, чтобы не искать на сайте тестовые ИР
        registry = "fixtures/inforesources.json" if backend == "mock" else "config/inforesources.json"
        return cls(
            backend=backend,
            base_url=os.getenv("IACPAAS_URL", "https://iacpaas.dvo.ru").rstrip("/"),
            api_key=os.getenv("IACPAAS_API_KEY") or None,
            jwt=os.getenv("IACPAAS_JWT") or None,
            username=os.getenv("IACPAAS_USERNAME") or None,
            password=os.getenv("IACPAAS_PASSWORD") or None,
            user_email=os.getenv("IACPAAS_USER_EMAIL") or os.getenv("IACPAAS_USERNAME") or None,
            timeout_s=_float("IACPAAS_TIMEOUT", 60.0),
            retries=_int("IACPAAS_RETRIES", 2),
            max_concurrency=_int("IACPAAS_MAX_CONCURRENCY", 2),
            cache_ttl_s=_float("IACPAAS_CACHE_TTL", 0.0),
            allow_write=_bool("IACPAAS_ALLOW_WRITE", False),
            remote_write=_bool("IACPAAS_REMOTE_WRITE", False),
            write_paths=[p.strip().strip("/") for p in os.getenv("IACPAAS_WRITE_PATHS", "").split(";") if p.strip()],
            confirm_ttl_s=_int("IACPAAS_CONFIRM_TTL", 300),
            require_dialog=_bool("IACPAAS_REQUIRE_DIALOG", False),
            confirm_mode=os.getenv("IACPAAS_CONFIRM_MODE", "").strip().lower(),
            pieces="marks" if os.getenv("IACPAAS_PIECES", "").strip().lower() == "marks" else "text",
            max_depth=_int("IACPAAS_MAX_DEPTH", 6),
            max_nodes=_int("IACPAAS_MAX_NODES", 400),
            max_result_chars=_int("IACPAAS_MAX_RESULT_CHARS", 60_000),
            registry_file=project_path(os.getenv("IACPAAS_REGISTRY_FILE", registry)),
            known_irs=[p.strip() for p in os.getenv("IACPAAS_KNOWN_IRS", "").split(";") if p.strip()],
            own_registry_file=project_path(os.getenv("IACPAAS_OWN_REGISTRY", "data/own_inforesources.json")),
            fixtures_dir=project_path(os.getenv("IACPAAS_FIXTURES", "fixtures")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=project_path(log_file) if log_file else None,
            mcp_host=os.getenv("MCP_HOST", "127.0.0.1"),
            mcp_port=_int("MCP_PORT", 8765),
            mcp_token=os.getenv("MCP_BEARER_TOKEN") or None,
        )

    @property
    def auth_mode(self) -> str:
        if self.api_key:
            return "api_key"
        if self.jwt:
            return "jwt"
        if self.username and self.password:
            return "signin"
        return "none"


def load_synonyms(file: Path | None = None) -> dict[str, str]:
    """Синонимы компонентов из config/components.toml («метан» → CH₄) – читаются при каждом вызове: правка
    файла без перезапуска. Нужны серверу (gas_query)."""
    import tomllib
    try:
        text = (file or PROJECT_ROOT / "config" / "components.toml").read_text(encoding="utf-8")
        return {str(k): str(v) for k, v in tomllib.loads(text).get("synonyms", {}).items()}
    except (OSError, ValueError):
        return {}
