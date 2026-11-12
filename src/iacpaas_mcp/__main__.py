"""Запуск MCP-сервера IACPaaS.

  python -m iacpaas_mcp                          # stdio – так его запускают VS Code / Claude Code / агент --stdio
  python -m iacpaas_mcp --transport http         # Streamable HTTP на 127.0.0.1:8765/mcp, нужен MCP_BEARER_TOKEN
  python -m iacpaas_mcp --check                  # проверить настройки и выйти
"""
import argparse
import hmac
import logging
import sys

from pydantic import AnyHttpUrl

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from . import logs
from .config import Settings, load_env
from .server import build_server, create_backend, write_mode

log = logging.getLogger("iacpaas_mcp.main")


class StaticTokenVerifier(TokenVerifier):
    """Один общий секрет для локального HTTP-режима. Для публичного развёртывания нужен OAuth 2.1."""

    def __init__(self, token: str, resource: str):
        self._token, self._resource = token, resource

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token.encode(), self._token.encode()):
            return AccessToken(token=token, client_id="local-client", scopes=["iacpaas"], resource=self._resource)
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="iacpaas-mcp", description="MCP-сервер-адаптер к IACPaaS")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default=None, help="по умолчанию MCP_HOST или 127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="по умолчанию MCP_PORT или 8765")
    ap.add_argument("--check", action="store_true", help="проверить настройки и выйти")
    args = ap.parse_args(argv)

    env_file = load_env()
    settings = Settings.from_env()
    logs.setup(settings.log_level, settings.log_file)
    try:
        backend = create_backend(settings)
    except ValueError as e:
        log.error("config_error", extra={"error": str(e)})
        print(f"Ошибка настройки: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    if backend.name == "http" and not settings.base_url.startswith("https://"):
        log.warning("insecure_platform_url", extra={"url": settings.base_url,
                                                    "hint": "ключ и пароль пойдут без шифрования; используйте https"})
    mode = write_mode(settings, backend)
    if settings.allow_write and backend.name == "http" and mode == "off":
        log.warning("remote_write_disabled", extra={"hint": "IACPAAS_ALLOW_WRITE=true, но IACPAAS_REMOTE_WRITE=false: "
                                                            "с реальной платформой – только чтение"})
    log.info("server_start", extra={"transport": args.transport, "backend": backend.name, "auth": settings.auth_mode,
                                    "write": mode, "env_file": str(env_file) if env_file else None,
                                    "log_file": str(settings.log_file) if settings.log_file else None})
    if args.check:
        print(f"OK: backend={backend.name}, auth={settings.auth_mode}, write={mode}, env={env_file}, "
              f"log={settings.log_file}", file=sys.stderr)
        return

    if args.transport == "stdio":
        build_server(settings, backend).run()           # stdout занят протоколом, логи – stderr/файл
        return

    host = args.host or settings.mcp_host
    port = args.port or settings.mcp_port
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("non_local_bind", extra={"host": host, "hint": "сервер виден в сети; нужен TLS и OAuth"})
    if not settings.mcp_token or len(settings.mcp_token) < 16:
        print("Задайте MCP_BEARER_TOKEN (не короче 16 символов): HTTP-режим без токена не запускается.", file=sys.stderr)
        raise SystemExit(2)
    resource = f"http://{'127.0.0.1' if host == 'localhost' else host}:{port}/mcp"
    mcp = build_server(
        settings, backend,
        token_verifier=StaticTokenVerifier(settings.mcp_token, resource),
        auth=AuthSettings(issuer_url=AnyHttpUrl(f"http://{host}:{port}"),     # отдельного OAuth-сервера нет
                          resource_server_url=AnyHttpUrl(resource),
                          required_scopes=["iacpaas"], validate_token_resource=True),
    )
    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
