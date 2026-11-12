import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
IR = "Учебные/База наблюдений"

# тесты не должны зависеть от личного .env разработчика
for k in list(os.environ):
    if k.startswith(("IACPAAS_", "MCP_", "AGENT_", "LOG_")):
        del os.environ[k]
os.environ["IACPAAS_ENV_FILE"] = str(ROOT / "tests" / "no.env")
os.environ["LOG_FILE"] = ""


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_settings(**overrides):
    from iacpaas_mcp.config import Settings
    s = Settings(backend="mock", fixtures_dir=FIXTURES, registry_file=FIXTURES / "inforesources.json",
                 log_file=None, cache_ttl_s=60, retries=0)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_server(**overrides):
    from iacpaas_mcp.server import build_server, create_backend
    s = make_settings(**overrides)
    backend = create_backend(s)
    return build_server(s, backend), backend
