"""Policy gate для изменяющих операций: работает в любом MCP-клиенте, не полагаясь на его интерфейс.

Шаг 1: вызов без confirm_token -> ничего не меняется, возвращается предпросмотр и токен.
Шаг 2: тот же вызов с теми же аргументами + confirm_token -> операция выполняется.
Токен = HMAC(операция, аргументы, время) с секретом процесса: его нельзя подделать, применить к другим
аргументам или использовать после истечения срока. Каждый токен одноразовый."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time


class WritePolicy:
    def __init__(self, enabled: bool, ttl_s: int = 300, secret: bytes | None = None):
        self.enabled = enabled
        self.ttl_s = ttl_s
        self._secret = secret or secrets.token_bytes(32)
        self._used: dict[str, float] = {}

    def _sign(self, op: str, args: dict, issued: int, nonce: str) -> str:
        payload = json.dumps([op, args, issued, nonce], sort_keys=True, ensure_ascii=False, default=str).encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()[:32]

    def issue(self, op: str, args: dict) -> str:
        # случайная часть: два предпросмотра одной операции в одну секунду давали один и тот же токен, и второй
        # сразу считался использованным
        issued, nonce = int(time.time()), secrets.token_hex(4)
        return f"{issued}.{nonce}.{self._sign(op, args, issued, nonce)}"

    def check(self, op: str, args: dict, token: str) -> str | None:
        """None – токен годен (и помечается использованным); иначе – причина отказа."""
        try:
            issued_s, nonce, sig = token.strip().split(".", 2)
            issued = int(issued_s)
        except ValueError:
            return "confirm_token имеет неверный формат – запросите предпросмотр заново (вызов без confirm_token)"
        now = time.time()
        if now - issued > self.ttl_s:
            return "confirm_token просрочен – запросите предпросмотр заново"
        if not hmac.compare_digest(sig, self._sign(op, args, issued, nonce)):
            return "confirm_token не соответствует аргументам – аргументы должны совпадать с предпросмотром"
        if token in self._used:
            return "confirm_token уже использован – для новой операции нужен новый предпросмотр"
        self._used = {t: ts for t, ts in self._used.items() if now - ts < self.ttl_s}
        self._used[token] = now
        return None
