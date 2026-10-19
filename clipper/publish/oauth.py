"""Sign-in with OAuth 2: PKCE values and the short-lived state of a login in progress."""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field

LOGIN_TTL = 15 * 60  # seconds a started login stays valid


def code_verifier() -> str:
    return secrets.token_urlsafe(48)[:64]  # 64 chars from the unreserved set


def challenge_b64(verifier: str) -> str:
    """S256 challenge as the OAuth spec defines it (Google)."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def challenge_hex(verifier: str) -> str:
    """S256 challenge as TikTok's desktop login expects it: hex, not base64."""
    return hashlib.sha256(verifier.encode()).hexdigest()


@dataclass
class PendingLogin:
    platform: str
    redirect_uri: str
    verifier: str = field(default_factory=code_verifier)
    created: float = field(default_factory=time.time)


class Logins:
    """Logins started in the browser, keyed by their random `state` value."""

    def __init__(self):
        self._pending: dict[str, PendingLogin] = {}
        self._lock = threading.Lock()

    def start(self, platform: str, redirect_uri: str) -> tuple[str, PendingLogin]:
        state = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            self._pending = {k: v for k, v in self._pending.items() if now - v.created < LOGIN_TTL}
            self._pending[state] = login = PendingLogin(platform, redirect_uri)
        return state, login

    def finish(self, platform: str, state: str) -> PendingLogin | None:
        """The login for this state (once only), if it was started here, for this platform, recently."""
        with self._lock:
            login = self._pending.pop(state, None)
        if login is None or login.platform != platform or time.time() - login.created > LOGIN_TTL:
            return None
        return login
