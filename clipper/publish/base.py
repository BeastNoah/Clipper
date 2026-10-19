"""What every platform provides: sign-in, token refresh, posting options and the upload itself."""

from __future__ import annotations

import time
from pathlib import Path

from .http import PublishError


class Platform:
    name = ""
    label = ""
    app_fields: tuple[str, ...] = ()  # keys of this platform's entry in apps.json
    setup_url = ""                    # where you create the developer app

    def __init__(self, app: dict):
        missing = [k for k in self.app_fields if not app.get(k)]
        if missing:
            raise PublishError(f"{self.label} isn't set up yet: add your app's {', '.join(missing)} "
                               f"(see the publishing guide).")
        self.app = app

    # -- sign-in
    def authorize_url(self, redirect_uri: str, state: str, verifier: str) -> str:
        raise NotImplementedError

    def finish_login(self, code: str, redirect_uri: str, verifier: str) -> dict:
        """Exchange the code; return the account to store (tokens, expiry, display name)."""
        raise NotImplementedError

    def refresh(self, account: dict) -> dict:
        raise PublishError(f"Your {self.label} login has expired. Connect the account again.")

    def fresh(self, account: dict) -> tuple[dict, bool]:
        """The account with a usable access token, and whether it was refreshed (so it gets saved)."""
        if account.get("expires_at") and account["expires_at"] - time.time() < 120:
            return self.refresh(account), True
        return account, False

    # -- posting
    def options(self, account: dict) -> dict:
        """What the publish form offers for this account (fetched each time the form opens)."""
        return {}

    def publish(self, account: dict, video: Path, form: dict, duration: float) -> dict:
        """Upload and post the clip; return {"id", "url", "status", "note"}."""
        raise NotImplementedError


def expires_at(seconds) -> float | None:
    try:
        return time.time() + float(seconds) if seconds else None
    except (TypeError, ValueError):
        return None


def clean_text(text, limit: int) -> str:
    return str(text or "").strip()[:limit]


def require_choice(form: dict, key: str, allowed, what: str):
    value = form.get(key)
    if value not in allowed:
        raise PublishError(f"Choose {what}.")
    return value
