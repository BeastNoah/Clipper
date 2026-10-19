"""Where app credentials and account logins are kept: outside the work folder, readable only by you."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


def config_dir() -> Path:
    return Path(os.environ.get("CLIPPER_CONFIG") or Path.home() / ".config" / "clipper")


class SecretStore:
    """A JSON file of {platform: {...}}, written atomically with 0600 permissions."""

    def __init__(self, name: str, root: Path | None = None):
        self.path = (root or config_dir()) / name
        self._lock = threading.Lock()

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, platform: str) -> dict | None:
        return self.load().get(platform)

    def put(self, platform: str, value: dict | None) -> None:
        with self._lock:
            data = self.load()
            if value is None:
                data.pop(platform, None)
            else:
                data[platform] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            tmp = self.path.with_name(self.path.name + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.path)


def apps(root: Path | None = None) -> SecretStore:
    """Developer-app credentials you create on each platform (client id and secret)."""
    return SecretStore("apps.json", root)


def accounts(root: Path | None = None) -> SecretStore:
    """Signed-in accounts: access and refresh tokens, plus the name shown in the UI."""
    return SecretStore("accounts.json", root)
