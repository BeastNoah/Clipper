"""Publishing finished clips to YouTube Shorts, TikTok and Instagram Reels with the platforms'
official APIs, using developer apps and logins that stay on this computer.

Logins are kept in ~/.config/clipper (accounts.json, apps.json; readable only by you). Every post is
recorded per clip in <work>/output/published.json, so a clip isn't posted to the same place twice by
accident.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..util import log, read_json, write_json
from . import store
from .base import Platform
from .http import PublishError
from .instagram import Instagram
from .oauth import Logins
from .tiktok import TikTok
from .youtube import YouTube

PLATFORMS: dict[str, type[Platform]] = {"youtube": YouTube, "tiktok": TikTok, "instagram": Instagram}

__all__ = ["PLATFORMS", "PublishError", "Logins", "status", "save_app", "connect_url", "finish_connect",
           "disconnect", "options", "publish_clip", "published"]


def _cls(name: str) -> type[Platform]:
    if name not in PLATFORMS:
        raise PublishError(f"Unknown platform '{name}' (use {', '.join(PLATFORMS)}).")
    return PLATFORMS[name]


def _platform(name: str) -> Platform:
    return _cls(name)(store.apps().get(name) or {})


def status() -> list[dict]:
    """Each platform: is the developer app set up, which account is connected."""
    apps, accounts = store.apps().load(), store.accounts().load()
    out = []
    for name, cls in PLATFORMS.items():
        app, acct = apps.get(name) or {}, accounts.get(name)
        out.append({"name": name, "label": cls.label, "setup_url": cls.setup_url,
                    "app_fields": list(cls.app_fields),
                    "configured": all(app.get(k) for k in cls.app_fields),
                    "connected": bool(acct), "account": (acct or {}).get("name")})
    return out


def save_app(name: str, values: dict) -> None:
    """Store a platform's developer-app credentials (only its known fields)."""
    cls = _cls(name)
    allowed = (*cls.app_fields, "config_id")
    app = {k: str(values[k]).strip() for k in allowed if str(values.get(k) or "").strip()}
    missing = [k for k in cls.app_fields if k not in app]
    if missing:
        raise PublishError(f"Missing {', '.join(missing)} for {cls.label}.")
    store.apps().put(name, app)


def connect_url(name: str, redirect_uri: str, logins: Logins) -> str:
    plat = _platform(name)
    state, login = logins.start(name, redirect_uri)
    return plat.authorize_url(redirect_uri, state, login.verifier)


def finish_connect(name: str, state: str, code: str, logins: Logins) -> str:
    login = logins.finish(name, state)
    if login is None:
        raise PublishError("This sign-in link is unknown or too old. Start connecting the account again.")
    account = _platform(name).finish_login(code, login.redirect_uri, login.verifier)
    account["connected"] = time.time()
    store.accounts().put(name, account)
    log(f"publish: connected {PLATFORMS[name].label} as {account.get('name')}")
    return account.get("name") or name


def disconnect(name: str) -> None:
    _cls(name)
    store.accounts().put(name, None)


def _account(name: str, plat: Platform) -> dict:
    acct = store.accounts().get(name)
    if not acct:
        raise PublishError(f"Connect your {plat.label} account first.")
    acct, refreshed = plat.fresh(acct)
    if refreshed:
        store.accounts().put(name, acct)
    return acct


def options(name: str) -> dict:
    plat = _platform(name)
    return plat.options(_account(name, plat))


def published(work: Path) -> dict:
    path = work / "output" / "published.json"
    return read_json(path) if path.exists() else {}


def publish_clip(work: Path, clip: dict, name: str, form: dict, again: bool = False) -> dict:
    """Upload one rendered clip and post it; returns the record that is also saved."""
    plat = _platform(name)
    if not clip.get("file"):
        raise PublishError("This clip hasn't been rendered yet.")
    video = work / "output" / clip["file"]
    if not video.is_file():
        raise PublishError(f"The clip file {clip['file']} is missing; render the clip again.")
    done = published(work).get(clip["file"], {}).get(name)
    if done and done.get("status") != "failed" and not again:
        raise PublishError(f"This clip was already posted to {plat.label}"
                           + (f": {done['url']}" if done.get("url") else "") + ".")
    account = _account(name, plat)
    duration = float(clip.get("final_duration") or clip.get("duration") or (clip["end"] - clip["start"]))
    log(f"publish: posting \"{clip['title']}\" to {plat.label} as {account.get('name')}")
    result = plat.publish(account, video, form, duration)
    record = {**result, "account": account.get("name"), "at": time.time()}
    records = published(work)
    records.setdefault(clip["file"], {})[name] = record
    write_json(work / "output" / "published.json", records)
    log(f"publish: {plat.label}: {result.get('note') or result.get('status')}"
        + (f" {result['url']}" if result.get("url") else ""))
    return record
