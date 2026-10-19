"""A small HTTP client on the standard library, enough for the platforms' REST APIs."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..util import ClipperError


class PublishError(ClipperError):
    """A platform refused or failed a request; the message is meant for the user."""


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes

    def json(self) -> dict:
        try:
            return json.loads(self.body or b"{}")
        except ValueError:
            return {}


def request(method: str, url: str, *, params: dict | None = None, form: dict | None = None,
            json_body: dict | None = None, data: bytes | None = None, headers: dict | None = None,
            timeout: float = 120, platform: str = "", ok: tuple[int, ...] = ()) -> Response:
    """Send a request; raise PublishError for HTTP errors (unless the status is listed in `ok`)."""
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = dict(headers or {})
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        headers.setdefault("Content-Type", "application/json; charset=UTF-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read())
    except urllib.error.HTTPError as e:
        resp = Response(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read() or b"")
        if e.code in ok:
            return resp
        raise PublishError(f"{platform or 'The platform'} answered {e.code}: {error_text(resp)}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise PublishError(f"Couldn't reach {platform or url}: {getattr(e, 'reason', e)}") from None


def error_text(resp: Response) -> str:
    """The most useful message in an error response from Google, TikTok or Meta."""
    body = resp.json()
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("error_user_msg") or err.get("code")
        if msg:
            return str(msg)
    if isinstance(err, str):
        return body.get("error_description") or err
    text = resp.body.decode(errors="replace").strip()
    return text[:300] or "no details"
