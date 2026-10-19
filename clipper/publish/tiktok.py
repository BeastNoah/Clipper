"""TikTok through the Content Posting API (Direct Post, file upload).

Until TikTok audits your developer app, posts can only be private ("only me"), and your account must
be set to private when you post. The form follows TikTok's content-sharing guidelines: you pick the
privacy level and interaction settings yourself, disclose branded content, and agree to the
Music Usage Confirmation before anything is uploaded.
"""

from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

from ..util import Progress, check_cancel
from .base import Platform, clean_text, expires_at
from .http import PublishError, request

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API_URL = "https://open.tiktokapis.com/v2"
SCOPES = "user.info.basic,video.publish"
MIN_CHUNK, MAX_CHUNK, CHUNK = 5 * 1024 * 1024, 64 * 1024 * 1024, 10 * 1024 * 1024
POLL_SECONDS, POLL_LIMIT = 5, 15 * 60


def chunk_plan(size: int) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) per TikTok's media transfer rules: 5-64 MB chunks, the last
    one absorbing the remainder; files up to 64 MB go in one piece."""
    if size <= 0:
        raise PublishError("The clip file is empty.")
    if size <= MAX_CHUNK:
        return size, 1
    return CHUNK, size // CHUNK


class TikTok(Platform):
    name, label = "tiktok", "TikTok"
    app_fields = ("client_key", "client_secret")
    setup_url = "https://developers.tiktok.com/apps"

    def authorize_url(self, redirect_uri: str, state: str, verifier: str) -> str:
        from .oauth import challenge_hex

        return AUTH_URL + "?" + urllib.parse.urlencode({
            "client_key": self.app["client_key"], "response_type": "code", "scope": SCOPES,
            "redirect_uri": redirect_uri, "state": state,
            "code_challenge": challenge_hex(verifier), "code_challenge_method": "S256",
        })

    def finish_login(self, code: str, redirect_uri: str, verifier: str) -> dict:
        tok = self._token({"code": code, "grant_type": "authorization_code", "redirect_uri": redirect_uri,
                           "code_verifier": verifier})
        account = _account(tok)
        user = self._api("GET", "/user/info/", account, params={"fields": "open_id,display_name"})
        account["name"] = (user.get("user") or {}).get("display_name") or "TikTok account"
        return account

    def refresh(self, account: dict) -> dict:
        if not account.get("refresh_token") or (account.get("refresh_expires_at") or 0) < time.time():
            return super().refresh(account)
        tok = self._token({"grant_type": "refresh_token", "refresh_token": account["refresh_token"]})
        return {**account, **_account(tok)}

    def options(self, account: dict) -> dict:
        info = self._api("POST", "/post/publish/creator_info/query/", account, json_body={})
        return {
            "account": info.get("creator_nickname") or account.get("name"),
            "username": info.get("creator_username"),
            "privacy_options": info.get("privacy_level_options") or [],
            "comment_disabled": bool(info.get("comment_disabled")),
            "duet_disabled": bool(info.get("duet_disabled")),
            "stitch_disabled": bool(info.get("stitch_disabled")),
            "max_duration": info.get("max_video_post_duration_sec"),
        }

    def check(self, info: dict, form: dict, duration: float) -> dict:
        """The post_info to send, after checking the form against TikTok's rules."""
        if form.get("consent") is not True:
            raise PublishError("Agree to TikTok's Music Usage Confirmation before posting.")
        privacy = form.get("privacy")
        if privacy not in info["privacy_options"]:
            raise PublishError("Choose who can see the video.")
        if info.get("max_duration") and duration > info["max_duration"]:
            raise PublishError(f"This clip is {duration:.0f}s; your account can post up to {info['max_duration']}s.")
        brand = form.get("commercial") is True
        your_brand, branded = form.get("your_brand") is True, form.get("branded_content") is True
        if brand and not (your_brand or branded):
            raise PublishError("Say whether the clip promotes yourself, a third party, or both.")
        if brand and branded and privacy == "SELF_ONLY":
            raise PublishError("Branded content can't be private on TikTok.")

        def allowed(key: str, disabled: bool) -> bool:
            return form.get(key) is True and not disabled

        return {
            "title": clean_text(form.get("title"), 2200),
            "privacy_level": privacy,
            "disable_comment": not allowed("allow_comment", info["comment_disabled"]),
            "disable_duet": not allowed("allow_duet", info["duet_disabled"]),
            "disable_stitch": not allowed("allow_stitch", info["stitch_disabled"]),
            "brand_organic_toggle": brand and your_brand,
            "brand_content_toggle": brand and branded,
        }

    def publish(self, account: dict, video: Path, form: dict, duration: float) -> dict:
        post_info = self.check(self.options(account), form, duration)  # fresh creator info, as required
        size = video.stat().st_size
        chunk_size, count = chunk_plan(size)
        init = self._api("POST", "/post/publish/video/init/", account, json_body={
            "post_info": post_info,
            "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk_size,
                            "total_chunk_count": count},
        })
        publish_id, upload_url = init.get("publish_id"), init.get("upload_url")
        if not publish_id or not upload_url:
            raise PublishError("TikTok didn't start the upload.")
        progress = Progress("Uploading to TikTok", size)
        with video.open("rb") as f:
            for n in range(count):
                first = n * chunk_size
                last = size - 1 if n == count - 1 else first + chunk_size - 1
                body = f.read(last - first + 1)
                request("PUT", upload_url, data=body, platform=self.label, timeout=600, headers={
                    "Content-Type": "video/mp4", "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {first}-{last}/{size}",
                })
                progress.update(last + 1)
        status = self._wait(account, publish_id)
        note = "TikTok is still processing the post; it can take a few minutes to appear."
        if status == "PUBLISH_COMPLETE":
            note = "Posted." + (" It's visible only to you until TikTok audits your app."
                                if post_info["privacy_level"] == "SELF_ONLY" else "")
        return {"id": publish_id, "url": None, "status": status.lower(), "note": note}

    def _wait(self, account: dict, publish_id: str) -> str:
        deadline = time.time() + POLL_LIMIT
        while True:
            data = self._api("POST", "/post/publish/status/fetch/", account, json_body={"publish_id": publish_id})
            status = data.get("status") or "PROCESSING_UPLOAD"
            if status == "FAILED":
                raise PublishError(f"TikTok couldn't publish the video: {data.get('fail_reason') or 'unknown reason'}")
            if status in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX") or time.time() > deadline:
                return status
            check_cancel()
            time.sleep(POLL_SECONDS)

    def _token(self, form: dict) -> dict:
        tok = request("POST", f"{API_URL}/oauth/token/", platform=self.label, form={
            "client_key": self.app["client_key"], "client_secret": self.app["client_secret"], **form}).json()
        if "access_token" not in tok:
            raise PublishError(f"TikTok sign-in failed: {tok.get('error_description') or tok.get('error') or tok}")
        return tok

    def _api(self, method: str, path: str, account: dict, **kw) -> dict:
        body = request(method, API_URL + path, platform=self.label,
                       headers={"Authorization": f"Bearer {account['access_token']}"}, **kw).json()
        err = body.get("error") or {}
        if err.get("code") not in (None, "ok"):
            raise PublishError(f"TikTok: {err.get('message') or err.get('code')}")
        return body.get("data") or {}


def _account(tok: dict) -> dict:
    return {"access_token": tok["access_token"], "refresh_token": tok.get("refresh_token"),
            "expires_at": expires_at(tok.get("expires_in")),
            "refresh_expires_at": expires_at(tok.get("refresh_expires_in")), "open_id": tok.get("open_id")}
