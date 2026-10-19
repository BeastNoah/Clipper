"""YouTube Shorts through the YouTube Data API (resumable upload). A vertical clip under 3 minutes
becomes a Short automatically."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from ..util import Progress
from .base import Platform, clean_text, expires_at, require_choice
from .http import PublishError, request

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_URL = "https://www.googleapis.com/youtube/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
CHUNK = 8 * 1024 * 1024  # a multiple of 256 KB, as resumable uploads require
PRIVACY = ("public", "unlisted", "private")


class YouTube(Platform):
    name, label = "youtube", "YouTube Shorts"
    app_fields = ("client_id", "client_secret")
    setup_url = "https://console.cloud.google.com/apis/credentials"

    def authorize_url(self, redirect_uri: str, state: str, verifier: str) -> str:
        from .oauth import challenge_b64

        return AUTH_URL + "?" + urllib.parse.urlencode({
            "client_id": self.app["client_id"], "redirect_uri": redirect_uri, "response_type": "code",
            "scope": SCOPES, "access_type": "offline", "prompt": "consent", "state": state,
            "code_challenge": challenge_b64(verifier), "code_challenge_method": "S256",
        })

    def finish_login(self, code: str, redirect_uri: str, verifier: str) -> dict:
        tok = request("POST", TOKEN_URL, platform=self.label, form={
            "code": code, "client_id": self.app["client_id"], "client_secret": self.app["client_secret"],
            "redirect_uri": redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
        }).json()
        account = {"access_token": tok["access_token"], "refresh_token": tok.get("refresh_token"),
                   "expires_at": expires_at(tok.get("expires_in"))}
        ch = request("GET", f"{API_URL}/channels", platform=self.label, params={"part": "snippet", "mine": "true"},
                     headers=_auth(account)).json().get("items") or []
        if not ch:
            raise PublishError("This Google account has no YouTube channel yet. Create one on youtube.com first.")
        account["name"], account["channel_id"] = ch[0]["snippet"]["title"], ch[0]["id"]
        return account

    def refresh(self, account: dict) -> dict:
        if not account.get("refresh_token"):
            return super().refresh(account)
        tok = request("POST", TOKEN_URL, platform=self.label, form={
            "client_id": self.app["client_id"], "client_secret": self.app["client_secret"],
            "refresh_token": account["refresh_token"], "grant_type": "refresh_token",
        }).json()
        return {**account, "access_token": tok["access_token"], "expires_at": expires_at(tok.get("expires_in"))}

    def options(self, account: dict) -> dict:
        return {"account": account.get("name"), "privacy_options": list(PRIVACY)}

    def publish(self, account: dict, video: Path, form: dict, duration: float) -> dict:
        title = clean_text(form.get("title"), 100).replace("<", "").replace(">", "")
        if not title:
            raise PublishError("Give the video a title.")
        privacy = require_choice(form, "privacy", PRIVACY, "who can see the video")
        kids = require_choice(form, "made_for_kids", (True, False), "whether the video is made for kids")
        size = video.stat().st_size
        meta = {
            "snippet": {"title": title, "categoryId": "22",
                        "description": clean_text(form.get("description"), 5000).replace("<", "").replace(">", "")},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": kids},
        }
        start = request("POST", UPLOAD_URL, platform=self.label, json_body=meta,
                        params={"uploadType": "resumable", "part": "snippet,status"},
                        headers={**_auth(account), "X-Upload-Content-Type": "video/mp4",
                                 "X-Upload-Content-Length": str(size)})
        session = start.headers.get("location")
        if not session:
            raise PublishError("YouTube didn't start the upload (no upload address).")
        progress = Progress("Uploading to YouTube", size)
        done, result = 0, None
        with video.open("rb") as f:
            while done < size:
                chunk = f.read(CHUNK)
                last = done + len(chunk) - 1
                r = request("PUT", session, data=chunk, platform=self.label, ok=(308,), timeout=600,
                            headers={**_auth(account), "Content-Type": "video/mp4",
                                     "Content-Range": f"bytes {done}-{last}/{size}"})
                done = last + 1
                progress.update(done)
                if r.status in (200, 201):
                    result = r.json()
        if not result or not result.get("id"):
            raise PublishError("YouTube didn't confirm the upload.")
        vid = result["id"]
        return {"id": vid, "url": f"https://youtube.com/shorts/{vid}", "status": "published",
                "note": "YouTube may take a few minutes to process the video."}


def _auth(account: dict) -> dict:
    return {"Authorization": f"Bearer {account['access_token']}"}
