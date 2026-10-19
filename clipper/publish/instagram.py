"""Instagram Reels through the Instagram Graph API with Facebook Login (the variant that accepts a
file uploaded straight from this computer). Needs a professional Instagram account linked to a
Facebook Page."""

from __future__ import annotations

import os
import time
import urllib.parse
from pathlib import Path

from ..util import Progress, check_cancel
from .base import Platform, clean_text, expires_at
from .http import PublishError, request

GRAPH_VERSION = os.environ.get("CLIPPER_META_GRAPH_VERSION", "v24.0")
DIALOG_URL = f"https://www.facebook.com/{GRAPH_VERSION}/dialog/oauth"
GRAPH_URL = f"https://graph.facebook.com/{GRAPH_VERSION}"
SCOPES = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,business_management"
POLL_SECONDS, POLL_LIMIT = 10, 10 * 60


class Instagram(Platform):
    name, label = "instagram", "Instagram Reels"
    app_fields = ("app_id", "app_secret")
    setup_url = "https://developers.facebook.com/apps"

    def authorize_url(self, redirect_uri: str, state: str, verifier: str) -> str:
        params = {"client_id": self.app["app_id"], "redirect_uri": redirect_uri, "state": state,
                  "response_type": "code"}
        if self.app.get("config_id"):  # Facebook Login for Business configuration, if you made one
            params["config_id"] = self.app["config_id"]
        else:
            params["scope"] = SCOPES
        return DIALOG_URL + "?" + urllib.parse.urlencode(params)

    def finish_login(self, code: str, redirect_uri: str, verifier: str) -> dict:
        short = request("GET", f"{GRAPH_URL}/oauth/access_token", platform=self.label, params={
            "client_id": self.app["app_id"], "client_secret": self.app["app_secret"],
            "redirect_uri": redirect_uri, "code": code}).json()
        long = request("GET", f"{GRAPH_URL}/oauth/access_token", platform=self.label, params={
            "grant_type": "fb_exchange_token", "client_id": self.app["app_id"],
            "client_secret": self.app["app_secret"], "fb_exchange_token": short["access_token"]}).json()
        token = long.get("access_token") or short["access_token"]
        pages = request("GET", f"{GRAPH_URL}/me/accounts", platform=self.label, params={
            "fields": "name,instagram_business_account{id,username}", "access_token": token}).json()
        linked = [p for p in pages.get("data", []) if p.get("instagram_business_account")]
        if not linked:
            raise PublishError("No Instagram professional account is linked to your Facebook Pages. Switch "
                               "Instagram to a Business or Creator account and link it to a Page, then connect again.")
        ig = linked[0]["instagram_business_account"]
        return {"access_token": token, "expires_at": expires_at(long.get("expires_in")),
                "ig_user_id": ig["id"], "name": "@" + (ig.get("username") or ig["id"]),
                "page": linked[0].get("name")}

    def options(self, account: dict) -> dict:
        return {"account": account.get("name")}

    def publish(self, account: dict, video: Path, form: dict, duration: float) -> dict:
        ig, token = account["ig_user_id"], account["access_token"]
        auth = {"Authorization": f"OAuth {token}"}
        container = request("POST", f"{GRAPH_URL}/{ig}/media", platform=self.label, headers=auth, form={
            "media_type": "REELS", "upload_type": "resumable",
            "caption": clean_text(form.get("caption"), 2200),
            "share_to_feed": "true" if form.get("share_to_feed", True) else "false",
        }).json()
        cid = container.get("id")
        if not cid:
            raise PublishError("Instagram didn't create the upload.")
        upload_uri = container.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{GRAPH_VERSION}/{cid}"
        size = video.stat().st_size
        progress = Progress("Uploading to Instagram", size)
        r = request("POST", upload_uri, data=video.read_bytes(), platform=self.label, timeout=900,
                    headers={**auth, "offset": "0", "file_size": str(size)}).json()
        if r.get("success") is False:
            raise PublishError(f"Instagram upload failed: {r.get('message') or r}")
        progress.update(size)
        self._wait(cid, auth)
        media = request("POST", f"{GRAPH_URL}/{ig}/media_publish", platform=self.label, headers=auth,
                        form={"creation_id": cid}).json()
        mid = media.get("id")
        if not mid:
            raise PublishError("Instagram didn't publish the reel.")
        link = request("GET", f"{GRAPH_URL}/{mid}", platform=self.label, headers=auth,
                       params={"fields": "permalink"}).json().get("permalink")
        return {"id": mid, "url": link, "status": "published", "note": "Posted."}

    def _wait(self, cid: str, auth: dict) -> None:
        """Instagram processes the video before it can be published."""
        deadline = time.time() + POLL_LIMIT
        while True:
            status = request("GET", f"{GRAPH_URL}/{cid}", platform=self.label, headers=auth,
                             params={"fields": "status_code,status"}).json()
            code = status.get("status_code")
            if code == "FINISHED":
                return
            if code in ("ERROR", "EXPIRED"):
                raise PublishError(f"Instagram couldn't process the video: {status.get('status') or code}")
            if time.time() > deadline:
                raise PublishError("Instagram is taking too long to process the video; try again later.")
            check_cancel()
            time.sleep(POLL_SECONDS)
