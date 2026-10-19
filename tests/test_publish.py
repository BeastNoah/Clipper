"""Publishing, against small stand-ins for the YouTube, TikTok and Instagram APIs."""
import hashlib
import json
import os
import stat
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from clipper import publish
from clipper.publish import instagram, oauth, store, tiktok, youtube
from clipper.publish.http import PublishError
from clipper.util import write_json


class FakeApi:
    """Routes (method, path) to canned answers; a list of answers is used one per call."""

    def __init__(self):
        self.routes, self.calls = {}, []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _any(self):
                url = urllib.parse.urlparse(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                api.calls.append({"method": self.command, "path": url.path,
                                  "query": dict(urllib.parse.parse_qsl(url.query)),
                                  "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                answer = api.routes.get((self.command, url.path))
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                if answer is None:
                    status, headers, data = 404, {}, {"error": {"message": f"no route {self.command} {url.path}"}}
                else:
                    status, headers, data = answer(api.calls[-1]) if callable(answer) else answer
                raw = data if isinstance(data, bytes) else json.dumps(data).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = do_PUT = _any

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def on(self, method, path, *answers):
        self.routes[(method, path)] = list(answers) if len(answers) > 1 else answers[0]

    def find(self, method, path):
        return [c for c in self.calls if c["method"] == method and c["path"] == path]


def ok(data, status=200, headers=None):
    return status, headers or {}, data


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_CONFIG", str(tmp_path / "config"))
    fake = FakeApi()
    monkeypatch.setattr(youtube, "TOKEN_URL", fake.url + "/google/token")
    monkeypatch.setattr(youtube, "API_URL", fake.url + "/youtube/v3")
    monkeypatch.setattr(youtube, "UPLOAD_URL", fake.url + "/upload/youtube/v3/videos")
    monkeypatch.setattr(tiktok, "API_URL", fake.url + "/tiktok/v2")
    monkeypatch.setattr(tiktok, "POLL_SECONDS", 0)
    monkeypatch.setattr(instagram, "GRAPH_URL", fake.url + "/graph")
    monkeypatch.setattr(instagram, "POLL_SECONDS", 0)
    yield fake
    fake.server.shutdown()


def make_clip(tmp_path, size=600_000, duration=30.0):
    work = tmp_path / "work" / "talk-1234abcd"
    (work / "output").mkdir(parents=True)
    video = work / "output" / "01_the-hook.mp4"
    video.write_bytes(os.urandom(size))
    clip = {"rank": 1, "id": "c1", "title": "The hook", "file": video.name, "start": 10.0, "end": 40.0,
            "final_duration": duration}
    write_json(work / "output" / "clips.json", [clip])
    return work, clip


def test_pkce_challenges():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"  # the example in RFC 7636, appendix B
    assert oauth.challenge_b64(verifier) == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    assert oauth.challenge_hex(verifier) == hashlib.sha256(verifier.encode()).hexdigest()
    v = oauth.code_verifier()
    assert 43 <= len(v) <= 128 and set(v) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def test_logins_are_single_use_and_per_platform():
    logins = oauth.Logins()
    state, _ = logins.start("youtube", "http://localhost:1/cb")
    assert logins.finish("tiktok", state) is None       # another platform can't use it
    state, _ = logins.start("youtube", "http://localhost:1/cb")
    assert logins.finish("youtube", state) is not None
    assert logins.finish("youtube", state) is None      # only once


def test_secrets_are_private(tmp_path):
    s = store.SecretStore("accounts.json", tmp_path / "cfg")
    s.put("youtube", {"access_token": "secret"})
    assert stat.S_IMODE(s.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(s.path.parent.stat().st_mode) == 0o700
    assert s.get("youtube") == {"access_token": "secret"}
    s.put("youtube", None)
    assert s.get("youtube") is None


def test_tiktok_chunk_plan_follows_the_rules():
    MB = 1024 * 1024
    assert tiktok.chunk_plan(3 * MB) == (3 * MB, 1)
    assert tiktok.chunk_plan(64 * MB) == (64 * MB, 1)
    for size in (64 * MB + 1, 100 * MB, 150_000_000, 900 * MB):
        chunk, count = tiktok.chunk_plan(size)
        last = size - chunk * (count - 1)
        assert 5 * MB <= chunk <= 64 * MB and 1 <= count <= 1000
        assert chunk <= last <= 128 * MB
    with pytest.raises(PublishError):
        tiktok.chunk_plan(0)


def test_status_and_app_setup(api):
    assert [p["configured"] for p in publish.status()] == [False, False, False]
    with pytest.raises(PublishError, match="client_secret"):
        publish.save_app("youtube", {"client_id": "id"})
    publish.save_app("youtube", {"client_id": " id ", "client_secret": "sec", "junk": "x"})
    assert store.apps().get("youtube") == {"client_id": "id", "client_secret": "sec"}
    yt = next(p for p in publish.status() if p["name"] == "youtube")
    assert yt["configured"] and not yt["connected"]
    with pytest.raises(PublishError, match="isn't set up"):
        publish.connect_url("tiktok", "http://localhost:1/cb", publish.Logins())


def test_youtube_connect_refresh_and_publish(api, tmp_path, monkeypatch):
    publish.save_app("youtube", {"client_id": "cid", "client_secret": "csec"})
    logins = publish.Logins()
    redirect = "http://localhost:8765/oauth/youtube/callback"
    url = urllib.parse.urlparse(publish.connect_url("youtube", redirect, logins))
    q = dict(urllib.parse.parse_qsl(url.query))
    assert url.netloc == "accounts.google.com" and q["client_id"] == "cid" and q["redirect_uri"] == redirect
    assert q["code_challenge_method"] == "S256" and q["access_type"] == "offline"

    api.on("POST", "/google/token", ok({"access_token": "at1", "refresh_token": "rt", "expires_in": 3600}),
           ok({"access_token": "at2", "expires_in": 3600}))
    api.on("GET", "/youtube/v3/channels", ok({"items": [{"id": "UC1", "snippet": {"title": "Noah Clips"}}]}))
    with pytest.raises(PublishError, match="unknown or too old"):
        publish.finish_connect("youtube", "forged-state", "code", logins)
    assert publish.finish_connect("youtube", q["state"], "the-code", logins) == "Noah Clips"
    token_call = api.find("POST", "/google/token")[0]
    form = dict(urllib.parse.parse_qsl(token_call["body"].decode()))
    pending = len(oauth.code_verifier())  # the verifier went along with the code
    assert form["code"] == "the-code" and form["redirect_uri"] == redirect and len(form["code_verifier"]) == pending
    assert api.find("GET", "/youtube/v3/channels")[0]["headers"]["authorization"] == "Bearer at1"

    # an expired token is refreshed (and saved) before posting
    acct = store.accounts().get("youtube")
    store.accounts().put("youtube", {**acct, "expires_at": time.time() - 10})

    session = api.url + "/upload/session/1"
    api.on("POST", "/upload/youtube/v3/videos", ok({}, headers={"Location": session}))
    monkeypatch.setattr(youtube, "CHUNK", 256 * 1024)
    api.on("PUT", "/upload/session/1", ok({}, 308), ok({}, 308), ok({"id": "vid123"}, 200))
    work, clip = make_clip(tmp_path, size=600_000)
    with pytest.raises(PublishError, match="who can see"):
        publish.publish_clip(work, clip, "youtube", {"title": "Hi", "made_for_kids": False})
    rec = publish.publish_clip(work, clip, "youtube", {"title": "The <hook>", "description": "Watch",
                                                       "privacy": "unlisted", "made_for_kids": False})
    assert rec["url"] == "https://youtube.com/shorts/vid123" and rec["account"] == "Noah Clips"
    assert store.accounts().get("youtube")["access_token"] == "at2"
    start = api.find("POST", "/upload/youtube/v3/videos")[-1]
    meta = json.loads(start["body"])
    assert meta["snippet"]["title"] == "The hook" and meta["status"] == {"privacyStatus": "unlisted",
                                                                        "selfDeclaredMadeForKids": False}
    assert start["headers"]["x-upload-content-length"] == "600000" and start["headers"]["authorization"] == "Bearer at2"
    puts = api.find("PUT", "/upload/session/1")
    assert [p["headers"]["content-range"] for p in puts] == [
        "bytes 0-262143/600000", "bytes 262144-524287/600000", "bytes 524288-599999/600000"]
    assert b"".join(p["body"] for p in puts) == (work / "output" / clip["file"]).read_bytes()
    assert publish.published(work)[clip["file"]]["youtube"]["id"] == "vid123"
    with pytest.raises(PublishError, match="already posted"):
        publish.publish_clip(work, clip, "youtube", {"title": "x", "privacy": "public", "made_for_kids": False})


def creator_info(**over):
    data = {"creator_nickname": "noah", "creator_username": "noah.clips",
            "privacy_level_options": ["SELF_ONLY"], "comment_disabled": False, "duet_disabled": True,
            "stitch_disabled": False, "max_video_post_duration_sec": 60, **over}
    return ok({"data": data, "error": {"code": "ok"}})


def test_tiktok_rules_and_publish(api, tmp_path):
    publish.save_app("tiktok", {"client_key": "ck", "client_secret": "cs"})
    store.accounts().put("tiktok", {"access_token": "tt", "expires_at": time.time() + 3600, "name": "noah"})
    api.on("POST", "/tiktok/v2/post/publish/creator_info/query/", creator_info())
    plat = tiktok.TikTok(store.apps().get("tiktok"))
    info = plat.options(store.accounts().get("tiktok"))
    assert info["account"] == "noah" and info["privacy_options"] == ["SELF_ONLY"]

    good = {"title": "My clip #fyp", "privacy": "SELF_ONLY", "consent": True,
            "allow_comment": True, "allow_duet": True, "allow_stitch": False}
    for bad, msg in [({**good, "consent": False}, "Music Usage"),
                     ({**good, "privacy": "PUBLIC_TO_EVERYONE"}, "who can see"),
                     ({**good, "privacy": None}, "who can see"),
                     ({**good, "commercial": True}, "promotes"),
                     ({**good, "commercial": True, "branded_content": True}, "can't be private")]:
        with pytest.raises(PublishError, match=msg):
            plat.check(info, bad, 30)
    with pytest.raises(PublishError, match="up to 60s"):
        plat.check(info, good, 75)
    post = plat.check(info, good, 30)
    assert post["disable_comment"] is False
    assert post["disable_duet"] is True      # turned off in the creator's settings, so it stays off
    assert post["disable_stitch"] is True    # not chosen: off by default
    assert post["brand_content_toggle"] is False and post["brand_organic_toggle"] is False

    upload = api.url + "/tiktok-upload/abc"
    api.on("POST", "/tiktok/v2/post/publish/video/init/",
           ok({"data": {"publish_id": "p_1", "upload_url": upload}, "error": {"code": "ok"}}))
    api.on("PUT", "/tiktok-upload/abc", ok({}, 201))
    api.on("POST", "/tiktok/v2/post/publish/status/fetch/",
           ok({"data": {"status": "PROCESSING_UPLOAD"}, "error": {"code": "ok"}}),
           ok({"data": {"status": "PUBLISH_COMPLETE"}, "error": {"code": "ok"}}))
    work, clip = make_clip(tmp_path, size=700_000)
    rec = publish.publish_clip(work, clip, "tiktok", good)
    assert rec["status"] == "publish_complete" and "only to you" in rec["note"]
    init = json.loads(api.find("POST", "/tiktok/v2/post/publish/video/init/")[0]["body"])
    assert init["source_info"] == {"source": "FILE_UPLOAD", "video_size": 700_000, "chunk_size": 700_000,
                                   "total_chunk_count": 1}
    assert init["post_info"]["privacy_level"] == "SELF_ONLY"
    put = api.find("PUT", "/tiktok-upload/abc")[0]
    assert put["headers"]["content-range"] == "bytes 0-699999/700000" and len(put["body"]) == 700_000
    assert len(api.find("POST", "/tiktok/v2/post/publish/creator_info/query/")) == 2  # fetched again to post

    api.on("POST", "/tiktok/v2/post/publish/status/fetch/",
           ok({"data": {"status": "FAILED", "fail_reason": "spam_risk"}, "error": {"code": "ok"}}))
    with pytest.raises(PublishError, match="spam_risk"):
        publish.publish_clip(work, clip, "tiktok", good, again=True)


def test_tiktok_api_errors_are_reported(api):
    publish.save_app("tiktok", {"client_key": "ck", "client_secret": "cs"})
    store.accounts().put("tiktok", {"access_token": "tt", "expires_at": time.time() + 3600})
    api.on("POST", "/tiktok/v2/post/publish/creator_info/query/",
           ok({"data": {}, "error": {"code": "spam_risk_too_many_posts", "message": "Too many posts today"}}))
    with pytest.raises(PublishError, match="Too many posts today"):
        publish.options("tiktok")


def test_instagram_connect_and_publish(api, tmp_path):
    publish.save_app("instagram", {"app_id": "aid", "app_secret": "asec"})
    logins = publish.Logins()
    url = urllib.parse.urlparse(publish.connect_url("instagram", "http://localhost:8765/oauth/instagram/callback", logins))
    q = dict(urllib.parse.parse_qsl(url.query))
    assert "instagram_content_publish" in q["scope"]
    api.on("GET", "/graph/oauth/access_token", ok({"access_token": "short"}),
           ok({"access_token": "long", "expires_in": 5_000_000}))
    api.on("GET", "/graph/me/accounts", ok({"data": [
        {"name": "Plain page"},
        {"name": "Noah Page", "instagram_business_account": {"id": "1784", "username": "noah.clips"}}]}))
    assert publish.finish_connect("instagram", q["state"], "code", logins) == "@noah.clips"
    assert store.accounts().get("instagram")["ig_user_id"] == "1784"

    rupload = api.url + "/rupload/ig-api-upload/v24.0/cont1"
    api.on("POST", "/graph/1784/media", ok({"id": "cont1", "uri": rupload}))
    api.on("POST", "/rupload/ig-api-upload/v24.0/cont1", ok({"success": True, "message": "Upload successful."}))
    api.on("GET", "/graph/cont1", ok({"status_code": "IN_PROGRESS"}), ok({"status_code": "FINISHED"}))
    api.on("POST", "/graph/1784/media_publish", ok({"id": "media9"}))
    api.on("GET", "/graph/media9", ok({"permalink": "https://www.instagram.com/reel/xyz/"}))
    work, clip = make_clip(tmp_path, size=400_000)
    rec = publish.publish_clip(work, clip, "instagram", {"caption": "New reel #clips"})
    assert rec["url"] == "https://www.instagram.com/reel/xyz/"
    create = dict(urllib.parse.parse_qsl(api.find("POST", "/graph/1784/media")[0]["body"].decode()))
    assert create == {"media_type": "REELS", "upload_type": "resumable", "caption": "New reel #clips",
                      "share_to_feed": "true"}
    up = api.find("POST", "/rupload/ig-api-upload/v24.0/cont1")[0]
    assert up["headers"]["authorization"] == "OAuth long" and up["headers"]["offset"] == "0"
    assert up["headers"]["file_size"] == "400000" and len(up["body"]) == 400_000
    assert dict(urllib.parse.parse_qsl(api.find("POST", "/graph/1784/media_publish")[0]["body"].decode())) == {
        "creation_id": "cont1"}

    api.on("GET", "/graph/cont1", ok({"status_code": "ERROR", "status": "Unsupported video"}))
    with pytest.raises(PublishError, match="Unsupported video"):
        publish.publish_clip(work, clip, "instagram", {"caption": "again"}, again=True)


def test_instagram_needs_a_linked_professional_account(api):
    publish.save_app("instagram", {"app_id": "aid", "app_secret": "asec"})
    logins = publish.Logins()
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(
        publish.connect_url("instagram", "http://localhost:1/cb", logins)).query))
    api.on("GET", "/graph/oauth/access_token", ok({"access_token": "short"}))
    api.on("GET", "/graph/me/accounts", ok({"data": [{"name": "Plain page"}]}))
    with pytest.raises(PublishError, match="professional account"):
        publish.finish_connect("instagram", q["state"], "code", logins)
    assert store.accounts().get("instagram") is None
