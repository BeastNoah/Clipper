"""Trending videos: creators, scoring, fetching through fake listings, and the web API."""
import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from clipper import trending
from clipper.cli import main
from clipper.trending import (
    clean_creator, fetch_all, listing_url, load_creators, save_creators, score_video, trending as fetch_trending,
)
from clipper.ui.server import App, Handler
from clipper.util import ClipperError

NOW = 1_800_000_000.0
H = 3600

CREATORS = [
    {"name": "Podcaster", "url": "https://www.youtube.com/@podcaster", "clipping": "program", "note": "pays clippers"},
    {"name": "Streamer", "url": "https://www.twitch.tv/streamer", "clipping": "allowed"},
    {"name": "Mystery", "url": "https://youtube.com/channel/UCxyz/videos", "clipping": "unknown"},
    {"name": "Gone", "url": "https://www.youtube.com/@gone", "clipping": "allowed"},
]

LISTINGS = {
    "https://www.youtube.com/@podcaster/videos": [
        {"id": "hot", "url": "https://www.youtube.com/watch?v=hot", "title": "Big interview", "view_count": 400_000,
         "duration": 5400, "timestamp": NOW - 20 * H, "thumbnails": [{"url": "t1", "width": 120}, {"url": "t2", "width": 480}]},
        {"id": "old", "url": "https://www.youtube.com/watch?v=old", "title": "Old talk", "view_count": 2_000_000,
         "duration": 3000, "timestamp": NOW - 40 * 86400},
        {"id": "sh", "url": "https://www.youtube.com/shorts/sh", "title": "A short", "view_count": 9_000_000, "duration": 30,
         "timestamp": NOW - H},
        {"id": "live", "url": "https://www.youtube.com/watch?v=live", "title": "Live now", "live_status": "is_live",
         "concurrent_view_count": 50_000},
        {"id": "trailer", "url": "https://www.youtube.com/watch?v=trailer", "title": "Trailer", "view_count": 300_000,
         "duration": 150, "timestamp": NOW - 30 * H},
    ],
    "https://www.twitch.tv/streamer/videos?filter=all&sort=time": [
        {"id": "v1", "url": "https://www.twitch.tv/videos/1", "title": "Ranked grind", "view_count": 30_000, "duration": 4 * H},
        {"id": "v2", "url": "https://www.twitch.tv/videos/2", "title": "Chill stream", "view_count": 3_000, "duration": 2 * H},
    ],
    "https://youtube.com/channel/UCxyz/videos": [
        {"id": "m1", "url": "https://www.youtube.com/watch?v=m1", "title": "Mystery vlog", "view_count": 80_000,
         "duration": 900, "timestamp": NOW - 5 * H, "like_count": 6_000},
    ],
}

DETAILS = {
    "https://www.twitch.tv/videos/1": {"timestamp": NOW - 10 * H, "view_count": 35_000, "thumbnail": "tw1"},
    "https://www.twitch.tv/videos/2": {"timestamp": NOW - 6 * 86400},
}


def fake_listing(url, limit):
    if url not in LISTINGS:
        raise RuntimeError("ERROR: [youtube:tab] gone: This channel does not exist.")
    return LISTINGS[url][:limit]


def fake_details(url):
    return DETAILS[url]


def test_creators_are_checked_and_saved(tmp_path):
    assert load_creators(tmp_path) == trending.DEFAULT_CREATORS  # the starting list, until you save your own
    saved = save_creators(tmp_path, CREATORS)
    assert [c["platform"] for c in saved] == ["youtube", "twitch", "youtube", "youtube"]
    assert load_creators(tmp_path) == saved
    assert save_creators(tmp_path, []) == [] and load_creators(tmp_path) == []
    assert clean_creator({"url": "https://www.twitch.tv/someone"})["name"] == "someone"
    assert clean_creator({"url": "https://www.youtube.com/@Handle/"})["name"] == "Handle"
    for bad in ({"url": "https://vimeo.com/x"}, {"url": "not a url"}, {"url": "https://www.youtube.com/@x", "clipping": "maybe"}, "text"):
        with pytest.raises(ClipperError):
            clean_creator(bad)
    with pytest.raises(ClipperError):
        save_creators(tmp_path, {"name": "x"})
    assert listing_url(clean_creator({"url": "https://www.youtube.com/@a"})) == "https://www.youtube.com/@a/videos"
    assert listing_url(clean_creator({"url": "https://www.youtube.com/@a/streams"})) == "https://www.youtube.com/@a/streams"
    assert listing_url(clean_creator({"url": "https://twitch.tv/b/videos?filter=clips"})) == "https://www.twitch.tv/b/videos?filter=all&sort=time"


def test_scoring_prefers_fresh_fast_long_videos():
    hot = score_video({"views": 400_000, "duration": 5400, "timestamp": NOW - 20 * H}, NOW)
    slow = score_video({"views": 400_000, "duration": 5400, "timestamp": NOW - 10 * 86400}, NOW)
    short = score_video({"views": 400_000, "duration": 150, "timestamp": NOW - 20 * H}, NOW)
    undated = score_video({"views": 400_000, "duration": 5400, "timestamp": None}, NOW)
    nothing = score_video({"views": None, "duration": None, "timestamp": None}, NOW)
    assert hot["score"] > slow["score"] > short["score"]
    assert hot["verdict"] == "worth it" and hot["score"] <= 100
    assert hot["reasons"] == ["20.0k views/h", "posted 20h ago", "long-form (1h 30m)"]
    assert "date unknown" in undated["reasons"][0] and undated["score"] < hot["score"]
    assert nothing == {"score": 0, "verdict": "skip", "reasons": ["no view count"]}
    liked = score_video({"views": 80_000, "likes": 6_000, "duration": 900, "timestamp": NOW - 5 * H}, NOW)
    assert "8% likes" in liked["reasons"] and liked["score"] == min(100, score_video({"views": 80_000, "duration": 900, "timestamp": NOW - 5 * H}, NOW)["score"] + 10)


def test_fetch_all_ranks_and_filters(monkeypatch):
    monkeypatch.setattr(trending, "time", _FrozenTime)
    creators = [clean_creator(c) for c in CREATORS]
    data = fetch_all(creators, fake_listing, fake_details)
    titles = [v["title"] for v in data["videos"]]
    # shorts, live streams and videos older than two weeks are left out; the rest best first
    assert "A short" not in titles and "Live now" not in titles and "Old talk" not in titles
    assert set(titles) == {"Big interview", "Trailer", "Ranked grind", "Chill stream", "Mystery vlog"}
    assert titles.index("Big interview") < titles.index("Trailer") and titles.index("Ranked grind") < titles.index("Chill stream")
    assert all(data["videos"][i]["score"] >= data["videos"][i + 1]["score"] for i in range(len(titles) - 1))
    by_title = {v["title"]: v for v in data["videos"]}
    assert by_title["Big interview"]["thumbnail"] == "t2" and by_title["Big interview"]["clipping"] == "program"
    grind = by_title["Ranked grind"]  # Twitch listings carry no date: looked up per video
    assert grind["timestamp"] == NOW - 10 * H and grind["views"] == 35_000 and grind["thumbnail"] == "tw1"
    assert grind["platform"] == "twitch" and grind["creator"] == "Streamer"
    assert data["errors"] == [{"creator": "Gone", "error": "[youtube:tab] gone: This channel does not exist."}]
    assert data["creators"] == 4


def test_trending_is_cached_per_creator_list(tmp_path, monkeypatch):
    monkeypatch.setattr(trending, "time", _FrozenTime)
    calls = []

    def counting(url, limit):
        calls.append(url)
        return fake_listing(url, limit)
    save_creators(tmp_path, CREATORS[:1])
    first = fetch_trending(tmp_path, listing=counting, details=fake_details)
    again = fetch_trending(tmp_path, listing=counting, details=fake_details)
    assert first["videos"] == again["videos"] and len(calls) == 1
    save_creators(tmp_path, CREATORS[:2])  # a changed list is looked up again
    fetch_trending(tmp_path, listing=counting, details=fake_details)
    assert len(calls) == 3
    fetch_trending(tmp_path, refresh=True, listing=counting, details=fake_details)
    assert len(calls) == 5


def test_cli_trending(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trending, "time", _FrozenTime)
    monkeypatch.setattr(trending, "_listing", fake_listing)
    monkeypatch.setattr(trending, "_details", fake_details)
    root = tmp_path / "work"
    root.mkdir()
    save_creators(root, CREATORS)
    assert main(["trending", "--work-dir", str(root)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[") and "Big interview" in out and "Mystery vlog" not in out
    assert main(["trending", "--work-dir", str(root), "--all"]) == 0
    assert "Mystery vlog" in capsys.readouterr().out
    save_creators(root, [])
    assert main(["trending", "--work-dir", str(root), "--refresh"]) == 0
    assert "Nothing to show" in capsys.readouterr().out


class _FrozenTime:
    """A stand-in for the time module with a fixed clock."""
    monotonic = staticmethod(time.monotonic)

    @staticmethod
    def time():
        return NOW


def call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
    r = conn.getresponse()
    return r.status, json.loads(r.read())


def test_trending_api(tmp_path, monkeypatch):
    monkeypatch.setattr(trending, "time", _FrozenTime)
    monkeypatch.setattr(trending, "_listing", fake_listing)
    monkeypatch.setattr(trending, "_details", fake_details)
    root = tmp_path / "work"
    root.mkdir()
    app = App(root)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"app": app}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        status, creators = call(port, "GET", "/api/creators")
        assert status == 200 and creators == trending.DEFAULT_CREATORS
        status, body = call(port, "POST", "/api/creators", {"creators": [{"url": "https://vimeo.com/nope"}]})
        assert status == 400 and "YouTube or Twitch" in body["error"]
        status, saved = call(port, "POST", "/api/creators", {"creators": CREATORS})
        assert status == 200 and [c["name"] for c in saved] == ["Podcaster", "Streamer", "Mystery", "Gone"]
        data = wait_loaded(port)
        assert {v["title"] for v in data["videos"][:2]} == {"Mystery vlog", "Big interview"} and data["videos"][0]["verdict"] == "worth it"
        assert [c["name"] for c in data["creators"]] == ["Podcaster", "Streamer", "Mystery", "Gone"]
        assert data["errors"][0]["creator"] == "Gone" and data["fetched"] == NOW
        status, data = call(port, "POST", "/api/trending/refresh", {})
        assert status == 200
        assert len(wait_loaded(port)["videos"]) == 5
        # with nobody to look up there's nothing to show, and no lookup either
        call(port, "POST", "/api/creators", {"creators": []})
        data = wait_loaded(port)
        assert data["videos"] == [] and data["creators"] == [] and data["error"] is None
    finally:
        httpd.shutdown()


def wait_loaded(port):
    for _ in range(100):
        status, data = call(port, "GET", "/api/trending")
        assert status == 200
        if not data["loading"] and data["fetched"] is not None:
            return data
        time.sleep(0.05)
    raise AssertionError(f"still loading: {data}")
