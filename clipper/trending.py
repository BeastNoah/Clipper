"""Trending videos worth clipping, from the creators you clip for.

You keep a list of creators (YouTube channels and Twitch channels) in `<work>/creators.json`, with
whether each one allows clipping. clipper fetches their latest videos through yt-dlp (listings only,
nothing is downloaded) and ranks them by how much they're worth clipping right now: how fast they're
gathering views, how fresh they are and whether they're long enough to hold several good moments.

Nothing here can tell whether a creator really allows clipping: that's what you record per creator
(from their clipping program, their channel rules or by asking them). The two creators the list
starts with run public, paid clipping programs as reported at the time of writing; check their
current rules before posting.
"""

from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .util import ClipperError, fingerprint, log, read_json, write_json

CREATORS_FILE = "creators.json"
CACHE_FILE = ".trending.json"
CACHE_AGE = 3600          # seconds before a listing is fetched again
PER_CREATOR = 8           # latest videos looked at per creator
MAX_AGE_DAYS = 14         # older videos aren't trending any more
DETAIL_LOOKUPS = 4        # per creator: videos whose listing gave no date get a full lookup
CLIPPING = ("allowed", "program", "unknown", "no")
CLIPPING_LABELS = {"allowed": "clipping allowed", "program": "runs a clipping program",
                   "unknown": "permission unknown", "no": "no clipping"}
VERDICTS = ((70, "worth it"), (45, "maybe"), (0, "skip"))

# Creators with public clipping programs, as reported by creator-economy press when this was
# written. Edit the list freely (Creators in the web interface): it's yours.
DEFAULT_CREATORS = [
    {"name": "MrBeast", "url": "https://www.youtube.com/@MrBeast", "clipping": "program",
     "note": "Runs a paid clipper program (reported by Whop/Opus coverage). Check its current rules."},
    {"name": "N3on", "url": "https://www.youtube.com/@N3onTV", "clipping": "program",
     "note": "Paid clippers to cut his streams into shorts (reported by CreatorDB). Check the current terms."},
]


# ---------------------------------------------------------------------------------------------
# Creators

def creators_path(root: Path) -> Path:
    return root / CREATORS_FILE


def load_creators(root: Path) -> list[dict]:
    path = creators_path(root)
    if not path.exists():
        return [dict(c) for c in DEFAULT_CREATORS]
    data = read_json(path)
    return [clean_creator(c) for c in data] if isinstance(data, list) else []


def save_creators(root: Path, creators: list) -> list[dict]:
    if not isinstance(creators, list):
        raise ClipperError("creators must be a list")
    cleaned = [clean_creator(c) for c in creators]
    write_json(creators_path(root), cleaned)
    return cleaned


def clean_creator(c: dict) -> dict:
    if not isinstance(c, dict):
        raise ClipperError("each creator must be an object")
    url = str(c.get("url") or "").strip()
    platform = platform_of(url)
    if platform is None:
        raise ClipperError(f"'{url or c.get('name') or '?'}' isn't a YouTube or Twitch channel link")
    clipping = str(c.get("clipping") or "unknown")
    if clipping not in CLIPPING:
        raise ClipperError(f"clipping must be one of {', '.join(CLIPPING)}")
    name = str(c.get("name") or "").strip() or default_name(url)
    return {"name": name[:80], "url": url, "platform": platform, "clipping": clipping,
            "note": str(c.get("note") or "").strip()[:300]}


def platform_of(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower() if url.startswith(("http://", "https://")) else ""
    if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        return "youtube"
    if host in ("twitch.tv", "www.twitch.tv", "m.twitch.tv"):
        return "twitch"
    return None


def default_name(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return (path.split("/")[0] if path else urlparse(url).hostname or url).lstrip("@")


def listing_url(creator: dict) -> str:
    """The page that lists a creator's latest videos."""
    url = creator["url"].rstrip("/")
    if creator["platform"] == "twitch":
        login = urlparse(url).path.strip("/").split("/")[0]
        return f"https://www.twitch.tv/{login}/videos?filter=all&sort=time"
    path = urlparse(url).path.rstrip("/")
    if re.search(r"/(videos|streams|live|shorts|playlists)$", path) or "list=" in url:
        return url
    return url + "/videos"


# ---------------------------------------------------------------------------------------------
# Fetching (yt-dlp listings only)

def _yt_dlp():
    try:
        import yt_dlp
    except ImportError as e:
        raise ClipperError("yt-dlp is needed to look up creators: pip install yt-dlp") from e
    return yt_dlp


def _listing(url: str, limit: int) -> list[dict]:
    """The latest `limit` videos of a channel page, as yt-dlp's flat entries."""
    yt_dlp = _yt_dlp()
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "playlistend": limit,
            "skip_download": True,
            # YouTube listings only say "3 days ago"; this turns that into an approximate timestamp
            "extractor_args": {"youtubetab": {"approximate_date": ["1"]}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") if info.get("_type") == "playlist" else [info]
    return [e for e in (entries or []) if e]


def _details(url: str) -> dict:
    yt_dlp = _yt_dlp()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False) or {}


def _thumbnail(entry: dict) -> str | None:
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        best = max(thumbs, key=lambda t: (t.get("width") or 0, t.get("height") or 0))
        return best.get("url")
    return entry.get("thumbnail")


def video_from_entry(entry: dict, creator: dict) -> dict:
    url = entry.get("url") or entry.get("webpage_url") or ""
    if creator["platform"] == "youtube" and entry.get("id") and not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={entry['id']}"
    return {
        "id": str(entry.get("id") or url), "url": url, "title": entry.get("title") or url,
        "creator": creator["name"], "platform": creator["platform"], "clipping": creator["clipping"],
        "views": entry.get("view_count"), "likes": entry.get("like_count"),
        "duration": entry.get("duration"), "timestamp": entry.get("timestamp"),
        "live": entry.get("live_status") in ("is_live", "is_upcoming") or bool(entry.get("is_live")),
        "short": "/shorts/" in url or creator["platform"] == "youtube" and (entry.get("duration") or 999) < 60,
        "thumbnail": _thumbnail(entry),
    }


def fetch_creator(creator: dict, listing=None, details=None, now: float | None = None) -> list[dict]:
    """The creator's latest videos, with a date looked up individually where the listing had none."""
    listing, details = listing or _listing, details or _details
    now = time.time() if now is None else now
    videos = [video_from_entry(e, creator) for e in listing(listing_url(creator), PER_CREATOR)]
    videos = [v for v in videos if not v["live"] and not v["short"]]
    looked_up = 0
    for v in videos:
        if v["timestamp"] is not None or looked_up >= DETAIL_LOOKUPS:
            continue
        looked_up += 1
        try:
            d = details(v["url"])
        except Exception as e:  # noqa: BLE001 - one video's missing details shouldn't lose the creator
            log(f"trending: no details for {v['url']}: {e}")
            continue
        v["timestamp"] = d.get("timestamp")
        for k, src in (("views", "view_count"), ("likes", "like_count"), ("duration", "duration")):
            if d.get(src) is not None:
                v[k] = d[src]
        if not v["thumbnail"]:
            v["thumbnail"] = _thumbnail(d)
    out = []
    for v in videos:
        if v["timestamp"] is not None and now - v["timestamp"] > MAX_AGE_DAYS * 86400:
            continue
        v.update(score_video(v, now))
        out.append(v)
    return out


def fetch_all(creators: list[dict], listing=None, details=None, workers: int = 4) -> dict:
    """Every creator's trending videos, best first, plus the creators that couldn't be read."""
    now = time.time()
    videos, errors = [], []

    def one(creator: dict) -> None:
        try:
            videos.extend(fetch_creator(creator, listing, details, now))
        except Exception as e:  # noqa: BLE001 - a channel that's gone or renamed must not hide the others
            msg = re.sub(r"^ERROR: ", "", str(e)).split("\n")[0]
            errors.append({"creator": creator["name"], "error": msg[:300]})

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(creators) or 1))) as pool:
        list(pool.map(one, creators))
    videos.sort(key=lambda v: (-v["score"], -(v["views"] or 0)))
    return {"fetched": now, "videos": videos, "errors": sorted(errors, key=lambda e: e["creator"]),
            "creators": len(creators)}


# ---------------------------------------------------------------------------------------------
# Scoring: is this worth clipping right now?

def score_video(v: dict, now: float | None = None) -> dict:
    """A 0-100 score with the reasons behind it. Velocity (views per hour) counts most, then how
    fresh the video is and whether it's long enough to hold several good moments."""
    now = time.time() if now is None else now
    views = v.get("views")
    duration = v.get("duration")
    ts = v.get("timestamp")
    reasons, score = [], 0.0

    age_h = max(1.0, (now - ts) / 3600) if ts is not None else None
    if views is not None and age_h is not None:
        velocity = views / age_h
        score += min(45.0, 10.0 * math.log10(velocity + 1))
        reasons.append(f"{_count(velocity)} views/h")
    elif views is not None:
        score += min(25.0, 5.0 * math.log10(views + 1))
        reasons.append(f"{_count(views)} views, date unknown")
    else:
        reasons.append("no view count")

    if age_h is not None:
        score += 25 if age_h <= 24 else 18 if age_h <= 72 else 10 if age_h <= 24 * 7 else 4
        reasons.append(f"posted {_age(age_h)} ago")

    if duration:
        if 600 <= duration <= 4 * 3600:
            score += 20
            reasons.append(f"long-form ({_dur(duration)})")
        elif duration > 4 * 3600:
            score += 15
            reasons.append(f"very long ({_dur(duration)})")
        elif duration >= 240:
            score += 12
            reasons.append(f"mid-length ({_dur(duration)})")
        else:  # room for one clip at best
            score -= 15
            reasons.append(f"short video ({_dur(duration)})")

    likes = v.get("likes")
    if likes is not None and views:
        ratio = likes / views
        bonus = 10 if ratio >= 0.05 else 6 if ratio >= 0.03 else 3 if ratio >= 0.01 else 0
        score += bonus
        if bonus:
            reasons.append(f"{ratio * 100:.0f}% likes")

    score = int(round(min(100.0, max(0.0, score))))
    return {"score": score, "verdict": next(label for floor, label in VERDICTS if score >= floor), "reasons": reasons}


def _count(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}"


def _age(hours: float) -> str:
    return f"{int(hours)}h" if hours < 48 else f"{int(hours // 24)}d"


def _dur(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"


# ---------------------------------------------------------------------------------------------
# Cache, shared by the web interface and the command line

def cache_path(root: Path) -> Path:
    return root / CACHE_FILE


def cached(root: Path, creators: list[dict]) -> dict | None:
    """The last result, if it's recent and was made for this creator list."""
    path = cache_path(root)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except ValueError:
        return None
    if data.get("key") != fingerprint(creators) or time.time() - data.get("fetched", 0) > CACHE_AGE:
        return None
    return data


def trending(root: Path, refresh: bool = False, listing=None, details=None) -> dict:
    creators = load_creators(root)
    data = None if refresh else cached(root, creators)
    if data is None:
        data = fetch_all(creators, listing, details)
        data["key"] = fingerprint(creators)
        write_json(cache_path(root), data)
    return data


class TrendingLoader:
    """Loads in the background for the web interface, one fetch at a time."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def loading(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, refresh: bool = False) -> bool:
        with self.lock:
            if self.loading():
                return False
            self.error = None

            def run() -> None:
                try:
                    trending(self.root, refresh=refresh)
                except Exception as e:  # noqa: BLE001 - shown in the UI
                    self.error = str(e).split("\n")[0][:300]
            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()
            return True

    def state(self) -> dict:
        creators = load_creators(self.root)
        data = cached(self.root, creators)
        if data is None and not self.loading() and self.error is None:
            self.start()
        out = {"loading": self.loading(), "error": self.error, "creators": creators,
               "fetched": None, "videos": [], "errors": []}
        if data is not None:
            out.update({k: data[k] for k in ("fetched", "videos", "errors")})
        return out
