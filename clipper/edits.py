"""Your own changes to a clip, on top of what Claude and clipper proposed.

The web editor shows Claude's proposal (the cut, the keywords, the jump cuts, the framing) as the
starting point and saves what you change here, in work/<video>/edits/<clip id>.json. The render
applies the changes; clearing them brings the proposal back. All times are seconds on the video's
own timeline (the transcript's), so the changes stay valid when a clip's parts move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .edit import MODES as JUMP_CUT_MODES
from .reframe import LAYOUTS
from .util import ClipperError, read_json, write_json

VERSION = 1
SHOT_LAYOUTS = LAYOUTS                     # per segment: auto | follow | split | main | fit (whole frame)
TEXT_POSITIONS = ("top", "middle", "bottom")
TRANSITIONS = ("cut", "fade")
MAX_TEXT = 120
MAX_ITEMS = 60


@dataclass
class ClipEdits:
    title: str | None = None                 # your own title (Claude's stays in clips.json)
    title_overlay: bool | None = None        # show the title at the start; None: as the options say
    keywords: list[str] | None = None        # words to highlight (and punch in on); None: Claude's
    jump_cuts: str | None = None             # off | pauses | on for this clip; None: as the options say
    layout: str | None = None                # the default framing of this clip; None: as the options say
    cuts_removed: list[list[float]] = field(default_factory=list)  # stretches you cut out yourself
    cuts_kept: list[list[float]] = field(default_factory=list)     # pauses clipper would cut that you keep
    layouts: list[dict] = field(default_factory=list)   # {"start", "end", "layout"}: framing per segment
    zoom: str = "auto"                       # auto (hook and keywords) | off
    zooms: list[list[float]] = field(default_factory=list)  # extra punch-ins
    texts: list[dict] = field(default_factory=list)     # {"text", "start", "end", "position"}
    broll: list[dict] = field(default_factory=list)     # {"file", "start", "end"}: a picture or video over the clip
    music: dict | None = None                # {"file", "volume"}: a music track under the sound
    transition: str = "cut"                  # between the clip's parts: cut | fade (dip to black)
    fade_in: bool = False
    fade_out: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "ClipEdits":
        data = data or {}
        if not isinstance(data, dict):
            raise ClipperError("edits must be an object")
        ed = cls()
        ed.title = _text(data.get("title"), 80, "title")
        ed.title_overlay = None if data.get("title_overlay") is None else bool(data["title_overlay"])
        if data.get("keywords") is not None:
            if not isinstance(data["keywords"], list):
                raise ClipperError("keywords must be a list")
            ed.keywords = [k for k in (_text(k, 60, "keyword") for k in data["keywords"][:20]) if k]
        ed.jump_cuts = _choice(data.get("jump_cuts"), JUMP_CUT_MODES, "jump cuts")
        ed.layout = _choice(data.get("layout"), LAYOUTS, "layout")
        ed.cuts_removed = _ranges(data.get("cuts_removed"), "cuts")
        ed.cuts_kept = _ranges(data.get("cuts_kept"), "kept pauses")
        ed.layouts = [{"start": a, "end": b, "layout": _choice(x.get("layout"), SHOT_LAYOUTS, "layout") or "auto"}
                      for x, (a, b) in _items(data.get("layouts"), "layouts")]
        ed.zoom = _choice(data.get("zoom"), ("auto", "off"), "zoom") or "auto"
        ed.zooms = _ranges(data.get("zooms"), "zooms")
        ed.texts = [{"text": _text(x.get("text"), MAX_TEXT, "text") or "", "start": a, "end": b,
                     "position": _choice(x.get("position"), TEXT_POSITIONS, "text position") or "top"}
                    for x, (a, b) in _items(data.get("texts"), "texts")]
        ed.texts = [t for t in ed.texts if t["text"]]
        ed.broll = [{"file": _file(x.get("file")), "start": a, "end": b}
                    for x, (a, b) in _items(data.get("broll"), "b-roll")]
        music = data.get("music")
        if music:
            if not isinstance(music, dict):
                raise ClipperError("music must be an object")
            volume = music.get("volume", 20)
            try:
                volume = int(volume)
            except (TypeError, ValueError):
                raise ClipperError("music volume must be a number") from None
            ed.music = {"file": _file(music.get("file")), "volume": min(100, max(0, volume))}
        ed.transition = _choice(data.get("transition"), TRANSITIONS, "transition") or "cut"
        ed.fade_in, ed.fade_out = bool(data.get("fade_in")), bool(data.get("fade_out"))
        return ed

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return self.to_dict() == ClipEdits().to_dict()

    def in_part(self, ranges: list[list[float]], start: float, end: float) -> list[tuple[float, float]]:
        """The parts of these ranges inside [start, end], in seconds from `start`."""
        out = []
        for a, b in ranges:
            a, b = max(a, start), min(b, end)
            if b > a:
                out.append((round(a - start, 3), round(b - start, 3)))
        return out

    def layouts_in_part(self, start: float, end: float) -> list[tuple[float, float, str]]:
        out = []
        for item in self.layouts:
            if item["layout"] == "auto":
                continue
            a, b = max(item["start"], start), min(item["end"], end)
            if b > a:
                out.append((round(a - start, 3), round(b - start, 3), item["layout"]))
        return out


# ---------------------------------------------------------------------------------------------

def _text(value, limit: int, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ClipperError(f"{what} must be text")
    value = " ".join(value.split())
    if len(value) > limit:
        raise ClipperError(f"keep the {what} to {limit} characters")
    return value or None


def _choice(value, allowed: tuple, what: str) -> str | None:
    if value is None or value == "":
        return None
    if value not in allowed:
        raise ClipperError(f"{what} must be one of {', '.join(allowed)}")
    return value


def _range(value, what: str) -> tuple[float, float]:
    try:
        a, b = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ClipperError(f"{what}: each range needs a start and an end time") from None
    if not (a >= 0 and b > a and b - a < 36000):
        raise ClipperError(f"{what}: a range must end after it starts")
    return round(a, 3), round(b, 3)


def _ranges(value, what: str) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ClipperError(f"{what} must be a list of at most {MAX_ITEMS} ranges")
    return sorted([list(_range(v, what)) for v in value])


def _items(value, what: str) -> list[tuple[dict, tuple[float, float]]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ClipperError(f"{what} must be a list of at most {MAX_ITEMS} items")
    out = []
    for item in value:
        if not isinstance(item, dict):
            raise ClipperError(f"{what}: each item must be an object")
        out.append((item, _range((item.get("start"), item.get("end")), what)))
    return sorted(out, key=lambda x: x[1])


def _file(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClipperError("choose an uploaded file")
    return value


# ---------------------------------------------------------------------------------------------
# Storage

def edits_path(work: Path, clip: dict) -> Path:
    return work / "edits" / f"{clip['id']}.json"


def load_edits(work: Path, clip: dict) -> ClipEdits:
    path = edits_path(work, clip)
    if not path.exists():
        return ClipEdits()
    try:
        return ClipEdits.from_dict(read_json(path).get("edits"))
    except (ClipperError, ValueError, AttributeError):
        return ClipEdits()


def save_edits(work: Path, clip: dict, edits: ClipEdits) -> None:
    path = edits_path(work, clip)
    if edits.is_empty:
        path.unlink(missing_ok=True)
        return
    write_json(path, {"version": VERSION, "edits": edits.to_dict()})


def check_assets(edits: ClipEdits, roots: list[Path]) -> None:
    """Pictures, videos and music must be files clipper knows about (uploads), not any file on the disk."""
    files = [b["file"] for b in edits.broll] + ([edits.music["file"]] if edits.music else [])
    for name in files:
        path = Path(name).expanduser()
        if not path.is_file():
            raise ClipperError(f"no such file: {name}")
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root.resolve()) for root in roots):
            raise ClipperError(f"{path.name} isn't an uploaded file; upload it under Upload first")


# ---------------------------------------------------------------------------------------------
# Applying the changes

def apply_cuts(keep: list[tuple[float, float]], removed: list[tuple[float, float]],
               kept: list[tuple[float, float]], duration: float, fps: float) -> list[tuple[float, float]]:
    """Segments to keep after your own changes: `kept` stretches play even where clipper would cut, then
    `removed` stretches are taken out. Times in seconds from the clip's start, on frame boundaries."""
    grid = lambda t: round(t * fps) / fps  # noqa: E731
    segs = [(a, b) for a, b in keep]
    for a, b in kept:
        segs.append((grid(max(0.0, a)), grid(min(duration, b))))
    merged: list[list[float]] = []
    for a, b in sorted(segs):
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out: list[tuple[float, float]] = []
    for a, b in merged:
        pieces = [(a, b)]
        for c0, c1 in removed:
            c0, c1 = grid(c0), grid(c1)
            nxt = []
            for s, e in pieces:
                if c1 <= s or c0 >= e:
                    nxt.append((s, e))
                    continue
                if c0 > s:
                    nxt.append((s, c0))
                if c1 < e:
                    nxt.append((c1, e))
            pieces = nxt
        out += pieces
    out = [(a, b) for a, b in out if b - a >= 1 / fps - 1e-9]
    if not out:
        raise ClipperError("Everything in the clip was cut out; keep at least a moment of it.")
    return out
