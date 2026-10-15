"""Step 4: decide where the 9:16 window sits in every frame of a clip.

Per clip: find camera cuts and detect faces 5 times a second. In each shot, follow the main person
with a calm virtual camera. When two people share a shot, watch whose mouth moves while someone is
speaking and cut to them, or stack both on a split screen when they talk back and forth. Shots
without a clear face show the full frame on a blurred background.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .formats import VERTICAL, Format
from .util import ClipperError, Progress, cached, fingerprint, log, read_frames

if TYPE_CHECKING:
    from .ingest import ClipVideo

REFRAME_VERSION = 2
ANALYSIS_WIDTH = 960     # faces in a wide two-shot are only ~40px at 640
SAMPLE_FPS = 5

# Cut detection (same idea as PySceneDetect's adaptive detector): a frame-to-frame change that is
# both large and much larger than the changes around it.
CUT_MIN_CHANGE = 15.0
CUT_RATIO = 3.0
CUT_WINDOW = 2
MIN_SHOT_SECONDS = 0.5

MIN_FACE_WIDTH = 0.03   # of frame width; smaller faces are background people
MIN_PRESENCE = 0.35     # the main face must be in at least this share of a shot's samples
TRACK_MAX_GAP = 5       # samples a face may vanish for and still be the same track
DEADZONE = 0.12         # of crop width; the camera ignores head movements smaller than this
MOVE_MIN_S, MOVE_MAX_S = 0.6, 1.6

SECOND_PERSON = 0.6     # a second face at least this big (relative to the main one) is a co-speaker
MIN_TURN_SAMPLES = 5    # a speaker must talk ~1s before the camera cuts to them
BANTER_WINDOW_S = 6.0   # two or more changes of speaker within this window counts as back-and-forth
MIN_LAYOUT_S = 4.0      # don't switch between split screen and single framing more often than this
LAYOUTS = ("auto", "follow", "split", "main")

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def crop_width(width: int, height: int, ratio: float = 9 / 16) -> int:
    """Width of a full-height window of the output's shape (9:16 by default), or the full width if
    the video is already that narrow."""
    cw = int(round(height * ratio / 2)) * 2
    return min(cw, width)


# ---------------------------------------------------------------------------------------------
# Faces

def _model_cache_dir() -> Path:
    return Path(os.environ.get("CLIPPER_CACHE", Path.home() / ".cache" / "clipper"))


class FaceDetector:
    """OpenCV's YuNet face detector (downloaded once, ~230 KB), or the bundled Haar cascade if
    the download isn't possible."""

    _warned = False

    def __init__(self, width: int, height: int):
        import cv2

        self.cv2 = cv2
        self.min_px = max(8, int(width * MIN_FACE_WIDTH))
        model = self._yunet_model()
        if model is not None:
            self.kind = "yunet"
            self.net = cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.7, 0.3, 50)
        elif hasattr(cv2, "CascadeClassifier"):
            self.kind = "haar"
            self.net = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            self.kind, self.net = "none", None
            log("reframe: no face detector available; every shot will use the full frame on a blur")

    @classmethod
    def _yunet_model(cls) -> Path | None:
        path = _model_cache_dir() / "face_detection_yunet_2023mar.onnx"
        if path.exists():
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(YUNET_URL, timeout=30) as r:
                tmp.write_bytes(r.read())
            os.replace(tmp, path)
            return path
        except Exception as e:  # noqa: BLE001 - any failure just means the fallback detector
            if not cls._warned:
                log(f"reframe: couldn't download the YuNet face model ({e}); using OpenCV's Haar detector")
                cls._warned = True
            return None

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        """Faces as (center x, center y, width, height) in frame pixels."""
        if self.kind == "yunet":
            _, faces = self.net.detect(frame)
            boxes = [] if faces is None else [tuple(map(float, f[:4])) for f in faces]
        elif self.kind == "none":
            boxes = []
        else:
            gray = self.cv2.equalizeHist(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY))
            rects = self.net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                              minSize=(self.min_px, self.min_px))
            boxes = [tuple(map(float, r)) for r in rects]
        return [(x + w / 2, y + h / 2, w, h) for x, y, w, h in boxes if w >= self.min_px]


@dataclass
class Track:
    boxes: dict[int, tuple] = field(default_factory=dict)  # sample -> (cx, cy, w, h, mouth activity)

    @property
    def last(self) -> int:
        return max(self.boxes)


def build_tracks(samples: list[list[tuple]]) -> list[Track]:
    """Link face detections across samples: a face continues a track if its centre is within about
    one face-width of where that track was last seen."""
    tracks: list[Track] = []
    for k, boxes in enumerate(samples):
        active = [t for t in tracks if k - t.last <= TRACK_MAX_GAP]
        pairs = []
        for ti, t in enumerate(active):
            tx, ty, tw = t.boxes[t.last][:3]
            for bi, (bx, by, bw, *_) in enumerate(boxes):
                d = np.hypot(bx - tx, by - ty) / max(tw, bw)
                if d < 1.0:
                    pairs.append((d, ti, bi))
        used_t, used_b = set(), set()
        for _, ti, bi in sorted(pairs):
            if ti not in used_t and bi not in used_b:
                active[ti].boxes[k] = boxes[bi]
                used_t.add(ti)
                used_b.add(bi)
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                tracks.append(Track({k: b}))
    return tracks


def _median(t: Track, k: int) -> float:
    return float(np.median([b[k] for b in t.boxes.values()]))


def people(tracks: list[Track], n_samples: int) -> list[Track]:
    """Faces on screen often enough to matter, most important first (biggest x most often)."""
    keep = [t for t in tracks if len(t.boxes) / n_samples >= MIN_PRESENCE]
    return sorted(keep, key=lambda t: -(len(t.boxes) / n_samples) * _median(t, 2))


def main_track(tracks: list[Track], n_samples: int) -> Track | None:
    """The face that is biggest and on screen most often, if it's on screen often enough."""
    found = people(tracks, n_samples)
    return found[0] if found else None


def mouth_activity(gray: np.ndarray, earlier: np.ndarray, box: tuple) -> float:
    """How much the mouth area changed between two nearby frames, minus how much the eyes did
    (so nodding or moving the whole head doesn't count as talking)."""
    cx, cy, w, h = box[:4]
    H, W = gray.shape

    def change(y0: float, y1: float, half_w: float) -> float:
        xa, xb = max(0, int(cx - half_w * w)), min(W, int(cx + half_w * w))
        ya, yb = max(0, int(cy + y0 * h)), min(H, int(cy + y1 * h))
        if xb - xa < 2 or yb - ya < 2:
            return 0.0
        return float(np.abs(gray[ya:yb, xa:xb].astype(np.int16) - earlier[ya:yb, xa:xb]).mean())

    return max(0.0, change(0.15, 0.5, 0.3) - change(-0.3, -0.05, 0.35))


def _merge_short_runs(values: np.ndarray, min_run: int) -> np.ndarray:
    """Absorb runs shorter than min_run into the run before them (or after, at the start)."""
    out = values.copy()
    for _ in range(2):
        bounds = [0, *(np.flatnonzero(np.diff(out)) + 1), len(out)]
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < min_run and len(bounds) > 2:
                out[lo:hi] = out[lo - 1] if lo > 0 else out[hi]
    return out


def banter(turns: np.ndarray, sample_fps: float = SAMPLE_FPS) -> np.ndarray:
    """Per sample: are the two people going back and forth right now?"""
    n = len(turns)
    switched = np.zeros(n)
    switched[1:] = np.diff(turns) != 0
    total = np.concatenate([[0], np.cumsum(switched)])
    half = int(BANTER_WINDOW_S * sample_fps / 2)
    busy = np.array([total[min(n, j + half + 1)] - total[max(0, j - half)] >= 2 for j in range(n)])
    return _merge_short_runs(busy.astype(int), int(MIN_LAYOUT_S * sample_fps)).astype(bool)


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) < 2:
        return a.astype(float)
    k = min(n, len(a))
    return np.convolve(np.pad(a.astype(float), (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")


def speaker_turns(act_a: np.ndarray, act_b: np.ndarray, speech: np.ndarray,
                  min_run: int = MIN_TURN_SAMPLES) -> np.ndarray:
    """0 or 1 per sample: which of two people is talking. Decided only while someone speaks, with
    short turns merged away so the camera doesn't flicker between them."""
    sa, sb = _smooth(act_a, 7), _smooth(act_b, 7)
    decided = [0 if x > 1.25 * y + 0.05 else 1 if y > 1.25 * x + 0.05 else -1 for x, y in zip(sa, sb)]
    first = next((d for d, sp in zip(decided, speech) if sp and d >= 0), 0)
    cur, out = first, []
    for d, sp in zip(decided, speech):
        if sp and d >= 0:
            cur = d
        out.append(cur)
    return _merge_short_runs(np.array(out, int), min_run)


# ---------------------------------------------------------------------------------------------
# Cuts and camera

def detect_cuts(changes: list[float], fps: float) -> list[int]:
    """Frame indexes where a new shot starts. changes[i] is the difference between frames i-1 and i."""
    n = len(changes)
    min_len = max(1, round(fps * MIN_SHOT_SECONDS))
    cuts, last = [], 0
    for i in range(1, n):
        c = changes[i]
        if c < CUT_MIN_CHANGE or i - last < min_len or n - i < min_len:
            continue
        around = [changes[j] for j in range(max(1, i - CUT_WINDOW), min(n, i + CUT_WINDOW + 1)) if j != i]
        avg = sum(around) / len(around) if around else 0.0
        if avg <= 1e-6 or c / avg >= CUT_RATIO:
            cuts.append(i)
            last = i
    return cuts


def _ease(t: float) -> float:
    return (1 - np.cos(np.pi * min(max(t, 0.0), 1.0))) / 2


def camera_path(sample_frames: list[int], centers: list[float | None], first: int, last: int,
                width: int, cw: int, fps: float) -> list[int]:
    """Left edge of the crop window for frames first..last (inclusive) of one shot.

    Behaves like a calm camera operator: holds still through small movements, and only when the
    subject has clearly moved does it ease over to them.
    """
    known = [(f, c) for f, c in zip(sample_frames, centers) if c is not None]
    lo, hi = cw / 2, width - cw / 2
    xs = np.interp(sample_frames, [f for f, _ in known], [c for _, c in known])
    if len(xs) >= 3:  # 1-second median removes detector jitter and one-off misdetections
        padded = np.pad(xs, 2, mode="edge")
        xs = np.array([np.median(padded[i:i + 5]) for i in range(len(xs))])
    xs = np.clip(xs, lo, hi)

    deadzone = DEADZONE * cw
    cam = float(np.median(xs[:3]))
    moves = []  # (start frame, end frame, from, to)
    busy_until = first
    for k in range(1, len(xs)):
        if sample_frames[k] < busy_until:
            continue
        sustained = k + 1 >= len(xs) or abs(xs[k + 1] - cam) > deadzone
        if abs(xs[k] - cam) > deadzone and sustained:
            # We know the future: aim for where the subject comes to rest, start moving as they
            # start, and take at least as long as they take to get there (one smooth move).
            j, direction = k, np.sign(xs[k] - cam)
            while j + 1 < len(xs) and (xs[j + 1] - xs[j]) * direction > deadzone / 4:
                j += 1
            dest = float(np.median(xs[j:j + 3]))
            start = max(sample_frames[k - 1], busy_until)
            travel = (sample_frames[j] - start) / fps
            seconds = max(min(MOVE_MAX_S, MOVE_MIN_S + abs(dest - cam) / cw), travel)
            end = start + max(1, round(seconds * fps))
            moves.append((start, end, cam, dest))
            cam, busy_until = dest, end

    out, center, m = [], float(np.median(xs[:3])), 0
    for f in range(first, last + 1):
        while m < len(moves) and f >= moves[m][1]:
            center = moves[m][3]
            m += 1
        if m < len(moves) and f >= moves[m][0]:
            s, e, a, b = moves[m]
            center = a + (b - a) * _ease((f - s) / (e - s))
        out.append(int(round(min(max(center - cw / 2, 0), width - cw))))
    return out


# ---------------------------------------------------------------------------------------------

def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def plan_shot(samples: list[list[tuple]], frames: list[int], a: int, b: int, W: int, H: int, cw: int,
              fps: float, layout: str, speaking: list[bool], fmt: Format = VERTICAL) -> tuple[str, list | None, dict]:
    """Framing for frames a..b-1 of one shot: ("fit", None), or a mode with one value per frame."""
    n = len(samples)
    found = people(build_tracks(samples), n) if n else []
    if not found:
        return "fit", None, {}

    def centers(t: Track) -> list[float | None]:
        return [t.boxes[j][0] if j in t.boxes else None for j in range(n)]

    main = found[0]
    other = next((t for t in found[1:] if _median(t, 2) >= SECOND_PERSON * _median(main, 2)), None)
    if other is None or layout == "main":
        return "track", camera_path(frames, centers(main), a, b - 1, W, cw, fps), \
            {"presence": round(len(main.boxes) / n, 2)}

    left, right = sorted([main, other], key=lambda t: _median(t, 0))
    span = _median(right, 0) - _median(left, 0) + max(_median(left, 2), _median(right, 2))
    if span <= 0.8 * cw and layout != "split":  # both fit in one window: frame them together
        mid = [(lc + rc) / 2 if lc is not None and rc is not None else lc if lc is not None else rc
               for lc, rc in zip(centers(left), centers(right))]
        return "track", camera_path(frames, mid, a, b - 1, W, cw, fps), {"people": 2, "framing": "together"}

    act = [np.array([t.boxes[j][4] if j in t.boxes else 0.0 for j in range(n)]) for t in (left, right)]
    turns = speaker_turns(act[0], act[1], np.array(speaking))
    spoken = turns[np.array(speaking)] if any(speaking) else turns
    share = float(spoken.mean()) if len(spoken) else 0.0
    switches = int(np.count_nonzero(np.diff(turns)))
    info = {"people": 2, "right_speaks": round(share, 2), "switches": switches}

    def per_frame(per_sample: list) -> list:
        out, k = [], 0
        for f in range(a, b):
            while k + 1 < n and frames[k + 1] <= f:
                k += 1
            out.append(per_sample[k](f - a))
        return out

    paths = [camera_path(frames, centers(t), a, b - 1, W, cw, fps) for t in (left, right)]
    follow = per_frame([lambda i, sp=int(sp): paths[sp][i] for sp in turns])
    if layout == "follow" or fmt.ratio >= 1:  # stacking two people only works in tall formats
        return "follow", follow, info

    hc = min(H, max(_even(0.4 * H), _even(3.5 * max(_median(left, 3), _median(right, 3)))))
    wc = min(W, _even(hc * fmt.width / (fmt.height / 2)))
    xs = [camera_path(frames, centers(t), a, b - 1, W, wc, fps) for t in (left, right)]
    ys = [int(min(max(_median(t, 1) - 0.42 * hc, 0), H - hc)) for t in (left, right)]
    split = [[xl, ys[0], xr, ys[1], hc] for xl, xr in zip(*xs)]
    busy = np.ones(n, bool) if layout == "split" else banter(turns)
    if busy.all():
        return "split", split, info
    if not busy.any():
        return "follow", follow, info
    mixed = per_frame([lambda i, on=bool(on): split[i] if on else follow[i] for on in busy])
    return "follow+split", mixed, info


def analyze(video: Path, meta: dict, start: float, end: float, words: list[dict] | None = None,
            layout: str = "auto", fmt: Format = VERTICAL) -> dict:
    import cv2

    W, H = meta["width"], meta["height"]
    fps = float(Fraction(meta["fps"]))
    cw = crop_width(W, H, fmt.ratio)
    aw = min(ANALYSIS_WIDTH, W) // 2 * 2
    ah = int(round(H * aw / W / 2)) * 2
    scale = W / aw
    step = max(1, round(fps / SAMPLE_FPS))
    detector = FaceDetector(aw, ah) if cw < W else None
    talk = sorted((w["s"] - start - 0.15, w["e"] - start + 0.15) for w in words or [])

    changes, sample_frames, samples, prev, grays = [], [], [], None, []
    n = 0
    progress = Progress("reframe", (end - start) * fps)
    for n, frame in enumerate(read_frames(video, start, end - start, aw, ah, scale=True), 1):
        progress.update(n)
        small = cv2.cvtColor(cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        small = small.astype(np.int16)
        changes.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
        if detector is None:
            continue
        grays = (grays + [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)])[-3:]
        if (n - 1) % step == 0:
            sample_frames.append(n - 1)
            boxes = detector.detect(frame)
            samples.append([(*bx, mouth_activity(grays[-1], grays[0], bx)) for bx in boxes])
    Progress.clear_line()

    if n == 0:
        raise ClipperError(f"No video frames between {start:.1f}s and {end:.1f}s")
    cuts = detect_cuts(changes, fps)
    bounds = [0, *cuts, n]
    shots, xs = [], [None] * n
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        idx = [k for k, f in enumerate(sample_frames) if a <= f < b]
        frames = [sample_frames[k] for k in idx]
        shot_samples = [[(bx[0] * scale, bx[1] * scale, bx[2] * scale, bx[3] * scale, bx[4]) for bx in samples[k]]
                        for k in idx]
        speaking = [any(t0 <= f / fps <= t1 for t0, t1 in talk) for f in frames]
        if not idx and shots and a > 0:
            # too short to have a face sample: keep framing it like the shot before
            xs[a:b] = [xs[a - 1]] * (b - a)
            shots.append({"start": a, "end": b, "mode": shots[-1]["mode"], "inherited": True})
            continue
        mode, values, info = plan_shot(shot_samples, frames, a, b, W, H, cw, fps, layout, speaking, fmt)
        if values is not None:
            xs[a:b] = values
        shots.append({"start": a, "end": b, "mode": mode, **info})

    framed = sum(s["end"] - s["start"] for s in shots if s["mode"] != "fit")
    modes = sorted({s["mode"] for s in shots})
    log(f"reframe: {len(shots)} shot(s) ({', '.join(modes)}), a face is framed in {framed / max(n, 1):.0%} "
        f"of frames" + (f" ({detector.kind} detector)" if detector else ""))
    return {"width": W, "height": H, "crop_w": cw, "frames": n, "shots": shots, "x": xs}


def reframe(work: Path, meta: dict, clip: dict, out_dir: Path, words: list[dict] | None = None,
            layout: str = "auto", force: bool = False, src: "ClipVideo | None" = None,
            fmt: Format = VERTICAL) -> dict:
    from .ingest import ClipVideo

    src = src or ClipVideo(work / "video.mp4", 0.0, meta["width"], meta["height"], meta["fps"])
    speech = [[round(w["s"], 2), round(w["e"], 2)] for w in words or []]
    params = {"version": REFRAME_VERSION, "start": clip["start"], "end": clip["end"], "layout": layout,
              "speech": fingerprint(speech), "video": src.key(), "format": fmt.name}
    # The file may be a section of the video: work in its own time, which starts at src.offset.
    local = [{**w, "s": w["s"] - src.offset, "e": w["e"] - src.offset} for w in words or []]
    dims = {"width": src.width, "height": src.height, "fps": src.fps}
    return cached(out_dir / "reframe.json", params,
                  lambda: analyze(src.path, dims, clip["start"] - src.offset, clip["end"] - src.offset,
                                  local, layout, fmt),
                  force=force, label=f"reframe #{clip['rank']} ({fmt.name})")
