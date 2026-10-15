"""Step 4: decide where the 9:16 window sits in every frame of a clip.

Per clip: find camera cuts, detect faces 5 times a second, follow the main person in each shot
with a calm virtual camera, and fall back to "full frame on a blurred background" for shots
without a clear face.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .util import ClipperError, cached, log, read_frames

REFRAME_VERSION = 1
ANALYSIS_WIDTH = 640
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

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def crop_width(width: int, height: int) -> int:
    """Width of a full-height 9:16 window, or the full width if the video is already that narrow."""
    cw = int(round(height * 9 / 16 / 2)) * 2
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
    boxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # sample -> box

    @property
    def last(self) -> int:
        return max(self.boxes)


def build_tracks(samples: list[list[tuple[float, float, float, float]]]) -> list[Track]:
    """Link face detections across samples: a face continues a track if its centre is within about
    one face-width of where that track was last seen."""
    tracks: list[Track] = []
    for k, boxes in enumerate(samples):
        active = [t for t in tracks if k - t.last <= TRACK_MAX_GAP]
        pairs = []
        for ti, t in enumerate(active):
            tx, ty, tw, _ = t.boxes[t.last]
            for bi, (bx, by, bw, _) in enumerate(boxes):
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


def main_track(tracks: list[Track], n_samples: int) -> Track | None:
    """The face that is biggest and on screen most often, if it's on screen often enough."""
    best, best_score = None, 0.0
    for t in tracks:
        presence = len(t.boxes) / n_samples
        if presence < MIN_PRESENCE:
            continue
        score = presence * float(np.median([b[2] for b in t.boxes.values()]))
        if score > best_score:
            best, best_score = t, score
    return best


# ---------------------------------------------------------------------------------------------
# Cuts and camera

def detect_cuts(changes: list[float], fps: float) -> list[int]:
    """Frame indexes where a new shot starts. changes[i] is the difference between frames i-1 and i."""
    n = len(changes)
    min_len = max(1, round(fps * MIN_SHOT_SECONDS))
    cuts, last = [], 0
    for i in range(1, n):
        c = changes[i]
        if c < CUT_MIN_CHANGE or i - last < min_len:
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

def analyze(video: Path, meta: dict, start: float, end: float) -> dict:
    import cv2

    W, H = meta["width"], meta["height"]
    fps = float(Fraction(meta["fps"]))
    cw = crop_width(W, H)
    aw = min(ANALYSIS_WIDTH, W) // 2 * 2
    ah = int(round(H * aw / W / 2)) * 2
    scale = W / aw
    step = max(1, round(fps / SAMPLE_FPS))
    detector = FaceDetector(aw, ah) if cw < W else None

    changes, sample_frames, samples, prev = [], [], [], None
    n = 0
    for n, frame in enumerate(read_frames(video, start, end - start, aw, ah, scale=True), 1):
        small = cv2.cvtColor(cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        small = small.astype(np.int16)
        changes.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
        if detector and (n - 1) % step == 0:
            sample_frames.append(n - 1)
            samples.append(detector.detect(frame))

    if n == 0:
        raise ClipperError(f"No video frames between {start:.1f}s and {end:.1f}s")
    cuts = detect_cuts(changes, fps)
    bounds = [0, *cuts, n]
    shots, xs = [], [None] * n
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        idx = [k for k, f in enumerate(sample_frames) if a <= f < b]
        track = main_track(build_tracks([samples[k] for k in idx]), len(idx)) if idx else None
        if track is None:
            shots.append({"start": a, "end": b, "mode": "fit"})
            continue
        frames = [sample_frames[k] for k in idx]
        centers = [track.boxes[j][0] * scale if j in track.boxes else None for j in range(len(idx))]
        xs[a:b] = camera_path(frames, centers, a, b - 1, W, cw, fps)
        shots.append({"start": a, "end": b, "mode": "track",
                      "presence": round(len(track.boxes) / len(idx), 2)})

    tracked = sum(s["end"] - s["start"] for s in shots if s["mode"] == "track")
    log(f"reframe: {len(shots)} shot(s), following a face in {tracked / max(n, 1):.0%} of frames"
        + (f" ({detector.kind} detector)" if detector else ""))
    return {"width": W, "height": H, "crop_w": cw, "frames": n, "shots": shots, "x": xs}


def reframe(work: Path, meta: dict, clip: dict, clip_dir: Path, force: bool = False) -> dict:
    params = {"version": REFRAME_VERSION, "start": clip["start"], "end": clip["end"],
              "video": [meta["width"], meta["height"], meta["fps"], meta["duration"]]}
    return cached(clip_dir / "reframe.json", params,
                  lambda: analyze(work / "video.mp4", meta, clip["start"], clip["end"]),
                  force=force, label=f"reframe #{clip['rank']}")
