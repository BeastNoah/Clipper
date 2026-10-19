"""Jump cuts: shorten long pauses inside a clip and (optionally) drop filler words like "um" or "äh".

The result is a list of segments of the clip to keep, aligned to frame boundaries so that picture
and sound are cut at exactly the same moments.
"""

from __future__ import annotations

import re
from typing import Callable

FILLERS = {
    "um", "umm", "uh", "uhh", "uhm", "erm", "er", "hmm", "mm", "mhm", "ah",
    "äh", "ähm", "ähh", "öh", "öhm", "hm", "eh", "ehm",
}
MAX_PAUSE = 0.45               # silences longer than this between words get shortened
KEEP_AFTER, KEEP_BEFORE = 0.18, 0.08  # silence left after a word and before the next one
MIN_CUT = 0.2                  # don't bother with cuts shorter than this

MODES = ("off", "pauses", "on")  # "on" = pauses and filler words


def is_filler(word: str) -> bool:
    return re.sub(r"[^\w']", "", word.lower()) in FILLERS


def plan_cuts(words: list[dict], start: float, end: float, fps: float, mode: str,
              quiet: Callable[[float, float], bool] | None = None) -> list[tuple[float, float]]:
    """Segments to keep, in seconds from the clip start.

    `words` have source times. A pause is only cut if `quiet(a, b)` (source times) confirms
    nothing audible happens there, so laughter or words Whisper missed are never cut.
    """
    duration = end - start
    grid = lambda t: round(t * fps) / fps  # noqa: E731
    if mode == "off" or not words:
        return [(0.0, grid(duration))]

    drop_fillers = mode == "on"
    rel = [(w["s"] - start, w["e"] - start, drop_fillers and is_filler(w["w"])) for w in words]
    spoken = [(s, e) for s, e, filler in rel if not filler]
    fillers = [(s, e) for s, e, filler in rel if filler]

    def silent_outside_fillers(a: float, b: float) -> bool:
        if quiet is None:
            return True
        edges = [a]
        for fs, fe in fillers:
            if a < fe and fs < b:
                edges += [max(a, fs - 0.05), min(b, fe + 0.05)]
        edges.append(b)
        return all(quiet(start + x, start + y) for x, y in zip(edges[::2], edges[1::2]) if y - x > 0.05)

    removed = []
    for (_, a_end), (b_start, _) in zip(spoken, spoken[1:]):
        if b_start - a_end <= MAX_PAUSE:
            continue
        c0, c1 = grid(a_end + KEEP_AFTER), grid(b_start - KEEP_BEFORE)
        if c1 - c0 >= MIN_CUT and silent_outside_fillers(c0, c1):
            removed.append((c0, c1))
    # Each filler word goes too, however short the pauses around it (never into a spoken word).
    for fs, fe in fillers:
        # The nearest spoken words on each side, even when Whisper's times overlap the filler's.
        before = max((e for s, e in spoken if s < fs), default=0.0)
        after = min((s for s, e in spoken if e > fe), default=duration)
        c0, c1 = grid(max(fs - 0.04, before + 0.05)), grid(min(fe + 0.06, after - 0.03))
        if c1 > c0:
            removed.append((c0, c1))
    merged: list[tuple[float, float]] = []
    for c0, c1 in sorted(removed):
        if merged and c0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], c1))
        else:
            merged.append((c0, c1))
    removed = merged

    keep, t = [], 0.0
    for c0, c1 in removed:
        if c0 > t:
            keep.append((t, c0))
        t = c1
    if grid(duration) > t:
        keep.append((t, grid(duration)))
    return [(a, b) for a, b in keep if b - a >= 1 / fps - 1e-9]


class EditMap:
    """Maps times in the clip (seconds from its start) to times in the jump-cut output."""

    def __init__(self, segments: list[tuple[float, float]]):
        self.segments = segments
        self.offsets, total = [], 0.0
        for a, b in segments:
            self.offsets.append(total)
            total += b - a
        self.duration = total

    def is_cut(self, t: float) -> bool:
        """Was the moment t (seconds from the clip start) removed?"""
        return not any(a <= t < b for a, b in self.segments)

    def to_output(self, t: float) -> float:
        for (a, b), off in zip(self.segments, self.offsets):
            if t < a:
                return off           # inside a cut: jump to where the next kept part starts
            if t < b:
                return off + t - a
        return self.duration

    def frame_ranges(self, fps: float) -> list[tuple[int, int]]:
        return [(round(a * fps), round(b * fps)) for a, b in self.segments]

    @property
    def is_identity(self) -> bool:
        return len(self.segments) == 1 and self.segments[0][0] == 0.0
