"""Step 5: captions as an ASS subtitle file, 2-4 words per line, the spoken word in yellow with a pop."""

from __future__ import annotations

from pathlib import Path

OUT_W, OUT_H = 1080, 1920
MAX_WORDS = 4
MAX_CHARS = 16       # roughly what fits across the frame at FONT_SIZE in a heavy font
LINE_GAP = 0.6       # a pause this long always starts a new line
HOLD = 0.3           # how long a line stays up after its last word when nothing follows
FONT_SIZE = 76
MARGIN_V = 520       # bottom of the text sits ~520px above the bottom edge: lower third, clear of
MARGIN_LR = 110      # the TikTok / Reels caption block and the right-hand buttons

WHITE, YELLOW = "&HFFFFFF&", "&H00FFFF&"  # ASS colours are BGR
POP = r"\fscx118\fscy118\t(0,120,\fscx100\fscy100)"
_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
_SOFT_BREAK = (",", ";", ":", "—", "–")


def clip_words(words: list[dict], start: float, end: float) -> list[dict]:
    """Words whose middle falls inside the clip, with times relative to the clip start."""
    out = []
    for w in words:
        if start <= (w["s"] + w["e"]) / 2 < end:
            text = w["w"].strip()
            if text:
                out.append({"w": text, "s": max(0.0, w["s"] - start), "e": min(end - start, w["e"] - start)})
    return out


def _phrases(words: list[dict]) -> list[list[dict]]:
    """Split at sentence ends and pauses; a line never crosses either."""
    phrases, cur = [], []
    for w in words:
        if cur and w["s"] - cur[-1]["e"] > LINE_GAP:
            phrases.append(cur)
            cur = []
        cur.append(w)
        if w["w"].rstrip("\"'”’)»").endswith(_SENTENCE_END):
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)
    return phrases


def _partition(phrase: list[dict]) -> list[list[dict]]:
    """Cut a phrase into lines of up to MAX_WORDS words that fit the frame width: as few lines as
    possible, preferring breaks after commas and avoiding one-word lines."""
    n = len(phrase)
    best = [0.0] + [float("inf")] * n
    prev = [0] * (n + 1)
    for j in range(1, n + 1):
        for size in range(1, min(MAX_WORDS, j) + 1):
            i = j - size
            line = phrase[i:j]
            chars = sum(len(w["w"]) for w in line) + size - 1
            cost = 1.0 + 0.1 * abs(size - 3)
            if size == 1 and n > 1:
                cost += 1.5
            if size > 1 and chars > MAX_CHARS:
                break  # would wrap onto a second row; only a single long word may do that
            if j < n and not line[-1]["w"].endswith(_SOFT_BREAK):
                cost += 0.4
            if best[i] + cost < best[j]:
                best[j], prev[j] = best[i] + cost, i
    lines, j = [], n
    while j > 0:
        lines.append(phrase[prev[j]:j])
        j = prev[j]
    return lines[::-1]


def group_lines(words: list[dict]) -> list[list[dict]]:
    return [line for phrase in _phrases(words) for line in _partition(phrase)]


def _display(text: str) -> str:
    text = text.rstrip(",.;:") or text  # commas and full stops look like noise on screen
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")")


def _ts(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(words: list[dict], duration: float, font: str) -> str:
    lines = group_lines(words)
    events = []
    for li, line in enumerate(lines):
        line_end = line[-1]["e"] + HOLD
        if li + 1 < len(lines):
            nxt = lines[li + 1][0]["s"]
            line_end = nxt if nxt - line[-1]["e"] < LINE_GAP else min(line_end, nxt)
        line_end = min(line_end, duration)
        for wi, w in enumerate(line):
            start = w["s"]
            end = line[wi + 1]["s"] if wi + 1 < len(line) else line_end
            if round(end * 100) <= round(start * 100):
                continue
            parts = []
            for x in line:
                if x is w:
                    parts.append(f"{{\\c{YELLOW}{POP}}}{_display(x['w'])}{{\\c{WHITE}\\fscx100\\fscy100}}")
                else:
                    parts.append(_display(x["w"]))
            events.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{' '.join(parts)}")

    header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, \
Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,\
2,{MARGIN_LR},{MARGIN_LR},{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header + "\n".join(events) + "\n"


def make_captions(transcript: dict, clip: dict, clip_dir: Path, font: str) -> Path:
    """Write captions.ass for the clip (cheap, so it's rebuilt every run; unchanged files are left alone)."""
    words = clip_words(transcript["words"], clip["start"], clip["end"])
    ass = build_ass(words, clip["end"] - clip["start"], font)
    path = clip_dir / "captions.ass"
    if not path.exists() or path.read_text(encoding="utf-8") != ass:
        clip_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(ass, encoding="utf-8")
    return path
