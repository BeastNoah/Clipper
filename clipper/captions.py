"""Step 5: captions as an ASS subtitle file.

Lines of 2-4 words; the word being spoken lights up with a small pop, Claude's keywords stay in a
second colour, and the clip's title is shown at the top for the first seconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from .edit import EditMap
from .formats import VERTICAL, Format

MAX_WORDS = 4
LINE_GAP = 0.6       # a pause this long always starts a new line
HOLD = 0.3           # how long a line stays up after its last word when nothing follows
MARGIN_LOWER = 520   # lower third: bottom of the text ~520px above the bottom edge, clear of the
MARGIN_LR = 110      # TikTok / Reels caption block and the right-hand buttons
TITLE_SECONDS = 2.6

_SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
_SOFT_BREAK = (",", ";", ":", "—", "–")


@dataclass(frozen=True)
class CaptionStyle:
    name: str = "bold"
    font: str = "Arial Black"
    size: int = 76
    color: str = "#FFFFFF"
    highlight: str = "#FFE600"   # the word being spoken
    emphasis: str = "#3CFF6E"    # Claude's keywords
    outline: int = 6
    shadow: int = 2
    box: bool = False
    pop: bool = True
    uppercase: bool = False
    position: str = "lower"      # lower | middle
    chars_at_76: int = 16        # how many characters of this font fit across the frame at size 76

    @property
    def max_chars(self) -> int:
        return max(8, int(self.chars_at_76 * 76 / self.size * (0.87 if self.uppercase else 1.0)))


PRESETS = {
    "bold": CaptionStyle(),
    "clean": CaptionStyle(name="clean", font="Arial", size=68, outline=3, shadow=1, pop=False, chars_at_76=20),
    "boxed": CaptionStyle(name="boxed", font="Arial", size=66, outline=14, shadow=0, box=True, chars_at_76=19),
}


def make_style(preset: str = "bold", *, font: str | None = None, size: int | None = None,
               highlight: str | None = None, uppercase: bool = False, position: str = "lower") -> CaptionStyle:
    if preset not in PRESETS:
        raise ValueError(f"unknown caption style '{preset}' (choose from {', '.join(PRESETS)})")
    style = PRESETS[preset]
    if size is not None and not 20 <= size <= 200:
        raise ValueError("caption size must be between 20 and 200 pixels")
    if highlight:
        _bgr(highlight)  # raises for anything that isn't #RRGGBB
    changes = {"uppercase": uppercase, "position": position}
    if font:
        changes["font"] = font
    if size:
        changes["size"] = size
    if highlight:
        changes["highlight"] = highlight
    return replace(style, **changes)


def _bgr(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"colours look like #FFE600, not {hex_rgb!r}")
    return f"{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR for style lines."""
    return f"&H{alpha:02X}{_bgr(hex_rgb)}"


def tag_color(hex_rgb: str) -> str:
    """#RRGGBB -> &HBBGGRR& for inline colour tags."""
    return f"&H{_bgr(hex_rgb)}&"


# ---------------------------------------------------------------------------------------------
# Words and lines

def map_words(words: list[dict], start: float, edit: EditMap) -> list[dict]:
    """Clip words (source times) -> times in the finished clip, dropping words that were cut out."""
    out = []
    for w in words:
        text = w["w"].strip()
        if not text or edit.is_cut((w["s"] + w["e"]) / 2 - start):
            continue
        s, e = edit.to_output(w["s"] - start), edit.to_output(w["e"] - start)
        out.append({"w": text, "s": s, "e": max(e, s + 0.05)})  # very short words still get a moment
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


def _partition(phrase: list[dict], max_chars: int) -> list[list[dict]]:
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
            if size > 1 and chars > max_chars:
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


def group_lines(words: list[dict], max_chars: int = PRESETS["bold"].max_chars) -> list[list[dict]]:
    return [line for phrase in _phrases(words) for line in _partition(phrase, max_chars)]


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def keyword_positions(words: list[dict], keywords: list[str]) -> set[int]:
    """Indexes of caption words that belong to one of the keyword phrases."""
    toks = [_norm(w["w"]) for w in words]
    hits: set[int] = set()
    for kw in keywords:
        parts = [p for p in (_norm(x) for x in kw.split()) if p]
        if not parts:
            continue
        for i in range(len(toks) - len(parts) + 1):
            if toks[i:i + len(parts)] == parts:
                hits.update(range(i, i + len(parts)))
    return hits


def _display(text: str, style: CaptionStyle) -> str:
    text = text.rstrip(",.;:") or text  # commas and full stops look like noise on screen
    if style.uppercase:
        text = text.upper()
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")")


def _ts(t: float) -> str:
    cs = int(round(max(t, 0.0) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------------------------
# ASS output

def _header(styles: list[str], fmt: Format = VERTICAL) -> str:
    return f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {fmt.width}
PlayResY: {fmt.height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, \
Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _caption_style_line(style: CaptionStyle, fmt: Format = VERTICAL) -> str:
    align, margin_v = (5, 0) if style.position == "middle" else (2, fmt.v(MARGIN_LOWER))
    outline_col = ass_color("#000000", 0x50 if style.box else 0)
    return (f"Style: Default,{style.font},{style.size},{ass_color(style.color)},{ass_color(style.color)},"
            f"{outline_col},{ass_color('#000000', 0x80)},-1,0,0,0,100,100,0,0,{3 if style.box else 1},"
            f"{style.outline},{style.shadow},{align},{MARGIN_LR},{MARGIN_LR},{margin_v},1")


def _title_style_line(style: CaptionStyle, name: str, size: int, align: int, margin_v: int) -> str:
    # BorderStyle 3 draws a box in the outline colour behind the text: black text on a white card.
    return (f"Style: {name},{style.font},{size},{ass_color('#111111')},{ass_color('#111111')},"
            f"{ass_color('#FFFFFF')},{ass_color('#000000', 0x90)},-1,0,0,0,100,100,0,0,3,18,0,"
            f"{align},90,90,{margin_v},1")


def _escape(text: str) -> str:
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(words: list[dict], duration: float, style: CaptionStyle = PRESETS["bold"],
              keywords: list[str] | None = None, title: str | None = None, fmt: Format = VERTICAL) -> str:
    """words have times in the finished clip."""
    emphasized = keyword_positions(words, keywords or [])
    index = {id(w): i for i, w in enumerate(words)}
    hi, em, base = (f"\\c{tag_color(c)}" for c in (style.highlight, style.emphasis, style.color))
    lines = group_lines(words, int(style.max_chars * fmt.width / 1080))
    events = []
    if title:
        fade_out = min(TITLE_SECONDS, duration)
        events.append(f"Dialogue: 1,{_ts(0)},{_ts(fade_out)},Title,,0,0,0,,{{\\fad(150,300)}}{_escape(title)}")
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
                text = _display(x["w"], style)
                is_key = index[id(x)] in emphasized
                if x is w:
                    pop = ""
                    if style.pop:
                        big = 126 if is_key else 118
                        pop = f"\\fscx{big}\\fscy{big}\\t(0,120,\\fscx100\\fscy100)"
                    parts.append(f"{{{em if is_key else hi}{pop}}}{text}{{{base}\\fscx100\\fscy100}}")
                elif is_key:
                    parts.append(f"{{{em}}}{text}{{{base}}}")
                else:
                    parts.append(text)
            events.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{' '.join(parts)}")
    styles = [_caption_style_line(style, fmt), _title_style_line(style, "Title", 62, 8, fmt.v(250))]
    return _header(styles, fmt) + "\n".join(events) + "\n"


def build_cover_ass(title: str, style: CaptionStyle = PRESETS["bold"], fmt: Format = VERTICAL) -> str:
    """A big title card for the clip's cover image."""
    event = f"Dialogue: 0,{_ts(0)},{_ts(10)},Cover,,0,0,0,,{_escape(title)}"
    return _header([_title_style_line(style, "Cover", 96, 5, 0)], fmt) + event + "\n"


def _write_if_changed(path: Path, text: str) -> Path:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path


def make_captions(words: list[dict], duration: float, out_dir: Path, style: CaptionStyle, fmt: Format = VERTICAL,
                  title: str | None = None, cover_title: str = "", keywords: list[str] | None = None) -> Path:
    """Write captions.ass (and cover.ass) for one clip in one format. `words` have times in the finished
    clip. Cheap, so it's rebuilt every run; unchanged files are left alone so renders stay cached."""
    _write_if_changed(out_dir / "cover.ass", build_cover_ass(cover_title or title or "", style, fmt))
    return _write_if_changed(out_dir / "captions.ass", build_ass(words, duration, style, keywords, title, fmt))


def srt_lines(words: list[dict], max_chars: int = 42) -> list[tuple[float, float, str]]:
    """Subtitle lines for a .srt file: sentences, split to at most max_chars characters."""
    out = []
    for phrase in _phrases(words):
        line: list[dict] = []
        for w in phrase:
            if line and len(" ".join(x["w"] for x in line + [w])) > max_chars:
                out.append((line[0]["s"], line[-1]["e"], " ".join(x["w"] for x in line)))
                line = []
            line.append(w)
        if line:
            out.append((line[0]["s"], line[-1]["e"], " ".join(x["w"] for x in line)))
    return out


def build_srt(words: list[dict]) -> str:
    def ts(t: float) -> str:
        ms = int(round(max(t, 0.0) * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    return "".join(f"{n}\n{ts(a)} --> {ts(b)}\n{text}\n\n" for n, (a, b, text) in enumerate(srt_lines(words), 1))
