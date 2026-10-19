"""Step 3: Claude proposes and scores clips; code then fixes the timing.

Imported as `clipper.select`. Don't run files in this package directly as scripts: this module's
name would then shadow Python's built-in `select` module.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

from .audio import Loudness, sentence_cues
from .llm import Claude, InvalidReply, extract_json
from .util import Cancelled, ClipperError, fingerprint, fmt_time, log, read_json, thread_pool, write_json

SELECT_VERSION = 3
CHUNK_SECONDS = 15 * 60
MAX_SENTENCE_SECONDS = 15.0  # longer "sentences" (run-ons, missing punctuation) get split
SENTENCE_GAP = 2.0           # a pause this long ends a sentence even without punctuation
PAD_BEFORE, PAD_AFTER = 0.2, 0.4
EXTRA_ROUNDS = 2             # how many times to ask Claude for more when too few clips survive
CLAUDE_AT_ONCE = 4           # requests in flight at the same time for the parts of a long video
MAX_EARLIER_PARTS = 3        # earlier parts (a set-up, a callback, montage moments) played before a clip
MIN_MAIN_PART = 4.0          # seconds the clip's own moment keeps, however long its earlier parts

_END = (".", "?", "!", "…", "。", "？", "！")
_TRAILING = "\"'”’)»]"
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "e.g.", "i.e.", "jr.", "sr."}

SYSTEM_PROMPT = """\
You are a senior short-form video editor. You cut long videos (podcasts, interviews, talks, \
streams) into vertical clips for TikTok, Instagram Reels and YouTube Shorts. You know what makes \
people stop scrolling, watch to the end and share: a first line that demands attention, one complete \
moment that pays off on its last line, and no need for outside context. You judge strictly. You \
answer with JSON only."""

MOMENT_TYPES = ("joke", "story", "hot take", "insight", "confession", "roast", "relatable", "other")

RUBRIC = """\
A clip must pass all three checks; one that fails any of them is not a clip, whatever else it has \
going for it:
- It makes sense to someone who has not seen the rest of the video.
- It does not depend on anything only visible on screen.
- It has a real ending: it stops on a line that lands, not where the speaker trails off.

Score each clip on four counts, each from 0 to 10:
- hook: would a stranger stop scrolling in the first 3 seconds? 9-10: an open question, a bold claim, \
or the middle of something happening ("I got fired on my second day, and it was the best thing..."). \
4-5: a decent line that still needs set-up to matter. 0-2: "So, um, the other thing is..."
- payoff: does the last line land? 9-10: a punchline, a reveal, a twist, or a line people would \
quote. 4-5: the point gets made, then it just stops. 0-2: trails off, or cuts before the point.
- reaction: what does a viewer do right after? 9-10: laughs out loud, says "wait, what?", sends it \
to a friend, or has to argue in the comments. 4-5: nods. 0-2: nothing.
- pace: how dense it is. 9-10: every line earns its place. 4-5: some rambling or repetition. \
0-2: mostly filler, false starts and "basically".
A note like [+1.8s non-speech sound] after a line inside the clip is usually laughter or applause \
from the people who were there: treat it as strong evidence for payoff and reaction.
Be strict: most moments in any video are 4-6 on every count. The score multiplies hook and payoff, \
so a clip with a weak hook or no payoff is a weak clip however good the rest.
Also name the kind of moment: one of {types}.""".format(types=", ".join(MOMENT_TYPES))


# ---------------------------------------------------------------------------------------------
# Sentences and chunks

@dataclass
class Sentence:
    i: int
    start: float
    end: float
    w0: int  # first word index (inclusive)
    w1: int  # last word index (inclusive)
    text: str


def _ends_sentence(token: str) -> bool:
    t = token.strip().rstrip(_TRAILING)
    if not t.endswith(_END):
        return False
    low = t.lower()
    return low not in _ABBREVIATIONS and not re.fullmatch(r"(?:[a-z]\.){2,}", low)


def _split_long(idx: list[int], words: list[dict]) -> list[list[int]]:
    """Split an over-long run of words at the best pause or comma, recursively."""
    dur = lambda ix: words[ix[-1]]["e"] - words[ix[0]]["s"]  # noqa: E731
    total = dur(idx)
    if total <= MAX_SENTENCE_SECONDS or len(idx) < 2:
        return [idx]
    best_j, best = 0, -math.inf
    for j in range(len(idx) - 1):
        a, b = idx[:j + 1], idx[j + 1:]
        gap = words[b[0]]["s"] - words[a[-1]]["e"]
        punct = words[a[-1]]["w"].strip().endswith((",", ";", ":", "—", "–"))
        score = 2 * gap + (1.0 if punct else 0.0) - abs(dur(a) - dur(b)) / total
        if score > best:
            best_j, best = j, score
    return _split_long(idx[:best_j + 1], words) + _split_long(idx[best_j + 1:], words)


def split_sentences(words: list[dict]) -> list[Sentence]:
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, w in enumerate(words):
        if cur and w["s"] - words[cur[-1]]["e"] > SENTENCE_GAP:
            groups.append(cur)
            cur = []
        cur.append(i)
        if _ends_sentence(w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    sentences: list[Sentence] = []
    for g in groups:
        for part in _split_long(g, words):
            text = "".join(words[k]["w"] for k in part).strip()
            sentences.append(Sentence(len(sentences), words[part[0]]["s"], words[part[-1]]["e"],
                                      part[0], part[-1], text))
    return sentences


def make_chunks(sentences: list[Sentence], overlap: float,
                chunk_seconds: float = CHUNK_SECONDS) -> list[tuple[int, int]]:
    """Overlapping, evenly sized chunks as (first, last) sentence indexes.

    The overlap is at least the longest clip, so every possible clip fits whole inside some chunk.
    """
    t0, t1 = sentences[0].start, sentences[-1].end
    total = t1 - t0
    if total <= chunk_seconds * 1.1:
        return [(sentences[0].i, sentences[-1].i)]
    overlap = min(overlap, chunk_seconds / 2)
    n = math.ceil((total - overlap) / (chunk_seconds - overlap))
    length = (total + (n - 1) * overlap) / n
    chunks = []
    for k in range(n):
        lo = t0 + k * (length - overlap)
        hi = lo + length
        idx = [s.i for s in sentences if s.start < hi and s.end > lo]
        if idx:
            chunks.append((idx[0], idx[-1]))
    return chunks


# ---------------------------------------------------------------------------------------------
# Reply formats

class SubScores(BaseModel):
    hook: int = Field(ge=0, le=10)
    payoff: int = Field(ge=0, le=10)
    reaction: int = Field(ge=0, le=10)
    pace: int = Field(ge=0, le=10)


def combine(scores: dict) -> int:
    """One 0-100 score from the four counts. Hook and payoff multiply: nobody sees the rest of a clip
    they scrolled past, and a clip that doesn't land isn't shared, so neither can be made up for
    elsewhere. Reaction and pace share the last factor."""
    rest = 0.6 * scores["reaction"] + 0.4 * scores["pace"]
    return int(round(scores["hook"] * scores["payoff"] * rest / 10))


def moment_type(value: str) -> str:
    t = re.sub(r"[\s_-]+", " ", str(value).strip().lower())
    return t if t in MOMENT_TYPES else "other"


class Span(BaseModel):
    start: float
    end: float


class ProposedClip(BaseModel):
    start: float
    end: float
    before: list[Span] = Field(default_factory=list, max_length=MAX_EARLIER_PARTS)
    title: str = Field(min_length=1, max_length=120)
    hook: str = Field(min_length=1)
    type: str = "other"
    scores: SubScores
    keywords: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        return moment_type(v)


class ProposalReply(BaseModel):
    clips: list[ProposedClip]


class Rescore(SubScores):
    id: str
    duplicate_of: str | None = None


class RescoreReply(BaseModel):
    ranking: list[Rescore]  # best first


class Trim(BaseModel):
    id: str
    first: int
    last: int


class TrimReply(BaseModel):
    trims: list[Trim]


def _validate(model: type[BaseModel], reply: str) -> BaseModel:
    try:
        return model.model_validate(extract_json(reply))
    except ValidationError as e:
        raise InvalidReply(str(e)) from e


def parse_proposals(reply: str, lo: float, hi: float) -> list[dict]:
    parsed = _validate(ProposalReply, reply)
    problems = []
    for n, c in enumerate(parsed.clips, 1):
        if c.end - c.start < 1:
            problems.append(f"clips[{n - 1}]: end ({c.end}) must be after start ({c.start})")
        elif c.start < lo - 2 or c.end > hi + 2:
            problems.append(f"clips[{n - 1}]: {c.start}-{c.end} is outside this part of the transcript "
                            f"({lo:.1f}-{hi:.1f}); use times from the lines above")
        for b in c.before:
            if b.end <= b.start:
                problems.append(f"clips[{n - 1}].before: end ({b.end}) must be after start ({b.start})")
            elif b.end > c.start + 1:
                problems.append(f"clips[{n - 1}].before: {b.start}-{b.end} must end before the clip starts "
                                f"({c.start}); earlier parts come from earlier in the video")
            elif b.start < lo - 2:
                problems.append(f"clips[{n - 1}].before: {b.start}-{b.end} is outside this part of the "
                                f"transcript ({lo:.1f}-{hi:.1f}); use times from the lines above")
    if problems:
        raise InvalidReply("\n".join(problems))
    return [c.model_dump() for c in parsed.clips]


def parse_rescores(reply: str, ids: list[str]) -> dict[str, dict]:
    """{clip id: {sub-scores, duplicate_of, order}}; order is the clip's place in Claude's ranking, 0 first."""
    parsed = _validate(RescoreReply, reply)
    got = [s.id for s in parsed.ranking]
    problems = []
    missing = [i for i in ids if i not in got]
    if missing:
        problems.append("missing scores for: " + ", ".join(missing))
    unknown = sorted(set(got) - set(ids))
    if unknown:
        problems.append("unknown ids: " + ", ".join(unknown))
    dupes = sorted({i for i in got if got.count(i) > 1})
    if dupes:
        problems.append("scored more than once: " + ", ".join(dupes))
    for s in parsed.ranking:
        if s.duplicate_of is not None and (s.duplicate_of not in ids or s.duplicate_of == s.id):
            problems.append(f"{s.id}: duplicate_of must be the id of another clip in the list, or null")
    if problems:
        raise InvalidReply("; ".join(problems) + ". Give exactly one entry per clip id.")
    return {s.id: {**s.model_dump(exclude={"id"}), "order": n} for n, s in enumerate(parsed.ranking)}


def parse_trims(reply: str, jobs: dict[str, tuple[int, int, float]], tl: "Timeline", min_len: float,
                max_len: float) -> dict[str, tuple[int, int]]:
    """Validate Claude's choice of which lines to keep: {trim id: (first, last)} with 1-based lines.
    jobs: {trim id: (first sentence, last sentence, seconds of earlier parts played before them)}."""
    parsed = _validate(TrimReply, reply)
    got = {t.id: t for t in parsed.trims}
    problems = []
    missing = [i for i in jobs if i not in got]
    if missing:
        problems.append("missing: " + ", ".join(missing))
    for tid, t in got.items():
        if tid not in jobs:
            problems.append(f"unknown id {tid}")
            continue
        si, ei, extra = jobs[tid]
        n = ei - si + 1
        if not 1 <= t.first <= t.last <= n:
            problems.append(f"{tid}: first and last must be line numbers with 1 <= first <= last <= {n}")
            continue
        length = tl.sentences[si + t.last - 1].end - tl.sentences[si + t.first - 1].start
        lo, hi = main_limits(min_len, max_len, extra)
        if not lo <= length <= hi - 0.6:
            problems.append(f"{tid}: lines {t.first}-{t.last} run {length:.1f}s, which is outside "
                            f"{lo:.0f}-{hi - 1:.0f}s")
    if problems:
        raise InvalidReply("; ".join(problems))
    return {tid: (t.first, t.last) for tid, t in got.items()}


# ---------------------------------------------------------------------------------------------
# Timing fixes

class Timeline:
    def __init__(self, sentences: list[Sentence], words: list[dict], duration: float,
                 lo: int = 0, hi: int | None = None):
        """lo..hi: the sentences clips may use (a chosen time range); snapping and fitting stay inside."""
        self.sentences, self.words, self.duration = sentences, words, duration
        self.lo, self.hi = lo, len(sentences) - 1 if hi is None else hi
        self._starts = [s.start for s in sentences]

    def clip_start(self, si: int) -> float:
        s = self.sentences[si]
        prev_end = self.words[s.w0 - 1]["e"] if s.w0 > 0 else 0.0
        return max(0.0, s.start - min(PAD_BEFORE, max(0.0, (s.start - prev_end) / 2)))

    def clip_end(self, ei: int) -> float:
        s = self.sentences[ei]
        nxt = self.words[s.w1 + 1]["s"] if s.w1 + 1 < len(self.words) else self.duration
        return min(self.duration, s.end + min(PAD_AFTER, max(0.0, (nxt - s.end) / 2)))

    def dur(self, si: int, ei: int) -> float:
        return self.clip_end(ei) - self.clip_start(si)

    def nearest_start(self, t: float) -> int:
        k = bisect_left(self._starts, t)
        near = [i for i in (k - 1, k) if self.lo <= i <= self.hi] or [min(max(k, self.lo), self.hi)]
        return min(near, key=lambda i: abs(self.sentences[i].start - t))

    def nearest_end(self, t: float, at_least: int) -> int:
        return min(range(max(at_least, self.lo), self.hi + 1), key=lambda i: abs(self.sentences[i].end - t))

    def text(self, si: int, ei: int) -> str:
        a, b = self.sentences[si].w0, self.sentences[ei].w1
        return "".join(w["w"] for w in self.words[a:b + 1]).strip()


def snap(tl: Timeline, start: float, end: float) -> tuple[int, int]:
    """Move a proposal's start and end to the nearest sentence boundaries."""
    si = tl.nearest_start(start)
    return si, tl.nearest_end(end, si)


def fit(tl: Timeline, si: int, ei: int, min_len: float, max_len: float,
        floor: int | None = None) -> tuple[int, int] | None:
    """Trim or extend sentences si..ei to fit the length limits.

    Trimming happens at the end (the start holds the hook). Extending adds the next sentence,
    or the previous one when there is no room after (but never before `floor`). Returns None if
    the clip can't be made to fit.
    """
    floor = tl.lo if floor is None else max(floor, tl.lo)
    while ei > si and tl.dur(si, ei) > max_len:
        ei -= 1
    if tl.dur(si, ei) > max_len:
        return None
    while tl.dur(si, ei) < min_len:
        if ei + 1 <= tl.hi and tl.dur(si, ei + 1) <= max_len:
            ei += 1
        elif si > floor and tl.dur(si - 1, ei) <= max_len:
            si -= 1
        else:
            return None
    return si, ei


def snap_and_fit(tl: Timeline, start: float, end: float, min_len: float, max_len: float) -> tuple[int, int] | None:
    return fit(tl, *snap(tl, start, end), min_len, max_len)


def main_limits(min_len: float, max_len: float, extra: float) -> tuple[float, float]:
    """Length limits for a clip's own moment when `extra` seconds of earlier parts play before it."""
    return max(min(MIN_MAIN_PART, min_len), min_len - extra), max_len - extra


def part_record(tl: Timeline, si: int, ei: int) -> dict:
    return {"si": si, "ei": ei, "start": round(tl.clip_start(si), 3), "end": round(tl.clip_end(ei), 3)}


def clip_parts(clip: dict) -> list[dict]:
    """The stretches of the video a clip is made of, in the order they are played: its earlier parts
    (a set-up, the question being answered, montage moments), then its own moment (start-end)."""
    main = {k: clip[k] for k in ("si", "ei", "start", "end") if k in clip}
    return [*clip.get("before", []), main]


def clip_spans(clip: dict) -> list[tuple[float, float]]:
    return [(p["start"], p["end"]) for p in clip_parts(clip)]


def merged_length(spans: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Seconds of [lo, hi] covered by the spans (overlapping spans counted once)."""
    total, reach = 0.0, lo
    for s, e in sorted(spans):
        s, e = max(s, reach), min(e, hi)
        if e > s:
            total += e - s
            reach = e
    return total


def rank_key(c: dict) -> tuple:
    """Best clip first: Claude's ranking of all candidates where there is one, else the score."""
    order = c.get("order")
    return (math.inf if order is None else order, -c["score"], c["start"])


def _overlap(a: dict, b: dict) -> bool:
    return any(s < y and e > x for s, e in clip_spans(a) for x, y in clip_spans(b))


def resolve_overlaps(cands: list[dict]) -> list[dict]:
    """Where clips overlap (any of their parts), or Claude marked two as telling the same story, keep the
    better-ranked one."""
    kept: list[dict] = []
    for c in sorted(cands, key=rank_key):
        clash = any(
            _overlap(c, k)
            or (c.get("duplicate_of") is not None and c.get("duplicate_of") == k.get("id"))
            or (k.get("duplicate_of") is not None and k.get("duplicate_of") == c.get("id"))
            for k in kept
        )
        if not clash:
            kept.append(c)
    return kept


def parts_text(tl: Timeline, before: list[tuple[int, int]], si: int, ei: int) -> str:
    """The clip's words in playing order; " … " marks a jump from one part of the video to another."""
    return " … ".join(tl.text(a, b) for a, b in [*before, (si, ei)])


def make_clip(tl: Timeline, rank: int, si: int, ei: int, base: dict | None = None,
              title: str | None = None, before: list[tuple[int, int]] | None = None) -> dict:
    """The clip record used by every later step (also for clips picked or edited in the UI). before: the
    earlier parts played first, as (first, last) sentences; by default the ones of `base`."""
    base = base or {}
    if before is None:
        before = [(p["si"], p["ei"]) for p in base.get("before", [])]
    before = [(a, b) for a, b in before if b < si]
    text = parts_text(tl, before, si, ei)
    start, end = round(tl.clip_start(si), 3), round(tl.clip_end(ei), 3)
    parts = [part_record(tl, a, b) for a, b in before]
    same_start = base.get("si") == si
    clip = {
        "rank": rank,
        "id": base.get("id") or f"u{si}-{ei}",
        "title": (title or base.get("title") or tl.sentences[si].text[:50]).strip(),
        "hook": base.get("hook") if same_start and base.get("hook") else tl.sentences[si].text,
        "type": base.get("type"),
        "score": base.get("score"),
        "scores": base.get("scores"),
        "start": start, "end": end,
        "duration": round(end - start + sum(p["end"] - p["start"] for p in parts), 3),
        "si": si, "ei": ei,
        "text": text,
        "keywords": [k for k in base.get("keywords", []) if k.lower() in text.lower()],
    }
    if parts:
        clip["before"] = parts
    return clip


def timeline_for(transcript: dict, meta: dict) -> Timeline:
    words = transcript["words"]
    return Timeline(split_sentences(words), words, meta["duration"])


# ---------------------------------------------------------------------------------------------
# Orchestration

CUE_NOTE = """\
Some lines end with notes in square brackets taken from the audio: [loud] means it was said \
noticeably louder than usual; [+1.8s non-speech sound] means a sound without speech follows \
(often laughter or applause); [pause 2.1s] means silence follows."""


EARLIER_NOTE = """\
Earlier parts: sometimes a moment only lands because of something said before it: the set-up of a \
joke or a callback, the question that is being answered, the fact or name the punchline depends on. \
Then add that earlier stretch to the clip in "before": one or a few whole lines from earlier in this \
transcript, with the start time of the first line and the end time of the last, exactly as written. \
They are played first, in the order you list them, and the clip (start-end) follows straight after; \
the viewer gets the set-up, then the payoff, without the minutes in between. The same works for a \
montage: several short moments on one theme, the strongest one last as the clip itself. Use at most \
{n} earlier parts, keep each one short, and only add them when the clip clearly needs them or becomes \
much better; most clips have none ("before": []). The length limit counts all parts together.""".format(
    n=MAX_EARLIER_PARTS)


def _proposal_prompt(title: str, tl: Timeline, a: int, b: int, count: int, min_len: float,
                     max_len: float, avoid: list[tuple[float, float]], cues: dict[int, str],
                     focus: str | None = None, earlier: bool = True) -> str:
    lines = "\n".join(f"{s.start:.1f}-{s.end:.1f} | {s.text}{cues.get(s.i, '')}" for s in tl.sentences[a:b + 1])
    has_cues = any(i in cues for i in range(a, b + 1))
    lo, hi = tl.sentences[a].start, tl.sentences[b].end
    avoid_txt = ""
    if avoid:
        spans = "\n".join(f"- {s:.1f}-{e:.1f}" for s, e in avoid)
        avoid_txt = ("\nThese time ranges are already taken. Do not propose anything that overlaps them:\n"
                     f"{spans}\n")
    target = round((min_len + max_len) / 2)
    focus_txt = ""
    if focus:
        focus_txt = (f"\nThe person making the clips is looking for: {focus.strip()}\nOnly propose moments that "
                     "fit this. If nothing in this part fits, return an empty list.\n")
    earlier_txt = f"\n{EARLIER_NOTE}\n" if earlier else ""
    example_before = ' "before": [{"start": 61.2, "end": 66.8}],' if earlier else ""
    return f"""\
Video title: {title}
{focus_txt}
Below is the transcript from {fmt_time(lo)} to {fmt_time(hi)}. Each line is one sentence: its \
start and end time in seconds, then the text.

<transcript>
{lines}
</transcript>
{CUE_NOTE if has_cues else ""}
Find up to {count} of the best moments in this part to cut into standalone short clips.
- Each clip must be {min_len:.0f}-{max_len:.0f} seconds long; around {target} seconds is ideal.
- Start on the sentence that grabs attention (the hook), never in the middle of a thought.
- End when the idea or story pays off.
- Clips must not overlap each other, and each must be about a different point or story.
- Use the start time of the clip's first line and the end time of its last line, exactly as written.
- Only propose clips that are genuinely good. Fewer is fine; an empty list is fine.
{avoid_txt}{earlier_txt}
{RUBRIC}

Reply with only a JSON object in exactly this format:
{{"clips": [{{"start": 123.4, "end": 161.0,{example_before} "title": "Short catchy title", \
"hook": "The opening line, quoted from the transcript", "type": "joke", \
"scores": {{"hook": 8, "payoff": 7, "reaction": 6, "pace": 8}}, \
"keywords": ["the word that lands"]}}]}}
- title: in the language of the video, at most 50 characters; it is shown on screen as the clip starts.
- keywords: 1-3 words or short phrases from the clip that carry its punch; they get highlighted \
in the captions."""


def _rescore_prompt(title: str, cands: list[dict], focus: str | None = None,
                    audio: dict[str, str] | None = None) -> str:
    """audio: per clip id, a note on what the audio shows inside the clip (laughter, applause), if anything."""
    audio = audio or {}
    focus_txt = (f"\n\nThe person making the clips is looking for: {focus}. Clips that don't fit this "
                 "go to the bottom.") if focus else ""
    def when(c: dict) -> str:
        parts = clip_parts(c)
        length = sum(p["end"] - p["start"] for p in parts)
        spans = " + ".join(f"{fmt_time(p['start'])}-{fmt_time(p['end'])}" for p in parts)
        return f"{spans} ({length:.0f}s{', earlier parts played first' if len(parts) > 1 else ''})"

    blocks = "\n\n".join(
        f"[{c['id']}] {when(c)} {c.get('type') or 'other'} \"{c['title']}\"\n{c['text']}"
        + (f"\n({audio[c['id']]})" if audio.get(c["id"]) else "")
        for c in cands
    )
    return f"""\
Video title: {title}

These candidate clips come from different parts of the same video. They were judged separately, \
so their scores are not comparable. Compare them with each other: put them in order, from the one \
most likely to do well as a short clip to the least, and score each again on one common scale. \
Between two clips of the same kind that would land the same way, the better one goes first and the \
other goes down the list, so the top of the ranking has some variety.{focus_txt}

{RUBRIC}

<clips>
{blocks}
</clips>
{"Notes in brackets under a clip say what the audio shows inside it." if audio else ""}\
{" Where a clip jumps from an earlier part of the video to a later one, its text shows … there." if any(c.get("before") for c in cands) else ""}
If a clip makes the same point or tells the same story as a better clip in the list, set its \
"duplicate_of" to that better clip's id; otherwise set it to null.

Reply with only a JSON object listing every clip id above exactly once, best first, in exactly \
this format:
{{"ranking": [{{"id": "{cands[0]['id']}", "hook": 8, "payoff": 7, "reaction": 6, "pace": 8, \
"duplicate_of": null}}]}}"""


def _trim_prompt(title: str, tl: Timeline, jobs: dict[str, tuple[int, int, str, float]], min_len: float,
                 max_len: float) -> str:
    """jobs: {trim id: (first sentence, last sentence, title, seconds of earlier parts played first)}."""
    blocks = []
    for tid, (si, ei, clip_title, extra) in jobs.items():
        t0 = tl.sentences[si].start
        lines = "\n".join(f"{n}. [{s.start - t0:.1f}-{s.end - t0:.1f}] {s.text}"
                          for n, s in enumerate(tl.sentences[si:ei + 1], 1))
        lo, hi = main_limits(min_len, max_len, extra)
        length = f' seconds="{lo:.0f}-{hi - 1:.0f}"' if extra else ""
        blocks.append(f'<clip id="{tid}" title="{clip_title}"{length}>\n{lines}\n</clip>')
    example = next(iter(jobs))
    return f"""\
Video title: {title}

These clips came out too long once aligned to whole sentences. For each one, choose a single \
continuous run of its numbered lines to keep: start on a strong hook, keep the payoff, and make \
it {min_len:.0f}-{max_len - 1:.0f} seconds long (the times are seconds from the start of the clip). \
A clip marked with seconds="..." is played after earlier parts of the video; its kept lines must be \
that long instead.

{chr(10).join(blocks)}

Reply with only a JSON object with one entry per clip, in exactly this format:
{{"trims": [{{"id": "{example}", "first": 1, "last": 9}}]}}"""


class Selector:
    def __init__(self, work: Path, meta: dict, transcript: dict, *, min_len: float, max_len: float,
                 model: str, claude_factory: Callable[[], Claude], force: bool = False,
                 focus: str | None = None, time_range: tuple[float | None, float | None] = (None, None),
                 new_search: bool = False, also_avoid: list[tuple[float, float]] = (), earlier: bool = True):
        """new_search: ask for clips at moments not suggested yet (nor in also_avoid), even if these
        options were used before. earlier: Claude may put earlier parts of the video in front of a clip."""
        words = transcript["words"]
        self.sentences = split_sentences(words)
        if not self.sentences:
            raise ClipperError("The transcript is empty.")
        self.cues = sentence_cues(Loudness.from_work(work), words, self.sentences, meta["duration"])
        self.title = meta.get("title", "")
        self.min_len, self.max_len = min_len, max_len
        lo, hi = time_range
        pool = [s for s in self.sentences
                if (lo is None or s.start >= lo - 0.5) and (hi is None or s.end <= hi + 0.5)]
        if not pool:
            raise ClipperError(f"No speech between {fmt_time(lo or 0)} and {fmt_time(hi or meta['duration'])}.")
        self.tl = Timeline(self.sentences, words, meta["duration"], pool[0].i, pool[-1].i)
        self.focus = (focus or "").strip() or None
        self.earlier = earlier
        self.chunks = make_chunks(pool, overlap=max_len + 30)
        self._claude_factory, self._claude = claude_factory, None
        self._pending: list[tuple[dict, str, int, int]] = []
        self.path = work / "select.json"
        params = {"version": SELECT_VERSION, "transcript": fingerprint(words), "min_len": min_len,
                  "max_len": max_len, "model": model, "chunks": [list(c) for c in self.chunks],
                  "focus": self.focus}
        if not earlier:
            params["earlier"] = False  # only when off, so searches made before this option existed still count
        previous = read_json(self.path) if self.path.exists() else None
        old = previous if not force else None
        if old and {k: old["params"].get(k) for k in ("version", "transcript")} != \
                {k: params[k] for k in ("version", "transcript")}:
            old = None  # suggestions for another transcript don't line up with this one's sentences
        if old and old["params"] == params and not new_search:
            state = old
        else:
            # A new search: what was suggested before stays (for the review screen); when asked for new clips,
            # Claude also steers clear of it.
            earlier = (old.get("earlier", []) + old["candidates"]) if old else []
            state = {"params": params, "done": [], "candidates": [], "rescored": {},
                     "search": old.get("search", 1) + 1 if old else 1, "earlier": earlier,
                     "avoid": [list(x) for x in sorted({span for c in earlier for span in clip_spans(c)}
                                                       | {tuple(x) for x in also_avoid})] if new_search else []}
        for key, default in (("search", 1), ("earlier", []), ("avoid", [])):
            state.setdefault(key, default)
        self.state = state
        # Ids already handed out, also by searches thrown away (--force) and for clips already rendered:
        # a new suggestion must never share an id with them (the review screen and posts go by id).
        rendered = work / "output" / "clips.json"
        previous = previous or {}
        self._taken = {c.get("id") for c in previous.get("earlier", []) + previous.get("candidates", [])}
        self._taken |= {c.get("id") for c in (read_json(rendered) if rendered.exists() else [])}
        self._taken |= {c["id"] for c in state["earlier"] + state["candidates"]}

    @property
    def claude(self) -> Claude:
        if self._claude is None:
            self._claude = self._claude_factory()
        return self._claude

    def _save(self) -> None:
        write_json(self.path, self.state)

    def _chunk_bounds(self, k: int) -> tuple[float, float]:
        a, b = self.chunks[k]
        return self.sentences[a].start, self.sentences[b].end

    def _earlier_parts(self, proposal: dict, si: int) -> tuple[list[tuple[int, int]], int]:
        """The proposal's earlier parts as (first, last) sentences, and the clip's first sentence. Parts that
        reach into the clip or into each other are dropped; one that runs straight into the clip just makes
        it start earlier. Parts are dropped from the front while they leave the clip too little time."""
        parts: list[tuple[int, int]] = []
        for b in (proposal.get("before") or [])[:MAX_EARLIER_PARTS] if self.earlier else []:
            a, z = snap(self.tl, b["start"], b["end"])
            if z < si and not any(a <= y and z >= x for x, y in parts):
                parts.append((a, z))
        if parts and parts[-1][1] == si - 1:
            si = parts.pop()[0]
        while parts and sum(self.tl.dur(a, z) for a, z in parts) > self.max_len - MIN_MAIN_PART:
            parts.pop(0)
        return parts, si

    def _add(self, proposal: dict, call: str) -> None:
        si, ei = snap(self.tl, proposal["start"], proposal["end"])
        before, si = self._earlier_parts(proposal, si)
        extra = sum(self.tl.dur(a, z) for a, z in before)
        if ei > si and self.tl.dur(si, ei) > self.max_len - extra:
            self._pending.append((proposal, call, si, ei, before))  # Claude decides what to cut, see _trim
        else:
            self._commit(proposal, call, si, ei, before)

    def _commit(self, proposal: dict, call: str, si: int, ei: int, before: list[tuple[int, int]] = (),
                trimmed: bool = False) -> None:
        before = list(before)
        extra = sum(self.tl.dur(a, z) for a, z in before)
        lo, hi = main_limits(self.min_len, self.max_len, extra)
        fitted = fit(self.tl, si, ei, lo, hi, floor=max(z for _, z in before) + 1 if before else None)
        if fitted is None:
            log(f"select: dropped \"{proposal['title']}\": can't fit {self.min_len:.0f}-{self.max_len:.0f}s")
            return
        si, ei = fitted
        text = parts_text(self.tl, before, si, ei)
        cands = self.state["candidates"]
        n = len(self.state["earlier"]) + len(cands) + 1
        while f"c{n}" in self._taken:
            n += 1
        self._taken.add(f"c{n}")
        cands.append({
            "id": f"c{n}",  # unique across searches
            "call": call,
            "search": self.state["search"],
            "focus": self.focus,
            "title": proposal["title"].strip().strip('"'),
            "hook": proposal["hook"].strip(),
            "type": proposal.get("type") or "other",
            "scores": proposal["scores"],
            "score": combine(proposal["scores"]),
            "keywords": [k.strip() for k in proposal.get("keywords", [])
                         if k.strip() and k.strip().lower() in text.lower()][:3],
            "proposed": [proposal["start"], proposal["end"]],
            "trimmed_by_claude": trimmed,
            "si": si, "ei": ei,
            "start": round(self.tl.clip_start(si), 3),
            "end": round(self.tl.clip_end(ei), 3),
            "text": text,
            **({"before": [part_record(self.tl, a, z) for a, z in before]} if before else {}),
        })
        if before:
            log(f"select: \"{proposal['title']}\" plays {len(before)} earlier part(s) first")

    def _trim(self) -> None:
        """Ask Claude which sentences to keep for proposals that ran over the maximum length."""
        if not self._pending:
            return
        jobs = {f"t{n}": item for n, item in enumerate(self._pending, 1)}
        self._pending = []
        log(f"select: asking Claude where to shorten {len(jobs)} clip(s) longer than {self.max_len:.0f}s")
        extra = {tid: sum(self.tl.dur(a, z) for a, z in before) for tid, (*_, before) in jobs.items()}
        spans = {tid: (si, ei, extra[tid]) for tid, (_, _, si, ei, _) in jobs.items()}
        prompt = _trim_prompt(self.title, self.tl,
                              {tid: (si, ei, p["title"], extra[tid]) for tid, (p, _, si, ei, _) in jobs.items()},
                              self.min_len, self.max_len)
        try:
            trims = self.claude.ask(SYSTEM_PROMPT, prompt,
                                    lambda r: parse_trims(r, spans, self.tl, self.min_len, self.max_len),
                                    what="trimming")
        except Cancelled:
            raise
        except ClipperError as e:
            log(f"select: {e}; trimming from the end instead")
            trims = {}
        for tid, (proposal, call, si, ei, before) in jobs.items():
            if tid in trims:
                first, last = trims[tid]
                self._commit(proposal, call, si + first - 1, si + last - 1, before, trimmed=True)
            else:
                self._commit(proposal, call, si, ei, before)

    def _round(self, key: str, count_for_chunk: Callable[[int], int], with_avoid: bool) -> None:
        jobs = []
        for k, (a, b) in enumerate(self.chunks):
            call = f"{key}:{k}"
            if call in self.state["done"]:
                continue
            lo, hi = self._chunk_bounds(k)
            avoid = [(s, e) for s, e in self.state["avoid"]]
            if with_avoid:
                avoid += [span for c in self.state["candidates"] for span in clip_spans(c)]
            avoid = sorted((s, e) for s, e in avoid if s < hi and e > lo)
            if avoid and hi - lo - merged_length(avoid, lo, hi) < self.min_len:
                self.state["done"].append(call)
                continue
            jobs.append((k, call, count_for_chunk(k), avoid))
        if not jobs:
            return
        claude = self.claude

        def ask(job: tuple) -> list[dict]:
            k, _, count, avoid = job
            (a, b), (lo, hi) = self.chunks[k], self._chunk_bounds(k)
            where = f"chunk {k + 1}/{len(self.chunks)}" if len(self.chunks) > 1 else "transcript"
            log(f"select: asking Claude for up to {count} clips ({where}, {fmt_time(lo)}-{fmt_time(hi)})")
            prompt = _proposal_prompt(self.title, self.tl, a, b, count, self.min_len, self.max_len, avoid,
                                      self.cues, self.focus, self.earlier)
            return claude.ask(SYSTEM_PROMPT, prompt, lambda r: parse_proposals(r, lo, hi),
                              what=f"proposals for {where}")

        # The parts of a long video are asked about at the same time; the answers are taken in order,
        # so each part's trims and its place in select.json are the same as when asked one by one.
        # If one part fails, the answers for the others are still kept, so a new run only asks again
        # about the part that failed.
        pool = thread_pool(min(CLAUDE_AT_ONCE, len(jobs)))
        failed: ClipperError | None = None
        try:
            futures = [pool.submit(ask, job) for job in jobs]
            for (k, call, count, _), future in zip(jobs, futures):
                try:
                    proposals = future.result()
                except Cancelled:
                    raise
                except ClipperError as e:
                    failed = failed or e
                    continue
                for p in proposals[:count]:
                    self._add(p, call)
                self._trim()
                self.state["done"].append(call)
                self._save()
        finally:
            pool.shutdown(cancel_futures=True)
        if failed:
            raise failed

    def _audio_note(self, c: dict) -> str:
        """What the audio cues say about a candidate, for the ranking prompt (the text alone has lost them)."""
        notes = [self.cues[i] for i in range(c["si"], c["ei"] + 1) if i in self.cues]
        sounds = sum("non-speech sound" in n for n in notes)
        loud = sum("[loud" in n for n in notes)
        parts = []
        if sounds:
            parts.append(f"non-speech sound, probably laughter or applause, after {sounds} line(s)")
        if loud:
            parts.append(f"{loud} line(s) said noticeably louder than usual")
        return "audio: " + "; ".join(parts) if parts else ""

    def _rescore(self) -> None:
        cands = self.state["candidates"]
        if len({c["call"] for c in cands}) < 2 or len(cands) < 2:
            return
        key = fingerprint([(c["id"], c["si"], c["ei"], *[(p["si"], p["ei"]) for p in c.get("before", [])])
                           for c in cands])
        scores = self.state["rescored"].get(key)
        if scores is None:
            log(f"select: asking Claude to rank all {len(cands)} candidates against each other")
            ids = [c["id"] for c in cands]
            audio = {c["id"]: self._audio_note(c) for c in cands}
            scores = self.claude.ask(SYSTEM_PROMPT, _rescore_prompt(self.title, cands, self.focus, audio),
                                     lambda r: parse_rescores(r, ids), what="re-scoring")
            self.state["rescored"] = {key: scores}  # only the latest set is ever needed again
            self._save()
        for c in cands:
            sc = dict(scores[c["id"]])
            c["duplicate_of"] = sc.pop("duplicate_of", None)
            c["order"] = sc.pop("order")
            c["scores"] = sc
            c["score"] = combine(sc)

    def _survivors(self) -> list[dict]:
        self._rescore()
        avoid = self.state["avoid"]
        # Claude was asked to stay clear of earlier suggestions; aligning to sentences can still overlap a little.
        fresh = [c for c in self.state["candidates"]
                 if merged_length(avoid, c["start"], c["end"]) <= 0.25 * (c["end"] - c["start"])]
        return resolve_overlaps(fresh)

    def run(self, num_clips: int) -> list[dict]:
        span_min = (self._chunk_bounds(0)[1] - self._chunk_bounds(0)[0]) / 60
        first_count = min(10, max(3, math.ceil(span_min * 0.6), math.ceil(num_clips * 1.5 / len(self.chunks))))
        self._round("initial", lambda k: first_count, with_avoid=False)
        survivors = self._survivors()

        for r in range(1, EXTRA_ROUNDS + 1):
            if len(survivors) >= num_clips:
                break
            key = f"more{num_clips}-{r}"
            cached_round = all(f"{key}:{k}" in self.state["done"] for k in range(len(self.chunks)))
            if not cached_round:
                log(f"select: only {len(survivors)} usable clips so far, asking Claude for more")
            more = min(10, math.ceil((num_clips - len(survivors)) * 1.5 / len(self.chunks)) + 1)
            before = len(self.state["candidates"])
            self._round(key, lambda k: more, with_avoid=True)
            if len(self.state["candidates"]) == before and not cached_round:
                break  # Claude has nothing more to offer
            survivors = self._survivors()
        self._save()

        if len(survivors) < num_clips:
            log(f"select: found {len(survivors)} good clips (asked for {num_clips})")
        return [make_clip(self.tl, n, c["si"], c["ei"], c) for n, c in enumerate(survivors[:num_clips], 1)]


def select_clips(work: Path, meta: dict, transcript: dict, *, num_clips: int, min_len: float,
                 max_len: float, model: str, claude_factory: Callable[[], Claude],
                 force: bool = False, focus: str | None = None,
                 time_range: tuple[float | None, float | None] = (None, None),
                 new_search: bool = False, also_avoid: list[tuple[float, float]] = (),
                 earlier: bool = True) -> list[dict]:
    if min_len <= 0 or max_len < min_len:
        raise ClipperError("--min-len must be positive and no larger than --max-len")
    sel = Selector(work, meta, transcript, min_len=min_len, max_len=max_len, model=model,
                   claude_factory=claude_factory, force=force, focus=focus, time_range=time_range,
                   new_search=new_search, also_avoid=also_avoid, earlier=earlier)
    if new_search and sel.state["avoid"]:
        log(f"select: looking for new clips, away from the {len(sel.state['avoid'])} suggested or made before")
    clips = sel.run(num_clips)
    write_json(work / "clips.json", clips)
    for c in clips:
        spans = " + ".join(f"{fmt_time(s)}-{fmt_time(e)}" for s, e in clip_spans(c))
        log(f"select: #{c['rank']} [{c['score']}] {spans} "
            f"({c['duration']:.0f}s) {c.get('type') or 'other'}: {c['title']}")
    return clips


def saved_candidates(work: Path) -> list[dict]:
    """Every clip Claude has suggested for this video, earlier searches first."""
    path = work / "select.json"
    if not path.exists():
        return []
    state = read_json(path)
    return state.get("earlier", []) + state["candidates"]


def clips_from_review(transcript: dict, meta: dict, candidates: list[dict], items: list[dict]) -> list[dict]:
    """Clips chosen and adjusted by hand: items are {"id"?, "si", "ei", "title"?, "before"?} in the wanted
    order. before: the earlier parts played first, as [first, last] sentences; without it, a suggestion
    keeps its own."""
    tl = timeline_for(transcript, meta)
    by_id = {c["id"]: c for c in candidates}
    n = len(tl.sentences)
    out = []
    for rank, item in enumerate(items, 1):
        si, ei = int(item["si"]), int(item["ei"])
        if not 0 <= si <= ei < n:
            raise ClipperError(f"Clip {rank}: sentences {si}-{ei} are outside the transcript")
        before = None
        if item.get("before") is not None:
            before = [(int(a), int(b)) for a, b in item["before"]]
            for k, (a, b) in enumerate(before):
                if not 0 <= a <= b < si or any(a <= y and b >= x for x, y in before[:k]):
                    raise ClipperError(f"Clip {rank}: earlier part {a}-{b} must come before the clip "
                                       "and not overlap its other parts")
        out.append(make_clip(tl, rank, si, ei, by_id.get(item.get("id")), item.get("title"), before))
    return out
