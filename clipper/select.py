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

from pydantic import BaseModel, Field, ValidationError

from .llm import Claude, InvalidReply, extract_json
from .util import ClipperError, fingerprint, fmt_time, log, read_json, write_json

SELECT_VERSION = 1
CHUNK_SECONDS = 15 * 60
MAX_SENTENCE_SECONDS = 15.0  # longer "sentences" (run-ons, missing punctuation) get split
SENTENCE_GAP = 2.0           # a pause this long ends a sentence even without punctuation
PAD_BEFORE, PAD_AFTER = 0.2, 0.4
EXTRA_ROUNDS = 2             # how many times to ask Claude for more when too few clips survive

_END = (".", "?", "!", "…", "。", "？", "！")
_TRAILING = "\"'”’)»]"
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "e.g.", "i.e.", "jr.", "sr."}

SYSTEM_PROMPT = """\
You are a senior short-form video editor. You cut long videos (podcasts, interviews, talks, \
streams) into vertical clips for TikTok, Instagram Reels and YouTube Shorts. You know what makes \
people stop scrolling and watch to the end: a strong first line, one complete idea or story with a \
payoff, and no need for outside context. You judge strictly. You answer with JSON only."""

RUBRIC = """\
Score each clip with four sub-scores, each from 0 to 25:
- hook: how strongly the first line or two make a viewer stop scrolling
- standalone: how well it makes sense to someone who has not seen the rest of the video
- emotion: how much emotion, surprise or humour it carries
- clarity: how clear and easy to follow it is (no rambling, false starts, or references to \
things only visible on screen)
A clip's score is the sum of its sub-scores (0-100). Be critical: a total above 80 is only for \
clips that would do really well on their own."""


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
        return [(0, len(sentences) - 1)]
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
    hook: int = Field(ge=0, le=25)
    standalone: int = Field(ge=0, le=25)
    emotion: int = Field(ge=0, le=25)
    clarity: int = Field(ge=0, le=25)


class ProposedClip(BaseModel):
    start: float
    end: float
    title: str = Field(min_length=1, max_length=120)
    hook: str = Field(min_length=1)
    scores: SubScores


class ProposalReply(BaseModel):
    clips: list[ProposedClip]


class Rescore(SubScores):
    id: str


class RescoreReply(BaseModel):
    scores: list[Rescore]


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
    if problems:
        raise InvalidReply("\n".join(problems))
    return [c.model_dump() for c in parsed.clips]


def parse_rescores(reply: str, ids: list[str]) -> dict[str, dict]:
    parsed = _validate(RescoreReply, reply)
    got = [s.id for s in parsed.scores]
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
    if problems:
        raise InvalidReply("; ".join(problems) + ". Give exactly one entry per clip id.")
    return {s.id: s.model_dump(exclude={"id"}) for s in parsed.scores}


# ---------------------------------------------------------------------------------------------
# Timing fixes

class Timeline:
    def __init__(self, sentences: list[Sentence], words: list[dict], duration: float):
        self.sentences, self.words, self.duration = sentences, words, duration
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
        near = [i for i in (k - 1, k) if 0 <= i < len(self.sentences)]
        return min(near, key=lambda i: abs(self.sentences[i].start - t))

    def nearest_end(self, t: float, at_least: int) -> int:
        return min(range(at_least, len(self.sentences)), key=lambda i: abs(self.sentences[i].end - t))

    def text(self, si: int, ei: int) -> str:
        a, b = self.sentences[si].w0, self.sentences[ei].w1
        return "".join(w["w"] for w in self.words[a:b + 1]).strip()


def snap_and_fit(tl: Timeline, start: float, end: float, min_len: float, max_len: float) -> tuple[int, int] | None:
    """Move a proposal to sentence boundaries, then trim or extend it to fit the length limits.

    Trimming happens at the end (the start holds the hook). Extending adds the next sentence,
    or the previous one when there is no room after. Returns None if the clip can't be made to fit.
    """
    n = len(tl.sentences)
    si = tl.nearest_start(start)
    ei = tl.nearest_end(end, si)
    while ei > si and tl.dur(si, ei) > max_len:
        ei -= 1
    if tl.dur(si, ei) > max_len:
        return None
    while tl.dur(si, ei) < min_len:
        if ei + 1 < n and tl.dur(si, ei + 1) <= max_len:
            ei += 1
        elif si > 0 and tl.dur(si - 1, ei) <= max_len:
            si -= 1
        else:
            return None
    return si, ei


def resolve_overlaps(cands: list[dict]) -> list[dict]:
    """Where clips overlap, keep the higher-scoring one."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: (-c["score"], c["start"])):
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in kept):
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------------------------
# Orchestration

def _proposal_prompt(title: str, tl: Timeline, a: int, b: int, count: int, min_len: float,
                     max_len: float, avoid: list[tuple[float, float]]) -> str:
    lines = "\n".join(f"{s.start:.1f}-{s.end:.1f} | {s.text}" for s in tl.sentences[a:b + 1])
    lo, hi = tl.sentences[a].start, tl.sentences[b].end
    avoid_txt = ""
    if avoid:
        spans = "\n".join(f"- {s:.1f}-{e:.1f}" for s, e in avoid)
        avoid_txt = ("\nThese time ranges are already taken. Do not propose anything that overlaps them:\n"
                     f"{spans}\n")
    target = round((min_len + max_len) / 2)
    return f"""\
Video title: {title}

Below is the transcript from {fmt_time(lo)} to {fmt_time(hi)}. Each line is one sentence: its \
start and end time in seconds, then the text.

<transcript>
{lines}
</transcript>

Find up to {count} of the best moments in this part to cut into standalone short clips.
- Each clip must be {min_len:.0f}-{max_len:.0f} seconds long; around {target} seconds is ideal.
- Start on the sentence that grabs attention (the hook), never in the middle of a thought.
- End when the idea or story pays off.
- Clips must not overlap each other.
- Use the start time of the clip's first line and the end time of its last line, exactly as written.
- Only propose clips that are genuinely good. Fewer is fine; an empty list is fine.
{avoid_txt}
{RUBRIC}

Reply with only a JSON object in exactly this format:
{{"clips": [{{"start": 123.4, "end": 161.0, "title": "Short catchy title", \
"hook": "The opening line, quoted from the transcript", \
"scores": {{"hook": 20, "standalone": 18, "emotion": 15, "clarity": 22}}}}]}}
Write the title in the language of the video, at most 60 characters."""


def _rescore_prompt(title: str, cands: list[dict]) -> str:
    blocks = "\n\n".join(
        f"[{c['id']}] {fmt_time(c['start'])}-{fmt_time(c['end'])} ({c['end'] - c['start']:.0f}s) "
        f"\"{c['title']}\"\n{c['text']}"
        for c in cands
    )
    return f"""\
Video title: {title}

These candidate clips come from different parts of the same video. They were scored separately, \
so their scores are not comparable. Score all of them again on one common scale, comparing them \
with each other.

{RUBRIC}

<clips>
{blocks}
</clips>

Reply with only a JSON object with one entry for every clip id above, in exactly this format:
{{"scores": [{{"id": "{cands[0]['id']}", "hook": 20, "standalone": 18, "emotion": 15, "clarity": 22}}]}}"""


class Selector:
    def __init__(self, work: Path, meta: dict, transcript: dict, *, min_len: float, max_len: float,
                 model: str, claude_factory: Callable[[], Claude], force: bool = False):
        words = transcript["words"]
        self.sentences = split_sentences(words)
        if not self.sentences:
            raise ClipperError("The transcript is empty.")
        self.tl = Timeline(self.sentences, words, meta["duration"])
        self.title = meta.get("title", "")
        self.min_len, self.max_len = min_len, max_len
        self.chunks = make_chunks(self.sentences, overlap=max_len + 30)
        self._claude_factory, self._claude = claude_factory, None
        self.path = work / "select.json"
        params = {"version": SELECT_VERSION, "transcript": fingerprint(words), "min_len": min_len,
                  "max_len": max_len, "model": model, "chunks": [list(c) for c in self.chunks]}
        state = read_json(self.path) if self.path.exists() and not force else None
        if not state or state.get("params") != params:
            state = {"params": params, "done": [], "candidates": [], "rescored": {}}
        self.state = state

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

    def _add(self, proposal: dict, call: str) -> None:
        fit = snap_and_fit(self.tl, proposal["start"], proposal["end"], self.min_len, self.max_len)
        if fit is None:
            log(f"select: dropped \"{proposal['title']}\": can't fit {self.min_len:.0f}-{self.max_len:.0f}s")
            return
        si, ei = fit
        cands = self.state["candidates"]
        cands.append({
            "id": f"c{len(cands) + 1}",
            "call": call,
            "title": proposal["title"].strip().strip('"'),
            "hook": proposal["hook"].strip(),
            "scores": proposal["scores"],
            "score": sum(proposal["scores"].values()),
            "proposed": [proposal["start"], proposal["end"]],
            "si": si, "ei": ei,
            "start": round(self.tl.clip_start(si), 3),
            "end": round(self.tl.clip_end(ei), 3),
            "text": self.tl.text(si, ei),
        })

    def _round(self, key: str, count_for_chunk: Callable[[int], int], with_avoid: bool) -> None:
        for k, (a, b) in enumerate(self.chunks):
            call = f"{key}:{k}"
            if call in self.state["done"]:
                continue
            lo, hi = self._chunk_bounds(k)
            avoid = []
            if with_avoid:
                avoid = sorted((c["start"], c["end"]) for c in self.state["candidates"]
                               if c["start"] < hi and c["end"] > lo)
                taken = sum(min(e, hi) - max(s, lo) for s, e in avoid)
                if hi - lo - taken < self.min_len:
                    self.state["done"].append(call)
                    continue
            count = count_for_chunk(k)
            where = f"chunk {k + 1}/{len(self.chunks)}" if len(self.chunks) > 1 else "transcript"
            log(f"select: asking Claude for up to {count} clips ({where}, {fmt_time(lo)}-{fmt_time(hi)})")
            prompt = _proposal_prompt(self.title, self.tl, a, b, count, self.min_len, self.max_len, avoid)
            proposals = self.claude.ask(SYSTEM_PROMPT, prompt, lambda r: parse_proposals(r, lo, hi),
                                        what=f"proposals for {where}")
            for p in proposals[:count]:
                self._add(p, call)
            self.state["done"].append(call)
            self._save()

    def _rescore(self) -> None:
        cands = self.state["candidates"]
        if len({c["call"] for c in cands}) < 2 or len(cands) < 2:
            return
        key = fingerprint([(c["id"], c["si"], c["ei"]) for c in cands])
        scores = self.state["rescored"].get(key)
        if scores is None:
            log(f"select: asking Claude to re-score all {len(cands)} candidates on one scale")
            ids = [c["id"] for c in cands]
            scores = self.claude.ask(SYSTEM_PROMPT, _rescore_prompt(self.title, cands),
                                     lambda r: parse_rescores(r, ids), what="re-scoring")
            self.state["rescored"] = {key: scores}  # only the latest set is ever needed again
            self._save()
        for c in cands:
            c["scores"] = scores[c["id"]]
            c["score"] = sum(scores[c["id"]].values())

    def _survivors(self) -> list[dict]:
        self._rescore()
        return resolve_overlaps(self.state["candidates"])

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
        picked = survivors[:num_clips]
        return [
            {"rank": n, **{k: c[k] for k in ("id", "title", "hook", "score", "scores", "start", "end", "text")},
             "duration": round(c["end"] - c["start"], 3)}
            for n, c in enumerate(picked, 1)
        ]


def select_clips(work: Path, meta: dict, transcript: dict, *, num_clips: int, min_len: float,
                 max_len: float, model: str, claude_factory: Callable[[], Claude],
                 force: bool = False) -> list[dict]:
    if min_len <= 0 or max_len < min_len:
        raise ClipperError("--min-len must be positive and no larger than --max-len")
    sel = Selector(work, meta, transcript, min_len=min_len, max_len=max_len, model=model,
                   claude_factory=claude_factory, force=force)
    clips = sel.run(num_clips)
    write_json(work / "clips.json", clips)
    for c in clips:
        log(f"select: #{c['rank']} [{c['score']}] {fmt_time(c['start'])}-{fmt_time(c['end'])} "
            f"({c['duration']:.0f}s) {c['title']}")
    return clips
