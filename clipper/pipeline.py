"""The six steps wired together, shared by the command line and the web UI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from pathlib import Path
from typing import Callable

from .audio import Loudness
from .captions import PRESETS, Watermark, _phrases, build_srt, keyword_positions, make_captions, make_style, map_words
from .edit import MODES as JUMP_CUT_MODES
from .edit import EditMap, plan_cuts
from .edits import ClipEdits, apply_cuts, load_edits
from .formats import Format, get_format, parse_formats, parse_time
from .ingest import DOWNLOAD_MODES, clip_video, ingest
from .llm import Claude
from .posts import format_post, write_posts
from .reframe import LAYOUTS, reframe
from .render import ENCODERS, LOGO_POSITIONS, Look, Piece, check_ffmpeg_has_libass, logo_size, read_logo, render
from .select import clip_parts, clip_spans, clips_from_review, saved_candidates, select_clips
from .transcribe import clip_words, transcribe, words_in
from .translate import spread, translate_clip
from .util import (
    Cancelled, ClipperError, check_cancel, log, read_json, report, require_tool, slugify, thread_pool,
    write_json,
)

STEPS = ["ingest", "transcribe", "select", "reframe", "captions", "render"]
SECTIONS_AT_ONCE = 3  # clip sections fetched in the background while earlier clips are being made


@dataclass
class Options:
    num_clips: int = 3
    min_len: float = 20
    max_len: float = 60
    prompt: str | None = None           # what Claude should look for, e.g. "funny moments"
    range_start: float | str | None = None  # only use this part of the video (seconds or "12:30")
    range_end: float | str | None = None
    work_dir: str = "work"
    download: str = "sections"          # links: sections (audio first, video only for the clips) | full
    model: str = "opus"                 # Claude model that picks the clips
    whisper_model: str = "small"
    whisper_backend: str = "auto"       # auto | faster-whisper | mlx
    language: str | None = None
    caption_model: str = "turbo"        # re-transcribes the chosen clips; "none" to skip
    caption_language: str | None = None  # translate the captions, e.g. "en" or "English"
    caption_style: str = "bold"         # bold | clean | boxed
    font: str | None = None
    caption_size: int | None = None
    highlight_color: str | None = None
    uppercase: bool = False
    caption_position: str = "lower"     # lower | middle
    title_overlay: bool = True
    formats: list[str] = field(default_factory=lambda: ["9:16"])  # 9:16, 4:5, 1:1, 16:9
    zoom: bool = True                   # punch in on the hook and the keywords
    progress_bar: bool = False
    logo: str | None = None             # image file, ideally a PNG with transparency
    logo_position: str = "top-right"
    watermark: str | None = None        # your own text on every clip, e.g. "@yourname"
    watermark_position: str = "bottom-left"
    watermark_opacity: int = 90         # percent, for the watermark text and the logo
    jump_cuts: str = "on"               # on | pauses | off
    earlier_parts: bool = True          # Claude may play earlier parts of the video (a set-up) before a clip
    layout: str = "auto"                # auto | follow | split | main
    encoder: str = "auto"               # auto (the Mac's hardware encoder if there is one) | x264 | videotoolbox
    crf: int = 20
    preset: str = "medium"
    posts: bool = True
    force: str | None = None
    stop_after: str | None = None

    def validate(self) -> "Options":
        problems = []
        if self.num_clips < 1:
            problems.append("number of clips must be at least 1")
        if self.min_len <= 0 or self.max_len < self.min_len:
            problems.append("minimum length must be positive and no larger than the maximum")
        for name, value, allowed in [
            ("caption style", self.caption_style, tuple(PRESETS)), ("jump cuts", self.jump_cuts, JUMP_CUT_MODES),
            ("layout", self.layout, LAYOUTS), ("encoder", self.encoder, ENCODERS),
            ("caption position", self.caption_position, ("lower", "middle")),
            ("download", self.download, DOWNLOAD_MODES), ("logo position", self.logo_position, LOGO_POSITIONS),
            ("watermark position", self.watermark_position, LOGO_POSITIONS),
            ("whisper backend", self.whisper_backend, ("auto", "faster-whisper", "mlx")),
            ("force", self.force, (None, *STEPS)), ("stop after", self.stop_after, (None, *STEPS)),
        ]:
            if value not in allowed:
                problems.append(f"{name} must be one of {', '.join(str(a) for a in allowed if a)}")
        try:
            self.formats = parse_formats(self.formats)
            self.range_start, self.range_end = parse_time(self.range_start), parse_time(self.range_end)
            if self.range_start is not None and self.range_end is not None and self.range_end <= self.range_start:
                problems.append("the end of the time range must come after its start")
            self.style  # noqa: B018 - builds the caption style, which checks font size and colour
        except ValueError as e:
            problems.append(str(e))
        if not 10 <= self.watermark_opacity <= 100:
            problems.append("watermark opacity must be between 10 and 100")
        for name in ("prompt", "caption_language", "logo", "watermark", "language"):
            value = getattr(self, name)
            setattr(self, name, value.strip() if isinstance(value, str) and value.strip() else None)
        if self.logo and not Path(self.logo).expanduser().is_file():
            problems.append(f"no logo file at {self.logo}")
        if self.watermark and len(self.watermark) > 80:
            problems.append("keep the watermark text to 80 characters")
        if problems:
            raise ClipperError("; ".join(problems))
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "Options":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names}).validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def forced(self, step: str) -> bool:
        return self.force is not None and STEPS.index(step) >= STEPS.index(self.force)

    @property
    def style(self):
        return make_style(self.caption_style, font=self.font, size=self.caption_size,
                          highlight=self.highlight_color, uppercase=self.uppercase,
                          position=self.caption_position)


def watermark_for(opts: Options, fmt: Format, logo_shape: tuple[int, ...] | None = None) -> Watermark | None:
    """The watermark text for one format; in the logo's corner it sits next to the logo, towards the middle."""
    if not opts.watermark:
        return None
    shift = 0
    if logo_shape is not None and opts.watermark_position == opts.logo_position:
        shift = logo_size(logo_shape, fmt)[1] + fmt.v(16)
    return Watermark(opts.watermark, opts.watermark_position, opts.watermark_opacity, shift)


def claude_factory(opts: Options, work: Path) -> Callable[[], Claude]:
    return lambda: Claude(opts.model, cwd=work)


def _publish(src: Path, dest: Path) -> None:
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def clip_dir_for(work: Path, clip: dict) -> Path:
    """Results for a stretch of the video are kept by its time range. A clip with earlier parts gets a
    folder of its own for what depends on all of them (captions, the render); each part's section,
    words and framing stay in the part's own folder."""
    name = f"{round(clip['start'] * 1000):08d}-{round(clip['end'] * 1000):08d}"
    if clip.get("before"):
        name += "+" + hashlib.sha1(json.dumps(clip_spans(clip)).encode()).hexdigest()[:8]
    return work / "clips" / name


def parts_of(clip: dict) -> list[dict]:
    """The clip's parts in playing order, each a clip record of its own (same rank, title, ...) for one
    stretch of the video."""
    single = {k: v for k, v in clip.items() if k != "before"}
    return [{**single, **part} for part in clip_parts(clip)]


def zoom_windows(words: list[dict], keywords: list[str], duration: float,
                 extra: list[tuple[float, float]] = (), auto: bool = True) -> list[tuple[float, float]]:
    """When to punch in: on the opening line (the hook) and on each of the keywords, plus any windows you
    added yourself (output time)."""
    windows = [(0.0, min(1.4, duration))] if auto else []
    if auto:
        for i in sorted(keyword_positions(words, keywords)):
            windows.append((max(0.0, words[i]["s"] - 0.05), min(duration, words[i]["e"] + 1.0)))
    windows += [(max(0.0, a), min(duration, b)) for a, b in extra if min(duration, b) - max(0.0, a) > 0.1]
    merged: list[tuple[float, float]] = []
    for a, b in sorted(windows):
        if merged and a - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


# ---------------------------------------------------------------------------------------------
# Caption corrections from the web UI

def _edits_path(work: Path, clip: dict) -> Path:
    return clip_dir_for(work, clip) / "words_edited.json"


def _edits_span(clip: dict) -> list:
    spans = [list(x) for x in clip_spans(clip)]
    return spans[0] if len(spans) == 1 else spans


def edited_words(work: Path, clip: dict) -> list[dict] | None:
    path = _edits_path(work, clip)
    if path.exists():
        data = read_json(path)
        if data.get("span") == _edits_span(clip):
            return data["words"]
    return None


def part_words(work: Path, part: dict) -> list[dict]:
    """The transcribed words of one part of a clip: the clip's own transcription where there is one, else the
    video's transcript."""
    path = clip_dir_for(work, part) / "words.json"
    if path.exists():
        return read_json(path)["result"]
    transcript = read_json(work / "transcript.json")["result"]
    return words_in(transcript["words"], part["start"], part["end"])


def caption_lines(work: Path, clip: dict) -> list[dict]:
    """The clip's words as editable sentences (source times), in playing order."""
    words = edited_words(work, clip)
    if words is None:
        words = [w for part in parts_of(clip) for w in part_words(work, part)]
    words = [{**w, "w": w["w"].strip()} for w in words if w["w"].strip()]
    at = lambda t: round(play_time(clip, t), 2)  # noqa: E731
    return [{"start": p[0]["s"], "end": p[-1]["e"], "at": at(p[0]["s"]), "until": at(p[-1]["e"]),
             "text": " ".join(w["w"] for w in p)} for p in _phrases(words)]


def play_time(clip: dict, t: float) -> float:
    """Where the moment t (source time) plays in the finished clip, after its parts are joined and the jump
    cuts made; a moment that was cut out gives where the clip goes on. Clips rendered before the cuts were
    kept are taken as their parts played uncut, which is close enough to follow along."""
    timeline = clip.get("timeline")
    if not timeline:
        timeline, out = [], 0.0
        for part in parts_of(clip):
            timeline.append([part["start"], part["end"], out])
            out += part["end"] - part["start"]
    for a, b, out in timeline:
        if a <= t < b:
            return out + t - a
    later = [out for a, b, out in timeline if a > t]
    if later:
        return min(later)
    a, b, out = timeline[-1]
    return out + b - a


def save_caption_lines(work: Path, clip: dict, lines: list[dict]) -> None:
    """Store corrected sentences; each one's time is spread over its (possibly changed) words."""
    words = []
    for line in lines:
        words += spread(float(line["start"]), float(line["end"]), str(line["text"]))
    if not words:
        raise ClipperError("The captions can't be empty.")
    # A part the lines don't cover (a section just added in the editor) keeps its transcribed words.
    for part in parts_of(clip):
        if not any(part["start"] <= w["s"] < part["end"] for w in words):
            words += [{"w": w["w"].strip(), "s": w["s"], "e": w["e"]} for w in part_words(work, part) if w["w"].strip()]
    words.sort(key=lambda w: w["s"])
    write_json(_edits_path(work, clip), {"span": _edits_span(clip), "words": words})


def apply_title(clip: dict, ed: ClipEdits) -> None:
    """Your own title replaces Claude's on the clip; Claude's is kept so clearing yours brings it back."""
    if ed.title:
        clip.setdefault("claude_title", clip["title"])
        clip["title"] = ed.title
    elif clip.get("claude_title"):
        clip["title"] = clip.pop("claude_title")


# ---------------------------------------------------------------------------------------------

def _prefetch(work: Path, meta: dict, clip: dict) -> None:
    """Fetch a clip part's section ahead of time; a failure is left for the clip's own turn to retry and report."""
    try:
        clip_video(work, meta, clip, clip_dir_for(work, clip))
    except Cancelled:
        raise
    except ClipperError:
        pass


def produce(work: Path, meta: dict, transcript: dict, clips: list[dict], opts: Options) -> list[dict]:
    """Steps 4-6 (reframe, captions, render) for clips that have already been chosen."""
    do_render = opts.stop_after not in ("reframe", "captions")
    if do_render:
        check_ffmpeg_has_libass()
    style = opts.style
    formats = [get_format(n) for n in opts.formats]
    logo_shape = read_logo(Path(opts.logo).expanduser()).shape if opts.logo and opts.watermark else None
    fps = float(Fraction(meta["fps"]))
    # Your own changes from the editor, per clip: they ride on top of what Claude and clipper proposed.
    edits = {c["id"]: load_edits(work, c) for c in clips}
    loud = Loudness.from_work(work) if any((edits[c["id"]].jump_cuts or opts.jump_cuts) != "off"
                                           for c in clips) else None
    claude = claude_factory(opts, work)
    out_dir = work / "output"
    out_dir.mkdir(exist_ok=True)
    # New files are made in a staging folder and only replace the old ones once every clip is done, so a
    # cancelled or failed render leaves the clips that were already there (and their list) untouched.
    stage = out_dir / ".next"
    if do_render:
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()

    # Sections of the video are fetched a few at a time in the background while earlier clips are made.
    pool = thread_pool(SECTIONS_AT_ONCE) if meta.get("video") == "sections" else None
    fetches = {(c["rank"], n): pool.submit(_prefetch, work, meta, part)
               for c in clips for n, part in enumerate(parts_of(c))} if pool else {}
    try:
        for clip in clips:
            check_cancel()
            report("step", f"clip {clip['rank']} of {len(clips)}")
            ed = edits[clip["id"]]
            apply_title(clip, ed)
            jump_cuts, layout = ed.jump_cuts or opts.jump_cuts, ed.layout or opts.layout
            keywords = list(ed.keywords) if ed.keywords is not None else list(clip.get("keywords") or [])
            cdir = clip_dir_for(work, clip)
            cdir.mkdir(parents=True, exist_ok=True)
            parts = parts_of(clip)
            fixed = edited_words(work, clip)
            if fixed is not None:
                log(f"captions #{clip['rank']}: using your corrections")
            # Each part (most clips have one) is cut, transcribed and framed on its own; they're played in a row.
            pieces, spoken, offset, timeline = [], [], 0.0, []
            for n, part in enumerate(parts):
                pdir = clip_dir_for(work, part)
                pdir.mkdir(parents=True, exist_ok=True)
                words = clip_words(work, transcript, part, pdir, opts.caption_model, opts.whisper_backend,
                                   force=opts.forced("captions"))
                if fixed is not None:
                    words = fixed if len(parts) == 1 else words_in(fixed, part["start"], part["end"])
                if (clip["rank"], n) in fetches:
                    fetches[clip["rank"], n].result()  # this part's section, fetched in the background
                src = clip_video(work, meta, part, pdir)
                keep = plan_cuts(words, part["start"], part["end"], fps, jump_cuts, loud.is_quiet if loud else None)
                if ed.cuts_removed or ed.cuts_kept:
                    keep = apply_cuts(keep, ed.in_part(ed.cuts_removed, part["start"], part["end"]),
                                      ed.in_part(ed.cuts_kept, part["start"], part["end"]),
                                      part["end"] - part["start"], fps)
                edit = EditMap(keep)
                pieces.append((part, pdir, words, src, edit))
                timeline += [[round(part["start"] + a, 3), round(part["start"] + b, 3), round(offset + o, 3)]
                             for (a, b), o in zip(edit.segments, edit.offsets)]
                spoken += [{**w, "s": w["s"] + offset, "e": w["e"] + offset}
                           for w in map_words(words, part["start"], edit)]
                offset += edit.duration
            duration = offset
            clip["final_duration"] = round(duration, 2)
            clip["timeline"] = timeline
            clip["edited"] = not ed.is_empty
            to_out = lambda t: play_time(clip, t)  # noqa: E731 - video time -> time in the finished clip
            cap_words, cap_title, cap_keywords = spoken, clip["title"], keywords
            if opts.caption_language:
                cap_words, cap_title = translate_clip(cdir, spoken, clip["title"], opts.caption_language, claude)
                cap_keywords = None  # the keywords were picked in the spoken language
            show_title = opts.title_overlay if ed.title_overlay is None else ed.title_overlay
            texts = [{**t, "start": to_out(t["start"]), "end": to_out(t["end"])} for t in ed.texts]
            look = Look(zoom=zoom_windows(spoken, keywords, duration, [(to_out(a), to_out(b)) for a, b in ed.zooms],
                                          auto=opts.zoom and ed.zoom != "off"),
                        logo=Path(opts.logo).expanduser() if opts.logo else None, logo_position=opts.logo_position,
                        opacity=opts.watermark_opacity, progress=style.highlight if opts.progress_bar else None,
                        broll=[(to_out(b["start"]), to_out(b["end"]), Path(b["file"]).expanduser()) for b in ed.broll
                               if to_out(b["end"]) - to_out(b["start"]) > 0.1],
                        fade_between=0.3 if ed.transition == "fade" and len(parts) > 1 else 0.0,
                        fade_in=0.4 if ed.fade_in else 0.0, fade_out=0.5 if ed.fade_out else 0.0,
                        music=(Path(ed.music["file"]).expanduser(), ed.music["volume"]) if ed.music else None)
            base = f"{clip['rank']:02d}_{slugify(clip['title'])}"
            outputs = {}
            for n, fmt in enumerate(formats):
                fdir = cdir / fmt.tag
                fdir.mkdir(exist_ok=True)
                plans = []
                for part, pdir, words, src, edit in pieces:
                    (pdir / fmt.tag).mkdir(exist_ok=True)
                    plan = reframe(work, meta, part, pdir / fmt.tag, words, layout,
                                   force=opts.forced("reframe"), src=src, fmt=fmt,
                                   layouts=ed.layouts_in_part(part["start"], part["end"]))
                    plans.append(Piece(src, part["start"], part["end"], plan, edit))
                ass = make_captions(cap_words, duration, fdir, style, fmt,
                                    title=cap_title if show_title else None, cover_title=cap_title,
                                    keywords=cap_keywords, watermark=watermark_for(opts, fmt, logo_shape), texts=texts)
                if not do_render:
                    continue
                mp4 = render(work, meta, clip, plans, ass, fdir, fmt=fmt, look=look, encoder=opts.encoder,
                             crf=opts.crf, preset=opts.preset, force=opts.forced("render"))
                name = base if n == 0 else f"{base}_{fmt.tag}"
                _publish(mp4, stage / f"{name}.mp4")
                _publish(fdir / "cover.jpg", stage / f"{name}.jpg")
                outputs[fmt.name] = {"file": f"{name}.mp4", "cover": f"{name}.jpg"}
            if do_render:
                clip["file"], clip["cover"] = outputs[formats[0].name]["file"], outputs[formats[0].name]["cover"]
                clip["formats"] = outputs
                (stage / f"{base}.srt").write_text(build_srt(spoken), encoding="utf-8")
                clip["srt"] = [f"{base}.srt"]
                if opts.caption_language:
                    tag = slugify(opts.caption_language, 12)
                    (stage / f"{base}.{tag}.srt").write_text(build_srt(cap_words), encoding="utf-8")
                    clip["srt"].append(f"{base}.{tag}.srt")
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)

    if do_render and opts.posts:
        try:
            posts = write_posts(work, meta, clips, claude, force=opts.forced("render"))
        except ClipperError as e:
            log(f"posts: skipped ({e})")
            posts = {}
        for clip in clips:
            if clip["id"] in posts:
                clip["post"] = posts[clip["id"]]
                text_file = stage / f"{clip['rank']:02d}_{slugify(clip['title'])}.txt"
                text_file.write_text(format_post(clip, posts[clip["id"]]), encoding="utf-8")
    if do_render:
        for old in out_dir.iterdir():
            if old.suffix in (".mp4", ".jpg", ".txt", ".srt"):
                old.unlink()
        for new in stage.iterdir():
            os.replace(new, out_dir / new.name)
        stage.rmdir()
    write_json(out_dir / "clips.json", clips)
    return clips


def process(source: str, opts: Options, new_clips: bool = False) -> tuple[Path, list[dict]]:
    """Run one video through the pipeline (or up to opts.stop_after). new_clips: once transcribed, ask Claude
    for clips at moments it hasn't suggested before, and make them alongside the clips already made."""
    for tool in ("ffmpeg", "ffprobe"):
        require_tool(tool)
    work_root = Path(opts.work_dir)
    report("step", "ingest")
    work, meta = ingest(source, work_root, force=opts.forced("ingest"), download_mode=opts.download)
    report("video", work.name)
    if opts.stop_after == "ingest":
        return work, []
    check_cancel()
    report("step", "transcribe")
    transcript = transcribe(work, meta, opts.whisper_model, opts.language, opts.whisper_backend,
                            force=opts.forced("transcribe"))
    if opts.stop_after == "transcribe":
        return work, []
    check_cancel()
    if new_clips:
        return work, find_new_clips(work, opts, render=opts.stop_after != "select")
    report("step", "select")
    clips = select_clips(work, meta, transcript, num_clips=opts.num_clips, min_len=opts.min_len,
                         max_len=opts.max_len, model=opts.model, claude_factory=claude_factory(opts, work),
                         force=opts.forced("select"), focus=opts.prompt,
                         time_range=(opts.range_start, opts.range_end), earlier=opts.earlier_parts)
    if not clips:
        raise ClipperError("Claude didn't find any usable clips in this video.")
    check_cancel()
    if opts.stop_after == "select":
        return work, clips
    return work, produce(work, meta, transcript, clips, opts)


def load_video(work: Path) -> dict:
    """Everything the UI shows about one processed video."""
    out = {"id": work.name, "meta": read_json(work / "meta.json") if (work / "meta.json").exists() else None}
    for name in ("clips", "review"):
        path = work / f"{name}.json"
        out[name] = read_json(path) if path.exists() else None
    out["transcribed"] = (work / "transcript.json").exists()
    out["candidates"] = saved_candidates(work)
    rendered = work / "output" / "clips.json"
    out["rendered"] = read_json(rendered) if rendered.exists() else []
    return out


def produce_reviewed(work: Path, items: list[dict], opts: Options) -> list[dict]:
    """Render clips chosen (and possibly adjusted) by hand in the review screen."""
    meta = read_json(work / "meta.json")
    transcript = read_json(work / "transcript.json")["result"]
    candidates = saved_candidates(work)
    rendered = work / "output" / "clips.json"
    if rendered.exists():  # clips made earlier keep their keywords even if their suggestion is gone
        known = {c["id"] for c in candidates}
        candidates += [c for c in read_json(rendered) if c.get("id") not in known]
    clips = clips_from_review(transcript, meta, candidates, items)
    if not clips:
        raise ClipperError("No clips were selected.")
    sig = ",".join(f"{c['id']}:{c['si']}-{c['ei']}" for c in candidates)  # same format as the UI's
    write_json(work / "review.json", {"items": items, "candidates": sig, "options": opts.to_dict()})
    return produce(work, meta, transcript, clips, opts)


def find_new_clips(work: Path, opts: Options, render: bool = False) -> list[dict]:
    """Ask Claude for clips at moments it hasn't suggested yet (for a video that's already transcribed).
    The earlier suggestions stay. With render, the new clips are made and added to the ones already made."""
    meta = read_json(work / "meta.json")
    if not (work / "transcript.json").exists():
        raise ClipperError("This video hasn't been transcribed yet.")
    transcript = read_json(work / "transcript.json")["result"]
    path = work / "output" / "clips.json"
    made = [c for c in read_json(path) if c.get("file")] if path.exists() else []
    report("step", "select")
    new = select_clips(work, meta, transcript, num_clips=opts.num_clips, min_len=opts.min_len,
                       max_len=opts.max_len, model=opts.model, claude_factory=claude_factory(opts, work),
                       focus=opts.prompt, time_range=(opts.range_start, opts.range_end), new_search=True,
                       also_avoid=[span for c in made for span in clip_spans(c)], earlier=opts.earlier_parts)
    if not new:
        raise ClipperError("Claude didn't find any new clips. Try other lengths, another part of the video, "
                           "or tell it what to look for.")
    if not render:
        return new
    items = [{"id": c.get("id"), "si": c["si"], "ei": c["ei"], "title": c["title"],
              "before": [[p["si"], p["ei"]] for p in c.get("before", [])]} for c in made + new]
    return produce_reviewed(work, items, opts)


def rerender(work: Path, opts: Options) -> list[dict]:
    """Render the current clips again (after caption corrections, or with other options)."""
    meta = read_json(work / "meta.json")
    transcript = read_json(work / "transcript.json")["result"]
    path = work / "output" / "clips.json"
    if not path.exists():
        raise ClipperError("There are no rendered clips to update.")
    keep = ("rank", "id", "title", "claude_title", "hook", "type", "score", "scores", "start", "end", "duration",
            "si", "ei", "text", "keywords", "before")
    clips = [{k: c[k] for k in keep if k in c} for c in read_json(path)]
    return produce(work, meta, transcript, clips, opts)
