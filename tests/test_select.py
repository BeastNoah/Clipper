import json
import re

import pytest

from clipper.llm import Claude, InvalidReply
from clipper.select import (
    CHUNK_SECONDS, Timeline, clips_from_review, make_chunks, parse_proposals, parse_rescores,
    merged_length, resolve_overlaps, saved_candidates, select_clips, snap, snap_and_fit, split_sentences,
)
from clipper.util import ClipperError
from conftest import make_words


def test_split_sentences_on_punctuation_and_pauses():
    words = make_words(["Hello there.", "Mr. Smith went home today!", "Is that so?"])
    # a long pause without punctuation also ends a sentence
    words += [{"w": " after", "s": words[-1]["e"] + 3, "e": words[-1]["e"] + 3.3, "p": 1},
              {"w": " pause", "s": words[-1]["e"] + 3.4, "e": words[-1]["e"] + 3.7, "p": 1}]
    sents = split_sentences(words)
    assert [s.text for s in sents] == ["Hello there.", "Mr. Smith went home today!", "Is that so?", "after pause"]
    assert sents[1].start == words[2]["s"] and sents[1].end == words[6]["e"]


def test_split_sentences_breaks_run_ons_at_commas():
    words = make_words(["one two three four five six seven eight, nine ten eleven twelve thirteen fourteen "
                        "fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree "
                        "twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty thirtyone "
                        "thirtytwo thirtythree thirtyfour thirtyfive thirtysix thirtyseven thirtyeight"])
    sents = split_sentences(words)
    assert len(sents) > 1
    assert all(s.end - s.start <= 15 for s in sents)
    assert sum(s.w1 - s.w0 + 1 for s in sents) == len(words)


def test_chunks_overlap_enough_for_any_clip():
    words = make_words([f"Sentence number {i} is here." for i in range(800)])  # ~40 minutes
    sents = split_sentences(words)
    overlap = 90
    chunks = make_chunks(sents, overlap=overlap)
    assert len(chunks) == 3
    spans = [(sents[a].start, sents[b].end) for a, b in chunks]
    assert spans[0][0] == sents[0].start and spans[-1][1] == sents[-1].end
    assert all(hi - lo <= CHUNK_SECONDS + 10 for lo, hi in spans)
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert hi - lo >= overlap - 5


def _timeline(n=30):
    words = make_words([f"This is sentence {i} of the test." for i in range(n)])
    sents = split_sentences(words)
    return Timeline(sents, words, words[-1]["e"] + 1)


def test_snap_moves_to_sentence_boundaries():
    tl = _timeline()
    s = tl.sentences
    si, ei = snap_and_fit(tl, s[3].start + 0.4, s[8].end - 0.3, 5, 60)
    assert (si, ei) == (3, 8)
    assert tl.clip_start(si) < s[3].start and tl.clip_end(ei) > s[8].end  # padded, but never into neighbours
    assert tl.clip_start(si) > tl.words[s[3].w0 - 1]["e"] and tl.clip_end(ei) < tl.words[s[8].w1 + 1]["s"]


def test_fit_trims_end_and_extends_short_clips():
    tl = _timeline()
    s = tl.sentences
    si, ei = snap_and_fit(tl, s[2].start, s[20].end, 5, 20)  # far too long: keep the start, trim the end
    assert si == 2 and tl.dur(si, ei) <= 20 and tl.dur(si, ei + 1) > 20
    si, ei = snap_and_fit(tl, s[5].start, s[5].end, 12, 20)  # too short: grow forwards
    assert si == 5 and ei > 5 and 12 <= tl.dur(si, ei) <= 20
    si, ei = snap_and_fit(tl, s[29].start, s[29].end, 12, 20)  # at the very end: grow backwards
    assert ei == 29 and si < 29 and tl.dur(si, ei) >= 12
    assert snap_and_fit(tl, s[0].start, s[29].end, 2000, 3000) is None


def test_resolve_overlaps_keeps_higher_score():
    cands = [{"start": 0, "end": 30, "score": 60}, {"start": 20, "end": 50, "score": 80},
             {"start": 50, "end": 70, "score": 40}]
    kept = resolve_overlaps(cands)
    assert [c["score"] for c in kept] == [80, 40]


def test_parse_proposals_reports_problems():
    good = {"clips": [{"start": 10, "end": 40, "title": "T", "hook": "H",
                       "scores": {"hook": 20, "standalone": 20, "emotion": 10, "clarity": 15}}]}
    assert parse_proposals("Here you go:\n```json\n" + json.dumps(good) + "\n```", 0, 100)[0]["start"] == 10
    with pytest.raises(InvalidReply, match="JSON"):
        parse_proposals("no json here", 0, 100)
    bad = json.loads(json.dumps(good))
    bad["clips"][0]["scores"]["hook"] = 30
    with pytest.raises(InvalidReply, match="less than or equal to 25"):
        parse_proposals(json.dumps(bad), 0, 100)
    with pytest.raises(InvalidReply, match="outside"):
        parse_proposals(json.dumps(good), 50, 100)


def test_parse_rescores_requires_every_id_once():
    item = lambda i: {"id": i, "hook": 1, "standalone": 2, "emotion": 3, "clarity": 4}  # noqa: E731
    assert parse_rescores(json.dumps({"scores": [item("c1"), item("c2")]}), ["c1", "c2"])["c2"]["clarity"] == 4
    with pytest.raises(InvalidReply, match="missing scores for: c2"):
        parse_rescores(json.dumps({"scores": [item("c1")]}), ["c1", "c2"])
    with pytest.raises(InvalidReply, match="more than once"):
        parse_rescores(json.dumps({"scores": [item("c1"), item("c1"), item("c2")]}), ["c1", "c2"])


def _calls(log):
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_long_video_is_chunked_rescored_and_cached(tmp_path, fake_claude):
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(700)])  # ~38 min
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    kw = dict(min_len=20, max_len=45, model="test", claude_factory=lambda: Claude("test"))
    clips = select_clips(tmp_path, meta, {"words": words}, num_clips=4, **kw)
    calls = _calls(fake_claude)
    proposals = [c for c in calls if "<transcript>" in c["prompt"]]
    rescores = [c for c in calls if "<clips>" in c["prompt"]]
    assert len(proposals) == 3 and len(rescores) == 1
    assert len(clips) == 4
    assert [c["rank"] for c in clips] == [1, 2, 3, 4]
    assert all(20 <= c["duration"] <= 45 for c in clips)
    assert all(a["score"] >= b["score"] for a, b in zip(clips, clips[1:]))
    for a in clips:
        for b in clips:
            assert a is b or a["end"] <= b["start"] or b["end"] <= a["start"]

    # Changing only the number of clips reuses everything Claude already said.
    n = len(calls)
    assert len(select_clips(tmp_path, meta, {"words": words}, num_clips=2, **kw)) == 2
    assert len(_calls(fake_claude)) == n


def test_parts_of_a_long_video_are_asked_about_at_the_same_time(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "1")
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(700)])  # 3 chunks
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    select_clips(tmp_path, meta, {"words": words}, num_clips=3, min_len=20, max_len=45, model="test",
                 claude_factory=lambda: Claude("test"))
    starts = [c["t"] for c in _calls(fake_claude) if "<transcript>" in c["prompt"]]
    assert len(starts) == 3
    assert max(starts) - min(starts) < 0.9  # one by one, each would start after the one before had answered


def test_clips_stay_inside_the_chosen_time_range():
    words = make_words([f"This is sentence {i} of the test." for i in range(60)])
    sents = split_sentences(words)
    lo, hi = 20, 29  # the sentences of the chosen range
    tl = Timeline(sents, words, words[-1]["e"] + 1, lo, hi)
    s = tl.sentences
    # a short clip at the end of the range has to grow, but only backwards (never past the range end)
    si, ei = snap_and_fit(tl, s[28].start, s[29].end, 8, 20)
    assert lo <= si <= ei == hi and tl.dur(si, ei) >= 8
    # times a little outside the range snap to its first and last sentence
    assert snap(tl, s[17].start, s[33].end) == (lo, hi)
    # at the very start of the range it grows forwards only
    si, ei = snap_and_fit(tl, s[20].start, s[20].end, 8, 20)
    assert si == lo and ei <= hi


def test_new_search_keeps_and_steers_clear_of_earlier_suggestions(tmp_path, fake_claude):
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(120)])  # ~7 min
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    kw = dict(min_len=20, max_len=40, model="test", claude_factory=lambda: Claude("test"))
    select_clips(tmp_path, meta, {"words": words}, num_clips=2, **kw)
    earlier = json.loads((tmp_path / "select.json").read_text())["candidates"]
    made = [(300.0, 330.0)]  # a clip made by hand
    new = select_clips(tmp_path, meta, {"words": words}, num_clips=2, new_search=True, also_avoid=made, **kw)
    state = json.loads((tmp_path / "select.json").read_text())
    assert state["search"] == 2 and state["earlier"] == earlier and len(new) == 2
    ids = [c["id"] for c in saved_candidates(tmp_path)]
    assert len(ids) == len(set(ids)) == len(earlier) + len(state["candidates"])
    taken = [(c["start"], c["end"]) for c in earlier] + made
    for c in new:
        assert c["id"] not in {e["id"] for e in earlier}
        assert merged_length(taken, c["start"], c["end"]) <= 0.25 * c["duration"]
    prompt = [c for c in _calls(fake_claude) if "<transcript>" in c["prompt"]][-1]["prompt"]
    assert "already taken" in prompt and "- 300.0-330.0" in prompt

    # the same options again reuse the newest search without asking Claude
    n = len(_calls(fake_claude))
    again = select_clips(tmp_path, meta, {"words": words}, num_clips=2, **kw)
    assert [c["id"] for c in again] == [c["id"] for c in new] and len(_calls(fake_claude)) == n
    # other options start another search, and everything suggested before stays listed
    select_clips(tmp_path, meta, {"words": words}, num_clips=2, **{**kw, "max_len": 35})
    assert len(saved_candidates(tmp_path)) > len(ids)
    # --force select starts over
    select_clips(tmp_path, meta, {"words": words}, num_clips=2, force=True, **kw)
    assert {c.get("search") for c in saved_candidates(tmp_path)} == {1}


def test_merged_length_counts_overlaps_once():
    assert merged_length([(0, 10), (5, 15), (20, 30)], 0, 25) == 20
    assert merged_length([(0, 10)], 20, 30) == 0


def test_asks_for_more_when_too_few_survive(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MAX", "1")
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(120)])  # ~6.5 min
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    clips = select_clips(tmp_path, meta, {"words": words}, num_clips=3, min_len=20, max_len=40,
                         model="test", claude_factory=lambda: Claude("test"))
    calls = [c for c in _calls(fake_claude) if "<transcript>" in c["prompt"]]
    assert len(calls) == 3 and "already taken" in calls[1]["prompt"]
    assert len(clips) == 3
    # the extra rounds came from separate calls, so everything was re-scored together
    assert any("<clips>" in c["prompt"] for c in _calls(fake_claude))


def test_bad_answer_is_sent_back_for_a_retry(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_BAD_FIRST", "1")
    claude = Claude("test")

    def parse(reply):
        data = json.loads(reply) if reply.startswith("{") else None
        if data is None:
            raise InvalidReply("not JSON")
        return data

    assert claude.ask("system", "prompt", parse) == {"clips": []}
    first, second = _calls(fake_claude)
    assert "--resume" not in first["argv"]
    session = first["argv"][first["argv"].index("--session-id") + 1]
    assert second["argv"][second["argv"].index("--resume") + 1] == session
    assert "not JSON" in second["prompt"]
    assert first["argv"][first["argv"].index("--tools") + 1] == ""


def test_claude_chooses_where_to_shorten_long_clips(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_LONG", "1")
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(120)])
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    clips = select_clips(tmp_path, meta, {"words": words}, num_clips=2, min_len=15, max_len=30,
                         model="test", claude_factory=lambda: Claude("test"))
    trims = [c for c in _calls(fake_claude) if '<clip id="' in c["prompt"]]
    assert len(trims) == 1
    state = json.loads((tmp_path / "select.json").read_text())
    assert all(c["trimmed_by_claude"] for c in state["candidates"])
    # the fake editor dropped each clip's first line, so the clip no longer starts where proposed
    assert all(c["start"] > c["proposed"][0] for c in state["candidates"])
    assert all(15 <= c["duration"] <= 30 for c in clips)


def test_duplicates_and_keywords(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_DUP", "1")
    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(700)])
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    select_clips(tmp_path, meta, {"words": words}, num_clips=30, min_len=20, max_len=45,
                 model="test", claude_factory=lambda: Claude("test"))
    state = json.loads((tmp_path / "select.json").read_text())
    c1, c2 = state["candidates"][:2]
    assert c2["duplicate_of"] == c1["id"]
    clips = json.loads((tmp_path / "clips.json").read_text())
    assert c2["id"] not in [c["id"] for c in clips]  # a duplicate never makes it into the final set
    assert all(c["keywords"] and c["keywords"][0].lower() in c["text"].lower() for c in clips)
    assert all({"si", "ei"} <= set(c) for c in clips)


def test_audio_cues_reach_the_prompt(tmp_path, fake_claude):
    import numpy as np
    import wave

    words = make_words([f"Sentence {i} is spoken here." for i in range(20)])
    words = [dict(w, s=w["s"] + (3 if i >= 25 else 0), e=w["e"] + (3 if i >= 25 else 0))
             for i, w in enumerate(words)]  # a 3s gap after sentence 5 ...
    rate = 16000
    audio = np.zeros(int((words[-1]["e"] + 1) * rate), np.int16)
    for w in words:
        audio[int(w["s"] * rate):int(w["e"] * rate)] = 3000
    gap0 = words[24]["e"]
    audio[int((gap0 + 0.3) * rate):int((gap0 + 2.6) * rate)] = 2500  # ... filled with laughter-like sound
    with wave.open(str(tmp_path / "audio.wav"), "wb") as wf:
        wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    select_clips(tmp_path, meta, {"words": words}, num_clips=1, min_len=5, max_len=20,
                 model="test", claude_factory=lambda: Claude("test"))
    prompt = next(c["prompt"] for c in _calls(fake_claude) if "<transcript>" in c["prompt"])
    assert re.search(r"Sentence 4 is spoken here\. \[\+3\.\ds non-speech sound\]", prompt)
    assert "non-speech sound] means" in prompt


def test_review_items_become_clips():
    words = make_words([f"This is sentence {i} of the test." for i in range(30)])
    meta = {"duration": words[-1]["e"] + 1}
    cand = {"id": "c7", "si": 3, "ei": 6, "title": "Claude's title", "hook": "This is sentence 3 of the test.",
            "score": 70, "scores": {}, "keywords": ["sentence 5", "missing"]}
    clips = clips_from_review({"words": words}, meta, [cand], [
        {"id": "c7", "si": 4, "ei": 6}, {"si": 10, "ei": 12, "title": "Mine"}])
    assert [c["rank"] for c in clips] == [1, 2]
    assert clips[0]["title"] == "Claude's title" and clips[0]["keywords"] == ["sentence 5"]
    assert clips[0]["hook"] == "This is sentence 4 of the test."  # the start moved, so the hook did too
    assert clips[1]["title"] == "Mine" and clips[1]["id"] == "u10-12"
    with pytest.raises(ClipperError):
        clips_from_review({"words": words}, meta, [], [{"si": 5, "ei": 99}])


def test_video_ids_are_ascii_and_bad_options_fail_early(tmp_path):
    from clipper.ingest import local_video_id
    from clipper.pipeline import Options

    f = tmp_path / "Gespräch über Ärger – 第一.mp4"
    f.write_bytes(b"x" * 5000)
    vid = local_video_id(f)
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]+", vid) and vid.startswith("gesprach-uber-arger")
    with pytest.raises(ClipperError, match="#FFE600"):
        Options(highlight_color="red").validate()
    with pytest.raises(ClipperError, match="layout"):
        Options(layout="sideways").validate()
