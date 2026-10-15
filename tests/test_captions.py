import re

from clipper.captions import MAX_CHARS, MAX_WORDS, build_ass, clip_words, group_lines
from conftest import make_words


def test_lines_are_short_and_never_cross_sentences():
    words = clip_words(make_words([
        "So here is the thing nobody ever tells you about starting out.",
        "Yes.",
        "Honestly, it changed everything for me and my family.",
    ]), 0, 100)
    lines = group_lines(words)
    assert all(1 <= len(l) <= MAX_WORDS for l in lines)
    assert sum(len(l) for l in lines) == len(words)
    texts = [" ".join(w["w"] for w in l) for l in lines]
    assert "Yes." in texts  # a one-word sentence stays on its own
    assert all(len(t) <= MAX_CHARS for t in texts if " " in t)  # fits across the frame
    assert sum(len(l) == 1 for l in lines) <= 2
    assert not any("Honestly, it changed" in t for t in texts)


def test_clip_words_are_relative_and_inside_the_clip():
    words = make_words(["one two three four five six"])
    cw = clip_words(words, words[1]["s"], words[4]["e"])
    assert [w["w"] for w in cw] == ["two", "three", "four", "five"]
    assert cw[0]["s"] == 0.0


def test_ass_highlights_each_word_in_turn():
    words = clip_words(make_words(["We tried it. And it worked!"]), 0, 100)
    ass = build_ass(words, 10, "Arial Black")
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == len(words)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    for ev, w in zip(events, words):
        text = ev.split(",,", 1)[1].split(",", 4)[-1]
        assert text.count(r"\c&H00FFFF&") == 1  # exactly one yellow word
        highlighted = re.search(r"\\c&H00FFFF&[^}]*\}([^{]+)\{", text)[1]
        assert highlighted == w["w"].rstrip(".,")
        assert r"\t(0,120" in text  # the pop
    # within a line the events hand over without gaps
    t = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))  # noqa: E731
    spans = [(t(e.split(",")[1]), t(e.split(",")[2])) for e in events]
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(spans, spans[1:]))
    assert spans[0][1] == spans[1][0]
