import re

from clipper.captions import (
    MAX_WORDS, PRESETS, build_ass, build_cover_ass, group_lines, keyword_positions, make_style, map_words,
    tag_color,
)
from clipper.edit import EditMap
from clipper.transcribe import words_in
from conftest import make_words

MAX_CHARS = PRESETS["bold"].max_chars


def _rel(words, start=0.0, end=100.0):
    return map_words(words_in(words, start, end), start, EditMap([(0.0, end - start)]))


def test_lines_are_short_and_never_cross_sentences():
    words = _rel(make_words([
        "So here is the thing nobody ever tells you about starting out.",
        "Yes.",
        "Honestly, it changed everything for me and my family.",
    ]))
    lines = group_lines(words, MAX_CHARS)
    assert all(1 <= len(l) <= MAX_WORDS for l in lines)
    assert sum(len(l) for l in lines) == len(words)
    texts = [" ".join(w["w"] for w in l) for l in lines]
    assert "Yes." in texts  # a one-word sentence stays on its own
    assert all(len(t) <= MAX_CHARS for t in texts if " " in t)  # fits across the frame
    assert sum(len(l) == 1 for l in lines) <= 2
    assert not any("Honestly, it changed" in t for t in texts)


def test_words_are_relative_and_inside_the_clip():
    words = make_words(["one two three four five six"])
    cw = _rel(words, words[1]["s"], words[4]["e"])
    assert [w["w"] for w in cw] == ["two", "three", "four", "five"]
    assert cw[0]["s"] == 0.0


def test_words_follow_jump_cuts():
    words = make_words(["one two"]) + [{"w": " three", "s": 5.0, "e": 5.3}]
    edit = EditMap([(0.0, 1.0), (4.9, 6.0)])  # the pause between "two" and "three" was cut
    mapped = map_words(words, 0.0, edit)
    assert [w["w"] for w in mapped] == ["one", "two", "three"]
    assert abs(mapped[2]["s"] - 1.1) < 1e-9 and abs(edit.duration - 2.1) < 1e-9


def _events(ass):
    return [l for l in ass.splitlines() if l.startswith("Dialogue: 0")]


def test_ass_highlights_each_word_in_turn():
    words = _rel(make_words(["We tried it. And it worked!"]))
    ass = build_ass(words, 10)
    events = _events(ass)
    yellow = tag_color(PRESETS["bold"].highlight)
    assert len(events) == len(words)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    for ev, w in zip(events, words):
        text = ev.split(",,", 1)[1].split(",", 4)[-1]
        assert text.count(yellow) == 1  # exactly one highlighted word
        highlighted = re.search(re.escape(yellow) + r"[^}]*\}([^{]+)\{", text)[1]
        assert highlighted == w["w"].rstrip(".,")
        assert r"\t(0,120" in text  # the pop
    t = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))  # noqa: E731
    spans = [(t(e.split(",")[1]), t(e.split(",")[2])) for e in events]
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(spans, spans[1:]))
    assert spans[0][1] == spans[1][0]


def test_keywords_title_and_styles():
    words = _rel(make_words(["This one habit changed my whole life."]))
    assert keyword_positions(words, ["whole life", "HABIT"]) == {2, 5, 6}
    style = make_style("boxed", uppercase=True, position="middle", highlight="#FF0000")
    ass = build_ass(words, 5, style, keywords=["habit"], title="My title")
    green = tag_color(style.emphasis)
    assert "HABIT" in ass and "habit" not in ass
    assert all(green in e for e in _events(ass) if "HABIT" in e)  # coloured while its line is up
    assert any(tag_color("#FF0000") in e for e in _events(ass))
    assert "Dialogue: 1,0:00:00.00,0:00:02.60,Title,,0,0,0,,{\\fad(150,300)}My title" in ass
    style_line = next(l for l in ass.splitlines() if l.startswith("Style: Default"))
    assert style_line.split(",")[15] == "3" and style_line.split(",")[18] == "5"  # boxed, middle
    assert "Cover" in build_cover_ass("A {weird} title")
