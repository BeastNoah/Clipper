import json
import re

import numpy as np
import pytest

from clipper.captions import build_ass, build_srt, map_words
from clipper.edit import EditMap
from clipper.formats import get_format, parse_formats, parse_time
from clipper.llm import Claude
from clipper.pipeline import Options, zoom_windows
from clipper.render import Look, Overlay, crop_frame, fit_frame, split_frame, zoom_at
from clipper.translate import spread, translate_clip
from clipper.util import ClipperError
from conftest import make_words


def test_parsing():
    assert parse_formats("9:16, 1x1,1:1") == ["9:16", "1:1"]
    with pytest.raises(ValueError):
        parse_formats("3:2")
    assert parse_time("75") == 75 and parse_time("1:15") == 75 and parse_time("1:02:03") == 3723
    assert parse_time(None) is None and parse_time("") is None
    with pytest.raises(ValueError):
        parse_time("soon")
    opts = Options(formats="1:1,9:16", range_start="1:00", range_end="2:30").validate()
    assert opts.formats == ["1:1", "9:16"] and (opts.range_start, opts.range_end) == (60, 150)
    with pytest.raises(ClipperError, match="after its start"):
        Options(range_start="2:00", range_end="1:00").validate()
    with pytest.raises(ClipperError, match="no logo file"):
        Options(logo="/nope/logo.png").validate()


def test_frames_in_every_format():
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for name in ("9:16", "4:5", "1:1", "16:9"):
        fmt = get_format(name)
        cw = min(1920, round(1080 * fmt.ratio / 2) * 2)
        assert crop_frame(frame, 0, cw, fmt).shape == (fmt.height, fmt.width, 3)
        assert crop_frame(frame, 0, cw, fmt, zoom=1.12).shape == (fmt.height, fmt.width, 3)
        assert fit_frame(frame, fmt).shape == (fmt.height, fmt.width, 3)
    sq = get_format("1:1")
    assert split_frame(frame, [0, 0, 900, 0, 540], sq).shape == (1080, 1080, 3)


def test_zoom_punches_in_and_eases_out():
    windows = [(1.0, 2.0)]
    assert zoom_at(0.5, windows) == 1.0
    assert 1.0 < zoom_at(1.09, windows) < 1.12
    assert zoom_at(1.5, windows) == pytest.approx(1.12)
    assert 1.0 < zoom_at(2.2, windows) < 1.12 and zoom_at(3.0, windows) == 1.0
    words = map_words(make_words(["This one habit changed my whole life today friends."]), 0.0, EditMap([(0.0, 10.0)]))
    zw = zoom_windows(words, ["habit"], 10.0)
    assert zw[0][0] == 0.0  # the hook, and the keyword shortly after it, merge into one punch-in
    assert len(zw) == 1 and zw[0][1] >= words[2]["e"]


def test_logo_and_progress_bar(tmp_path):
    import cv2

    logo = np.zeros((40, 80, 4), np.uint8)
    logo[:, :, 2] = 255   # red
    logo[:, :, 3] = 255
    cv2.imwrite(str(tmp_path / "logo.png"), logo)
    fmt = get_format("9:16")
    ov = Overlay(Look(logo=tmp_path / "logo.png", progress="#00FF00"), fmt)
    img = ov.apply(np.zeros((fmt.height, fmt.width, 3), np.uint8), 0.5)
    x, y = ov.pos
    assert img[y + 5, x + 5, 2] > 200 and x > fmt.width / 2           # red logo at the top right
    assert (img[2, : fmt.width // 2 - 2] == [0, 255, 0]).all()         # half the bar is filled
    assert (img[2, fmt.width // 2 + 2:] == 0).all()


@pytest.mark.parametrize("ratio", ["9:16", "16:9", "1:1"])
def test_tall_and_16_bit_logos(tmp_path, ratio):
    import cv2

    logo = np.full((1200, 100, 4), 65535, np.uint16)  # a white, fully opaque, very tall 16-bit logo
    cv2.imwrite(str(tmp_path / "logo.png"), logo)
    fmt = get_format(ratio)
    for position in ("top-left", "bottom-right"):
        ov = Overlay(Look(logo=tmp_path / "logo.png", logo_position=position), fmt)
        img = ov.apply(np.zeros((fmt.height, fmt.width, 3), np.uint8), 0.0)  # used to crash on 16:9 and 1:1
        x, y = ov.pos
        h, w = ov.logo.shape[:2]
        assert 0 <= x and x + w <= fmt.width and 0 <= y and y + h <= fmt.height
        assert img[y + h // 2, x + w // 2].min() >= 225  # white at 90% opacity, not wrapped-around noise


def test_formats_change_the_caption_canvas():
    words = map_words(make_words(["We tried it. And it worked!"]), 0.0, EditMap([(0.0, 10.0)]))
    sq = build_ass(words, 5, fmt=get_format("1:1"))
    assert "PlayResX: 1080" in sq and "PlayResY: 1080" in sq
    margin = int(next(l for l in sq.splitlines() if l.startswith("Style: Default")).split(",")[-2])
    assert margin == round(520 * 1080 / 1920)


def test_srt():
    words = map_words(make_words(["Hello there, how are you today?", "Fine."]), 0.0, EditMap([(0.0, 10.0)]))
    srt = build_srt(words)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,")
    assert "Hello there, how are you today?" in srt and "\n2\n" in srt and "Fine." in srt


def test_translation(tmp_path, fake_claude):
    assert [w["w"] for w in spread(1.0, 2.0, "eins zwei")] == ["eins", "zwei"]
    assert spread(1.0, 2.0, "a bbbbbbb")[1]["s"] == pytest.approx(1.2)
    words = map_words(make_words(["Das ist gut.", "Wirklich gut!"]), 0.0, EditMap([(0.0, 10.0)]))
    out, title = translate_clip(tmp_path, words, "Titel", "English", lambda: Claude("test"))
    assert title == "[EN] Titel" and out[0]["w"] == "[EN]" and out[0]["s"] == words[0]["s"]
    assert out[-1]["e"] == pytest.approx(words[-1]["e"])
    calls = len(fake_claude.read_text().splitlines())
    translate_clip(tmp_path, words, "Titel", "English", lambda: Claude("test"))
    assert len(fake_claude.read_text().splitlines()) == calls  # cached


def test_focus_and_time_range_reach_claude(tmp_path, fake_claude):
    from clipper.select import select_clips

    words = make_words([f"Here is a thought number {i} worth sharing." for i in range(120)])
    meta = {"duration": words[-1]["e"] + 1, "title": "Test"}
    clips = select_clips(tmp_path, meta, {"words": words}, num_clips=2, min_len=15, max_len=30, model="t",
                         claude_factory=lambda: Claude("test"), focus="moments about money",
                         time_range=(120.0, 300.0))
    prompt = next(json.loads(l)["prompt"] for l in fake_claude.read_text().splitlines() if "<transcript>" in l)
    assert "looking for: moments about money" in prompt
    first_line = prompt.split("<transcript>\n")[1].split("\n")[0]
    assert float(first_line.split("-")[0]) >= 119.5
    assert all(119.5 <= c["start"] and c["end"] <= 330 for c in clips)


def test_translation_into_a_script_without_spaces():
    from clipper.captions import build_ass, srt_lines

    words = spread(0.0, 6.0, "我们今天去公园散步，然后吃了一顿非常好吃的晚饭。")
    assert len(words) > 4 and all(len(w["w"]) <= 5 for w in words)  # not one huge "word" running off screen
    assert "".join(w["w"] for w in words) == "我们今天去公园散步，然后吃了一顿非常好吃的晚饭。"
    assert [w["w"] for w in spread(0, 3, "今日はiPhone15で撮影しました")][:2] == ["今日は", "iPhone15"]
    ass = build_ass(words, 7.0)
    shown = [re.sub(r"\{[^}]*\}", "", l.split(",,0,0,0,,")[1]) for l in ass.splitlines() if l.startswith("Dialogue: 0")]
    assert "我们今天去公园散" in shown and not any(" " in t for t in shown)  # joined without spaces on screen
    assert all(len(text) <= 12 for _, _, text in srt_lines(words, 20))


def test_no_words_no_translation(tmp_path, fake_claude):
    assert translate_clip(tmp_path, [], "Titel", "English", lambda: Claude("test")) == ([], "Titel")
    assert not fake_claude.exists()
