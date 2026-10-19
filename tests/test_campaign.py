import json

import pytest

from clipper.campaign import (checklist, enforce_post, focus_with, interpret, lengths_with, load_campaigns,
                              parse_rules, save_campaign, delete_campaign)
from clipper.llm import Claude, InvalidReply
from clipper.util import ClipperError

RULES = {"name": "BrandX", "focus": "the product", "avoid": ["swearing"], "min_len": 15, "max_len": 45,
         "hashtags": ["brandx"], "mentions": ["@brandx"], "must_include": ["link in bio"], "caption_notes": None,
         "disclosure": True, "platforms": ["tiktok"], "on_screen_text": None, "no_watermark": False, "manual": []}


def test_rules_are_cleaned_up():
    r = parse_rules(json.dumps({"hashtags": ["#BrandX", "brandx", " #two words "], "mentions": ["brandx", "@BrandX"],
                                "platforms": ["Instagram Reels", "Shorts", "myspace"], "manual": ["Use a sound", " "]}))
    assert r["hashtags"] == ["BrandX", "twowords"] and r["mentions"] == ["@brandx"]
    assert r["platforms"] == ["instagram", "youtube"] and r["manual"] == ["Use a sound"]
    with pytest.raises(InvalidReply):
        parse_rules('{"min_len": 60, "max_len": 30}')
    with pytest.raises(InvalidReply):
        parse_rules("no json here")


def test_required_text_is_always_in_the_post():
    post = {"tiktok": "So good #BrandX", "instagram": "Wow @BRANDX link in bio", "youtube_title": "A title",
            "youtube_description": "x" * 5000, "hashtags": ["fyp", "brandx"]}
    out = enforce_post(post, RULES)
    assert out["tiktok"] == "So good #BrandX\n\n@brandx link in bio #ad"  # #brandx was already there
    assert out["instagram"] == "Wow @BRANDX link in bio\n\n#brandx #ad"
    assert len(out["youtube_description"]) <= 5000 and out["youtube_description"].endswith("@brandx link in bio #brandx #ad")
    assert out["youtube_title"] == "A title" and out["hashtags"] == ["brandx", "ad", "fyp"]
    assert enforce_post(post, None) is post and enforce_post(out, RULES) == out  # nothing added twice


def test_lengths_and_focus():
    assert lengths_with(20, 60, None) == (20, 60) and lengths_with(20, 60, RULES) == (15, 45)
    assert lengths_with(20, 60, {"min_len": 90}) == (90, 130)   # only a minimum, above your maximum
    assert lengths_with(20, 60, {"max_len": 10}) == (5, 10)
    assert focus_with("funny bits", RULES) == "funny bits. the product. Never pick moments with: swearing"
    assert focus_with(None, None) is None and focus_with("x", None) == "x"
    assert checklist(RULES)[0] == "Post only to TikTok" and "paid partnership" in checklist(RULES)[1]


def test_saved_campaigns(tmp_path):
    a = save_campaign(tmp_path, "", "Clip our podcast", RULES)
    assert a["name"] == "BrandX" and load_campaigns(tmp_path) == [a]
    b = save_campaign(tmp_path, "Mine", "Other brief", None)
    a2 = save_campaign(tmp_path, "Renamed", "Clip our podcast, v2", RULES, cid=a["id"])
    assert a2["id"] == a["id"] and [c["name"] for c in load_campaigns(tmp_path)] == ["Mine", "Renamed"]
    delete_campaign(tmp_path, b["id"])
    assert [c["id"] for c in load_campaigns(tmp_path)] == [a["id"]]
    with pytest.raises(ClipperError, match="Paste"):
        save_campaign(tmp_path, "x", "   ")


def test_a_brief_is_read_once(tmp_path, fake_claude):
    rules = interpret("Clip BrandX's podcast. 8-12s. Tag @brandx.", tmp_path, lambda: Claude("opus"))
    assert rules["name"] == "BrandX" and rules["hashtags"] == ["brandx", "clips"] and rules["platforms"] == ["tiktok", "youtube"]
    assert interpret("Clip BrandX's podcast. 8-12s. Tag @brandx.", tmp_path, lambda: Claude("opus")) == rules
    assert len(fake_claude.read_text().splitlines()) == 1
