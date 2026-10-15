#!/usr/bin/env python3
"""Stand-in for the `claude` CLI in tests: answers clipper's prompts with plausible JSON.

FAKE_CLAUDE_LOG: append each call (argv + prompt) as a JSON line.
FAKE_CLAUDE_BAD_FIRST=1: the first answer in a conversation is malformed, to exercise the retry.
FAKE_CLAUDE_MAX: propose at most this many clips per request.
"""
import json
import os
import re
import sys

argv = sys.argv[1:]
prompt = sys.stdin.read()
resume = argv[argv.index("--resume") + 1] if "--resume" in argv else None
if os.environ.get("FAKE_CLAUDE_LOG"):
    with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
        f.write(json.dumps({"argv": argv, "prompt": prompt}) + "\n")


SESSION = argv[argv.index("--session-id") + 1] if "--session-id" in argv else resume


def reply(obj, session=SESSION):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    print(json.dumps({"type": "result", "is_error": False, "result": text, "session_id": session}))
    sys.exit(0)


if os.environ.get("FAKE_CLAUDE_BAD_FIRST") and not resume:
    reply("Sure! Here are some great clips: (no JSON, oops)")

if "<transcript>" in prompt:
    body = prompt.split("<transcript>")[1].split("</transcript>")[0]
    lines = [re.match(r"([\d.]+)-([\d.]+) \| (.*)", l) for l in body.strip().splitlines()]
    lines = [(float(m[1]), float(m[2]), m[3]) for m in lines if m]
    lo = float(re.search(r"must be (\d+)-(\d+) seconds", prompt)[1])
    count = int(re.search(r"Find up to (\d+)", prompt)[1])
    count = min(count, int(os.environ.get("FAKE_CLAUDE_MAX", "99")))
    avoid = [(float(a), float(b)) for a, b in re.findall(r"^- ([\d.]+)-([\d.]+)$", prompt, re.M)]
    clips, i = [], 0
    while i < len(lines) and len(clips) < count:
        s = lines[i][0]
        j = i
        while j < len(lines) and lines[j][1] - s < lo + 5:
            j += 1
        if j >= len(lines):
            break
        e = lines[j][1]
        if not any(s < b and e > a for a, b in avoid):
            n = len(clips)
            clips.append({"start": s, "end": e, "title": f"Moment {s:.0f}", "hook": lines[i][2],
                          "scores": {"hook": 10 + n % 15, "standalone": 15, "emotion": 12, "clarity": 20}})
        i = j + 2
    reply({"clips": clips})

if "<clips>" in prompt:
    ids = re.findall(r"^\[(c\d+)\]", prompt, re.M)
    reply({"scores": [{"id": c, "hook": 25 - k % 20, "standalone": 20, "emotion": 15, "clarity": 18}
                      for k, c in enumerate(ids)]})

if resume:  # a correction request after a bad first answer: the original prompt isn't resent
    reply({"clips": []})

reply({"error": "unrecognised prompt"})
