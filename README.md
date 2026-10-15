# clipper

clipper turns one long video (a podcast, interview, talk or stream) into a few short vertical
clips for TikTok, Reels and Shorts, with word-by-word captions burned in. Claude picks the
moments; everything else runs locally.

```bash
clipper "https://www.youtube.com/watch?v=..." -n 5
clipper ~/Movies/interview.mp4 --min-len 30 --max-len 75
```

The finished clips land in `work/<video_id>/output/` as `01_<title>.mp4`, `02_<title>.mp4`, …
(ranked best first), next to a `clips.json` with each clip's title, hook, score and timing.

## Install (macOS)

You need Python 3.10+, ffmpeg built with libass (for the captions), and the Claude Code CLI
logged in to your Claude account.

```bash
brew install ffmpeg deno       # deno: yt-dlp needs a JavaScript runtime for YouTube
                               # if captions fail with "built without libass": brew install ffmpeg-full
npm install -g @anthropic-ai/claude-code && claude   # log in once
git clone https://github.com/BeastNoah/Clipper && cd Clipper
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

On first use clipper downloads the Whisper speech model (about 500 MB for `small`) and OpenCV's
YuNet face model (about 230 KB, cached in `~/.cache/clipper`). If YuNet can't be downloaded it
falls back to OpenCV's built-in face detector.

## Options

| option | default | |
|---|---|---|
| `-n`, `--num-clips` | 3 | how many clips to make |
| `--min-len` / `--max-len` | 20 / 60 | clip length limits in seconds |
| `--model` | `opus` | Claude model that picks the clips (`opus`, `sonnet`, `haiku`, or a full model name) |
| `--whisper-model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, … (bigger is slower and more accurate) |
| `--language` | detect | spoken language, e.g. `en`, `de` |
| `--font` | Arial Black | caption font |
| `--crf` / `--preset` | 20 / medium | x264 quality and speed |
| `--work-dir` | `./work` | where results are kept |
| `--force STEP` | | redo STEP and every step after it |
| `--stop-after STEP` | | stop early, e.g. `--stop-after select` to see Claude's picks before rendering |

Steps are `ingest`, `transcribe`, `select`, `reframe`, `captions`, `render`.

## How it works

clipper runs in six steps. Each step saves its result in `work/<video_id>/`, so a rerun skips
anything already done. Changing only `--num-clips`, for example, reuses the download, the
transcript and Claude's proposals.

**1. Ingest** (`ingest.py`). Takes a local file, or downloads a URL with yt-dlp (best quality up
to 1080p). ffmpeg converts it to H.264 at a constant frame rate, which keeps sound and picture
in sync later, and pulls out the audio as a 16 kHz mono file for the speech recognizer.

**2. Transcribe** (`transcribe.py`). faster-whisper, a local speech-to-text model, writes down
what's said, with a start and end time for every word. On a Mac's CPU the `small` model runs at
about 7–10x real time, so a 20-minute video takes 2–3 minutes.

**3. Select the clips** (`select.py`). This is where Claude comes in.
- The transcript is split into sentences and sent to Claude with timestamps, one per line.
- Acting as a short-form video editor, Claude proposes clips, each with a title, the opening hook
  line and a 0–100 score made of four 0–25 sub-scores: hook strength, makes sense on its own,
  emotion/surprise/humour, and clarity.
- Long videos are sent in overlapping 15-minute chunks (the overlap is longer than the longest
  clip, so no moment gets cut in half). Claude then sees all candidates together and re-scores
  them on one common scale.
- Claude doesn't get the final word on timing. clipper moves each start and end to the nearest
  sentence boundary so no word is cut off, then trims the end or adds sentences until the clip
  fits your length limits. Where two clips overlap, the higher-scoring one wins. If too few
  clips survive, Claude is asked for more (up to two extra rounds), excluding what's taken.
- Requests go through the `claude` command-line tool, so they use your subscription. Each answer
  must match a fixed JSON format; if it doesn't, the validation error goes back to Claude in the
  same conversation and it tries again.

**4. Reframe to vertical** (`reframe.py`). A 16:9 frame becomes 9:16 by cutting a tall window out
of it. clipper finds camera cuts, looks for faces 5 times a second, and in each shot follows the
main person: the face that is biggest and on screen most often. The window moves like a calm
camera operator: it ignores small head movements, only moves when the person really shifts,
then eases over in one smooth move, and it resets at every cut. A shot with no clear face (a
title card, a wide shot) shows the full frame over a blurred copy of itself instead.

**5. Captions** (`captions.py`). Using the word timings, words are grouped into short lines of
2–4 words that never run across a sentence and fit the width of the frame. The result is an ASS
subtitle file in which the word being spoken lights up yellow with a small pop. The text sits in
the lower third, above where TikTok and Instagram put their own caption and buttons.

**6. Render** (`render.py`). ffmpeg decodes the clip frame by frame; Python cuts out the vertical
window (or builds the blurred-background frame) and hands each frame to a second ffmpeg, which
burns in the captions, adds the original audio for the same time range and writes a 1080×1920
MP4.

## The work folder

```
work/<video_id>/
  meta.json            size, frame rate, duration, title
  video.mp4            normalised video
  audio.wav            16 kHz mono audio
  transcript.json      every word with its start and end time
  select.json          Claude's proposals, scores and re-scores
  clips.json           the chosen clips
  clips/<start>-<end>/ per clip: reframe.json (window position per frame), captions.ass, clip.mp4
  output/              the finished, ranked clips and clips.json
```

Per-clip results are stored by time range, so a clip that's still selected after you change
options isn't analysed or rendered again. Delete a video's folder to start over completely.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The tests use `tests/fake_claude.py` in place of the real `claude` command, and the pipeline test
generates a short video with ffmpeg, so they need ffmpeg but no network or Claude account.

Run clipper through the `clipper` command or `python -m clipper`, not by running files inside
the `clipper/` folder directly: `select.py` would then shadow Python's built-in `select` module.
