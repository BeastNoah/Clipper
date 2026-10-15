# clipper

clipper turns long videos (podcasts, interviews, talks, streams) into short vertical clips for
TikTok, Reels and Shorts, with word-by-word captions burned in. Claude picks the moments;
everything else runs on your computer.

```bash
clipper ui                                          # the web interface
clipper "https://www.youtube.com/watch?v=..." -n 5  # or straight from the command line
clipper ~/Movies/interview.mp4 --min-len 30 --max-len 75
clipper "https://www.youtube.com/playlist?list=..."  # every video in a playlist
```

Finished clips land in `work/<video_id>/output/`: `01_<title>.mp4`, `02_<title>.mp4`, … (best
first), each with a cover image (`.jpg`) and post text with hashtags (`.txt`).

## Install (macOS)

You need Python 3.10+, ffmpeg built with libass (for the captions), and the Claude Code CLI
logged in to your Claude account.

```bash
brew install ffmpeg deno       # deno: yt-dlp needs a JavaScript runtime for YouTube
npm install -g @anthropic-ai/claude-code && claude   # log in once
git clone https://github.com/BeastNoah/Clipper && cd Clipper
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mac]'        # [mac] adds mlx-whisper: much faster speech recognition on Apple Silicon
clipper doctor                 # checks all of the above and says what's missing
```

If captions fail with "built without libass", install a full ffmpeg build (`brew install
ffmpeg-full`). On first use clipper downloads the Whisper models (about 500 MB for `small`, the
transcript model, and 1.6 GB for `turbo`, the caption model) and OpenCV's YuNet face model
(230 KB, kept in `~/.cache/clipper`).

## The web interface

`clipper ui` opens a page in your browser (it only listens on 127.0.0.1).

- **Make clips:** paste a link or a file path, or drop a video file. Pick how many clips and how
  long; more settings are under "More settings".
- **Review:** with "Let me review the clips before rendering" ticked, clipper stops after Claude
  has made its suggestions. Each suggestion shows Claude's score and sub-scores, the hook, the
  highlight keywords and whether Claude thinks it repeats a better clip. You can preview it in
  the player, tick the ones you want, rename them, and move the start or end by a sentence.
  Click two sentences in the transcript to add a clip of your own.
- **Render:** renders the ticked clips. Jobs run one at a time and show progress, a time
  estimate and the log; they can be cancelled.
- **Clips:** watch and download the finished clips and covers, and copy the post text for
  TikTok, Instagram or YouTube.

## Command-line options

| option | default | |
|---|---|---|
| `-n`, `--num-clips` | 3 | clips per video |
| `--min-len` / `--max-len` | 20 / 60 | clip length limits in seconds |
| `--model` | `opus` | Claude model that picks the clips (`opus`, `sonnet`, `haiku` or a full name) |
| `--whisper-model` | `small` | model for the full transcript |
| `--caption-model` | `turbo` | bigger model that re-transcribes only the chosen clips, for accurate captions; `none` reuses the transcript |
| `--whisper-backend` | `auto` | `mlx` (Apple Silicon GPU) when installed, else `faster-whisper` (CPU) |
| `--language` | detect | spoken language, e.g. `en`, `de` |
| `--caption-style` | `bold` | `bold` (heavy white text, yellow pop), `clean` (lighter, no pop), `boxed` (text on a dark box) |
| `--font`, `--caption-size`, `--highlight-color`, `--uppercase` | | fine-tune the captions |
| `--caption-position` | `lower` | `lower` third or `middle` |
| `--no-title` | | don't show the clip title for the first seconds |
| `--jump-cuts` | `on` | `on` cuts long pauses and filler words ("um", "äh"), `pauses` only pauses, `off` nothing |
| `--layout` | `auto` | two people in a shot: `follow` whoever talks, `split` screen, `auto` (split during back-and-forth, follow otherwise), `main` person only |
| `--encoder` | `x264` | `videotoolbox` uses the Mac's hardware encoder: much faster, slightly bigger files |
| `--crf` / `--preset` | 20 / medium | x264 quality and speed |
| `--no-posts` | | don't write post text |
| `--work-dir` | `./work` | where results are kept |
| `--force STEP` | | redo STEP and every step after it |
| `--stop-after STEP` | | stop early, e.g. `--stop-after select` to see Claude's picks before rendering |

Steps are `ingest`, `transcribe`, `select`, `reframe`, `captions`, `render`.

## How it works

clipper runs in six steps. Each step saves its result in `work/<video_id>/`, so a rerun skips
anything already done. Changing only `--num-clips`, for example, reuses the download, the
transcript and Claude's proposals.

**1. Ingest** (`ingest.py`). Takes a local file, or downloads a URL with yt-dlp (best quality up
to 1080p; playlists are expanded into their videos). ffmpeg converts it to H.264 at a constant
frame rate, which keeps sound and picture in sync later, and pulls out the audio as a 16 kHz mono
file for the speech recognizer.

**2. Transcribe** (`transcribe.py`). Whisper writes down what's said, with a start and end time
for every word: mlx-whisper on an Apple Silicon GPU, or faster-whisper on the CPU (the `small`
model runs at about 7–10x real time there, so a 20-minute video takes 2–3 minutes).

**3. Select the clips** (`select.py`, `audio.py`). This is where Claude comes in.
- The transcript is split into sentences and sent to Claude with timestamps, one per line. Lines
  carry cues from the audio: `[loud]`, `[+1.8s non-speech sound]` (often laughter or applause)
  and `[pause 2.1s]`, so Claude also notices reactions, not just words.
- Acting as a short-form video editor, Claude proposes clips, each with a title, the opening hook
  line, 1–3 keywords to highlight, and a 0–100 score made of four 0–25 sub-scores: hook
  strength, makes sense on its own, emotion/surprise/humour, and clarity.
- Long videos are sent in overlapping 15-minute chunks (the overlap is longer than the longest
  clip, so no moment gets cut in half). Claude then sees all candidates together, re-scores
  them on one common scale and marks any that repeat the point of a better one.
- clipper moves each start and end to the nearest sentence boundary so no word is cut off. If a
  clip then runs too long, Claude chooses which sentences to keep (so the punchline survives);
  if it's too short, code adds the next sentence. Where clips overlap or repeat each other, the
  higher-scoring one wins. If too few survive, Claude is asked for more.
- Requests go through the `claude` command-line tool, so they use your subscription. Each answer
  must match a fixed JSON format; if it doesn't, the validation error goes back to Claude in the
  same conversation and it tries again.

**4. Reframe to vertical** (`reframe.py`). A 16:9 frame becomes 9:16 by cutting a tall window out
of it. clipper finds camera cuts and looks for faces 5 times a second. In each shot it follows the
main person (the biggest face, on screen most often) like a calm camera operator: it ignores small
head movements and eases over in one smooth move when the person really shifts. When two people
share a shot, it watches whose mouth moves while someone is speaking: it cuts to that person, or,
while they go back and forth, stacks both in a split screen. Shots without a clear face (a title
card, a wide shot) show the full frame over a blurred copy of itself.

**5. Captions** (`transcribe.py`, `edit.py`, `captions.py`). Each chosen clip is transcribed again
with a bigger model, because misheard words in captions are what viewers notice. Jump cuts remove
long pauses and filler words, but only where the audio is really quiet, so laughter or words
Whisper missed are never cut. Words are grouped into lines of 2–4 that never run across a sentence
and fit the frame; the word being spoken lights up with a small pop and Claude's keywords stay in
a second colour. The clip's title is shown at the top for the first seconds. Captions sit in the
lower third, above where TikTok and Instagram put their own text and buttons.

**6. Render** (`render.py`, `posts.py`). ffmpeg decodes the clip frame by frame; Python builds each
vertical frame (window, split screen or blurred background), skips the frames removed by jump
cuts, and hands the rest to a second ffmpeg, which burns in the captions, adds the matching audio
and writes a 1080×1920 MP4. A cover image and post text for each platform are written alongside.

## The work folder

```
work/<video_id>/
  meta.json            size, frame rate, duration, title
  video.mp4            normalised video
  audio.wav            16 kHz mono audio
  transcript.json      every word with its start and end time
  select.json          Claude's proposals, trims, scores and re-scores
  clips.json           Claude's picks
  review.json          your picks from the web interface
  posts.json           post text per clip
  clips/<start>-<end>/ per clip: words.json, reframe.json, captions.ass, clip.mp4, cover.jpg
  output/              the finished clips, covers, post text and clips.json
```

Per-clip results are stored by time range, so a clip that's still selected after you change
options isn't analysed or rendered again. Delete a video's folder to start over completely.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The tests use `tests/fake_claude.py` in place of the real `claude` command and generate short
videos with ffmpeg, so they need ffmpeg but no network or Claude account.

Run clipper through the `clipper` command or `python -m clipper`, not by running files inside the
`clipper/` folder directly: `select.py` would then shadow Python's built-in `select` module.
