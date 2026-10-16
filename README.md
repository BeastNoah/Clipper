# clipper

clipper turns long videos (podcasts, interviews, talks, streams) into short vertical clips for
TikTok, Reels and Shorts, with word-by-word captions burned in. Claude picks the moments;
everything else runs on your computer.

```bash
clipper ui                                          # the web interface
clipper "https://www.youtube.com/watch?v=..." -n 5  # or straight from the command line
clipper ~/Movies/interview.mp4 --min-len 30 --max-len 75
clipper "https://www.youtube.com/playlist?list=..."  # every video in a playlist
clipper talk.mp4 --prompt "funny moments" --from 10:00 --to 45:00 --format 9:16,1:1
```

Finished clips land in `work/<video_id>/output/`: `01_<title>.mp4`, `02_<title>.mp4`, … (best
first), each with a cover image (`.jpg`), a caption file (`.srt`) and post text with hashtags
(`.txt`). Extra formats get a suffix, e.g. `01_<title>_1x1.mp4`.

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

`clipper ui` opens a page in your browser. By default only this computer can use it.

- **Make clips:** paste a link or a file path, or drop a video file, and optionally say what Claude
  should look for ("funny moments", "everything about pricing"). Pick how many clips and how long;
  "More settings" has the rest: a time range, formats, translation, logo, zoom, progress bar,
  caption style and more.
- **Review:** with "Let me review the clips before rendering" ticked, clipper stops after Claude
  has made its suggestions. Each suggestion shows Claude's score and sub-scores, the hook, the
  highlight keywords and whether Claude thinks it repeats a better clip. You can preview it in
  the player (for YouTube links that's YouTube's own player, since only the sound has been
  downloaded at that point), tick the ones you want, rename them, and move the start or end by a sentence.
  Click two sentences in the transcript to add a clip of your own.
- **Render:** renders the ticked clips. Jobs run one at a time and show progress, a time
  estimate and the log; they can be cancelled.
- **Clips:** watch and download the finished clips (switch between formats), covers and caption
  files, and copy the post text for TikTok, Instagram or YouTube. **Fix captions** lets you correct
  misheard words; the clip is then rendered again with your text.

### On your phone

```bash
clipper ui --phone
```

clipper then also answers other devices on your network and shows a QR code in the terminal
(and under **Phone** in the interface). Scan it with the phone's camera while on the same Wi-Fi;
after that, a home-screen bookmark is enough. On the phone you can paste links, upload videos
from the camera roll, review, render and download the clips. The work still runs on the computer,
so it has to stay on.

The link contains an access key, kept in `work/.ui-key`; without it, other devices only get a
"needs the access link" page. `clipper ui --phone --new-key` makes a new key and locks out old
links. If the phone can't connect, allow incoming connections for Python when macOS asks (or in
System Settings > Network > Firewall). Away from home, a VPN such as Tailscale gives the phone a
way back to the computer; then use the computer's Tailscale address in the link.

## Command-line options

| option | default | |
|---|---|---|
| `-n`, `--num-clips` | 3 | clips per video |
| `--min-len` / `--max-len` | 20 / 60 | clip length limits in seconds |
| `--model` | `opus` | Claude model that picks the clips (`opus`, `sonnet`, `haiku` or a full name) |
| `--prompt TEXT` | | what Claude should look for, e.g. `"funny moments"` or `"everything about pricing"` |
| `--from` / `--to` | | only use part of the video, e.g. `--from 10:00 --to 45:00` |
| `--whisper-model` | `small` | model for the full transcript |
| `--caption-model` | `turbo` | bigger model that re-transcribes only the chosen clips, for accurate captions; `none` reuses the transcript |
| `--whisper-backend` | `auto` | `mlx` (Apple Silicon GPU) when installed, else `faster-whisper` (CPU) |
| `--language` | detect | spoken language, e.g. `en`, `de` |
| `--caption-language` | | translate the captions (and title), e.g. `English`; also writes a translated `.srt` |
| `--caption-style` | `bold` | `bold` (heavy white text, yellow pop), `clean` (lighter, no pop), `boxed` (text on a dark box) |
| `--font`, `--caption-size`, `--highlight-color`, `--uppercase` | | fine-tune the captions |
| `--caption-position` | `lower` | `lower` third or `middle` |
| `--no-title` | | don't show the clip title for the first seconds |
| `--format` | `9:16` | output shapes, comma separated: `9:16`, `4:5`, `1:1`, `16:9` |
| `--no-zoom` | | no automatic zoom-ins on the hook and Claude's keywords |
| `--progress-bar` | | a progress bar along the top |
| `--logo FILE`, `--logo-position` | top-right | your logo on every clip (a PNG with transparency works best) |
| `--jump-cuts` | `on` | `on` cuts long pauses and filler words ("um", "äh"), `pauses` only pauses, `off` nothing |
| `--layout` | `auto` | two people in a shot: `follow` whoever talks, `split` screen, `auto` (split during back-and-forth, follow otherwise), `main` person only |
| `--encoder` | `auto` | `auto` uses the Mac's hardware encoder (`videotoolbox`: much faster, slightly bigger files) when there is one, else `x264` (best quality per MB) |
| `--crf` / `--preset` | 20 / medium | x264 quality and speed |
| `--no-posts` | | don't write post text |
| `--download` | `sections` | `sections` takes only the sound at first, then just the video of the chosen clips (from the link or from your file); `full` downloads and converts the whole video first |
| `--work-dir` | `./work` | where results are kept |
| `--force STEP` | | redo STEP and every step after it |
| `--stop-after STEP` | | stop early, e.g. `--stop-after select` to see Claude's picks before rendering |

Steps are `ingest`, `transcribe`, `select`, `reframe`, `captions`, `render`.

## How it works

clipper runs in six steps. Each step saves its result in `work/<video_id>/`, so a rerun skips
anything already done. Changing only `--num-clips`, for example, reuses the download, the
transcript and Claude's proposals.

**1. Ingest** (`ingest.py`). clipper starts with just the sound: that's all the next two steps
need. For a link, yt-dlp downloads only the audio, in a smaller version that's plenty for speech
recognition. For a file on your computer, ffmpeg copies its sound out in seconds (the clips are
later cut from the original, so leave it where it is until you're done). The picture is taken
later and only for the clips (plus 3 seconds around each): ffmpeg reads just that stretch of the
stream or file, converts it to H.264 at a constant frame rate (at most 1080p), and the section's
sound is matched against the full audio track so captions and cuts line up exactly. While one clip
is being made, the next ones are fetched in the background. For a two-hour podcast that means a
few minutes of video instead of gigabytes, and no conversion of the whole thing. `--download full`
downloads and converts the whole video up front instead. The sound is saved as a 16 kHz mono file
for the speech recognizer. Playlists are expanded into their videos.

**2. Transcribe** (`transcribe.py`). Whisper writes down what's said, with a start and end time
for every word: mlx-whisper on an Apple Silicon GPU, or faster-whisper on the CPU (the `small`
model runs at about 7–10x real time there, so a 20-minute video takes 2–3 minutes).

**3. Select the clips** (`select.py`, `audio.py`). This is where Claude comes in.
- The transcript is split into sentences and sent to Claude with timestamps, one per line (only
  the part between `--from` and `--to`, if given, and with your `--prompt` saying what to look for). Lines
  carry cues from the audio: `[loud]`, `[+1.8s non-speech sound]` (often laughter or applause)
  and `[pause 2.1s]`, so Claude also notices reactions, not just words.
- Acting as a short-form video editor, Claude proposes clips, each with a title, the opening hook
  line, 1–3 keywords to highlight, and a 0–100 score made of four 0–25 sub-scores: hook
  strength, makes sense on its own, emotion/surprise/humour, and clarity.
- Long videos are sent in overlapping 15-minute chunks (the overlap is longer than the longest
  clip, so no moment gets cut in half), up to four at the same time. Claude then sees all
  candidates together, re-scores them on one common scale and marks any that repeat the point of
  a better one.
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
lower third, above where TikTok and Instagram put their own text and buttons. With
`--caption-language`, Claude translates each sentence and its time is spread over the translated
words. Corrections made in the web interface replace the transcribed words. A `.srt` caption file
is written for each clip as well.

**6. Render** (`render.py`, `posts.py`). ffmpeg decodes the clip frame by frame; Python builds each
frame (window, split screen or blurred background), zooms in a little on the hook and on Claude's
keywords, draws the logo and progress bar, skips the frames removed by jump cuts, and hands the
rest to a second ffmpeg, which burns in the captions, adds the matching audio and writes the MP4
(with the Mac's hardware encoder when there is one):
1080×1920 for 9:16, 1080×1350 for 4:5, 1080×1080 for 1:1, 1920×1080 for 16:9. Each format gets its
own framing (split screen only in the tall ones). A cover image and post text for each platform
are written alongside.

## The work folder

```
work/<video_id>/
  meta.json            size, frame rate, duration, title
  video.mp4            the whole video, converted (only with --download full)
  audio_source.*       the downloaded sound (links)
  thumb.jpg            the video's thumbnail (links)
  audio.wav            16 kHz mono audio
  transcript.json      every word with its start and end time
  select.json          Claude's proposals, trims, scores and re-scores
  clips.json           Claude's picks
  review.json          your picks from the web interface
  posts.json           post text per clip
  clips/<start>-<end>/ per clip: section.mp4 (just this part of the video), words.json,
                       words_edited.json (your caption fixes), and per format (9x16/, 1x1/, …):
                       reframe.json, captions.ass, clip.mp4, cover.jpg
  output/              the finished clips, covers, caption files, post text and clips.json
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
