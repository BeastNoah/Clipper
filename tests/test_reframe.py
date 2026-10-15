import numpy as np

from clipper.reframe import build_tracks, camera_path, crop_width, detect_cuts, main_track
from clipper.render import OUT_H, OUT_W, crop_frame, fit_frame


def test_crop_width():
    assert crop_width(1920, 1080) == 608
    assert crop_width(1080, 1920) == 1080


def test_detect_cuts_finds_spikes_not_motion():
    fps = 30
    changes = [0.0] + [2.0] * 59 + [60.0] + [2.0] * 59  # one hard cut at frame 60
    changes += [30.0] * 30                              # sustained fast motion: not a cut
    assert detect_cuts(changes, fps) == [60]
    # two spikes closer than the minimum shot length: only the first counts
    changes = [0.0] + [1.0] * 30 + [50.0] + [1.0] * 5 + [50.0] + [1.0] * 30
    assert detect_cuts(changes, fps) == [31]


def test_main_track_is_the_big_frequent_face():
    big = (300, 200, 80, 80)
    small = (100, 100, 25, 25)
    samples = [[big, small] if k % 4 else [small] for k in range(20)]  # big face in 75% of samples
    tracks = build_tracks(samples)
    assert len(tracks) == 2
    main = main_track(tracks, len(samples))
    assert main.boxes[1] == big
    assert main_track(build_tracks([[big]] + [[]] * 9), 10) is None  # seen once: not a clear face


def test_camera_ignores_jitter_then_eases_to_a_real_move():
    W, cw, fps = 1920, 608, 30
    frames = list(range(0, 300, 6))
    jitter = [960 + (15 if k % 2 else -15) for k in range(len(frames))]
    xs = camera_path(frames, jitter, 0, 299, W, cw, fps)
    assert len(set(xs)) == 1 and abs(xs[0] + cw // 2 - 960) <= 15

    centers = [700.0] * 25 + [1300.0] * 25  # the speaker walks to the right halfway through
    xs = camera_path(frames, centers, 0, 299, W, cw, fps)
    assert xs[0] == 700 - cw // 2 and xs[-1] == 1300 - cw // 2
    steps = np.diff(xs)
    assert (steps >= 0).all()                 # one smooth move, no back-and-forth
    assert steps.max() < 600 * 0.1            # eased over many frames, not a jump
    moving = np.nonzero(steps)[0]
    assert 0.5 * fps <= len(moving) <= 1.7 * fps


def test_camera_stays_inside_the_frame_and_fills_gaps():
    frames = list(range(0, 60, 6))
    centers = [10.0, None, None, 15.0, None, 12.0, None, None, None, 11.0]
    xs = camera_path(frames, centers, 0, 59, 1920, 608, 30)
    assert set(xs) == {0}


def test_frame_builders_make_vertical_frames():
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert crop_frame(frame, 500, 608).shape == (OUT_H, OUT_W, 3)
    out = fit_frame(frame)
    assert out.shape == (OUT_H, OUT_W, 3)
    fh = round(1080 * OUT_W / 1920)
    y0 = (OUT_H - fh) // 2
    assert np.abs(out[y0 + 10:y0 + fh - 10].astype(int).mean() - frame.mean()) < 3  # sharp copy in the middle
    assert out[:y0 - 5].mean() < frame.mean() * 0.7                                 # darkened blur above
