from clipper.edit import EditMap, is_filler, plan_cuts


def w(text, s, e):
    return {"w": " " + text, "s": s, "e": e}


WORDS = [w("So", 10.0, 10.3), w("this", 10.4, 10.7), w("um", 11.0, 11.3), w("is", 12.5, 12.8),
         w("it.", 12.9, 13.2), w("Next", 15.0, 15.3)]


def test_pauses_are_shortened_on_frame_boundaries():
    keep = plan_cuts(WORDS, 9.8, 15.6, 30, "pauses")
    # the 1.2s after "um" and the 1.8s before "Next" are shortened; the 0.3s before "um" stays
    assert len(keep) == 3
    for (_, a), (b, _), (word_end, next_start) in zip(keep, keep[1:], [(11.3, 12.5), (13.2, 15.0)]):
        assert abs(a - (word_end - 9.8 + 0.18)) <= 1 / 60 + 1e-9
        assert abs(b - (next_start - 9.8 - 0.08)) <= 1 / 60 + 1e-9
    assert all(abs(x * 30 - round(x * 30)) < 1e-9 for seg in keep for x in seg)


def test_filler_words_are_cut_with_the_pause_around_them():
    keep = plan_cuts(WORDS, 9.8, 15.6, 30, "on")
    assert len(keep) == 3
    first_cut = (keep[0][1], keep[1][0])
    assert first_cut[0] < 11.0 - 9.8 and first_cut[1] > 11.3 - 9.8  # "um" is inside the first cut
    assert is_filler(" Ähm,") and not is_filler("ham")


def test_filler_cut_never_reaches_into_overlapping_words():
    # Whisper's times often overlap: "basically" starts before "um" ends
    words = [w("So", 0.0, 0.5), w("um", 0.6, 1.0), w("basically", 0.9, 1.6), w("yes.", 1.7, 2.0)]
    keep = plan_cuts(words, 0.0, 3.0, 30, "on")
    spoken = [(0.0, 0.5), (0.9, 1.6), (1.7, 2.0)]
    for s, e in spoken:  # every spoken word is kept whole
        assert any(a <= s + 1e-9 and e - 1e-9 <= b for a, b in keep), (s, e, keep)


def test_audible_gaps_are_kept():
    def quiet(a, b):  # something audible (laughter, say) between 13.5s and 14.5s
        return b <= 13.5 or a >= 14.5

    keep = plan_cuts(WORDS, 9.8, 15.6, 30, "pauses", quiet=quiet)
    assert len(keep) == 2 and keep[1][1] == round(5.8 * 30) / 30
    assert keep[1][0] < 13.5 - 9.8  # the second, audible gap was left alone
    assert plan_cuts(WORDS, 9.8, 15.6, 30, "off") == [(0.0, round(5.8 * 30) / 30)]


def test_edit_map():
    m = EditMap([(0.0, 2.0), (3.0, 5.0)])
    assert m.duration == 4.0
    assert m.to_output(1.0) == 1.0 and m.to_output(2.5) == 2.0 and m.to_output(4.0) == 3.0
    assert m.frame_ranges(10) == [(0, 20), (30, 50)]
    assert not m.is_identity and EditMap([(0.0, 3.0)]).is_identity


def test_a_filler_right_after_a_word_is_cut_completely():
    words = [w("So,", 10.0, 10.3), w("um,", 10.3, 10.8), w("I", 11.0, 11.1), w("think", 11.2, 11.5)]
    keep = plan_cuts(words, 9.9, 11.7, 30, "on")
    edit = EditMap(keep)
    # "um" goes, apart from at most ~70 ms kept after "So," because Whisper's boundaries are fuzzy
    assert all(edit.is_cut(t - 9.9) for t in (10.38, 10.55, 10.78))
    assert not any(edit.is_cut(t - 9.9) for t in (10.1, 10.28, 11.05, 11.4))  # the real words stay
    tight = [w("So", 10.0, 10.3), w("uh", 10.35, 10.5), w("I", 10.55, 10.7)]
    assert EditMap(plan_cuts(tight, 9.9, 10.9, 30, "on")).is_cut(10.42 - 9.9)  # even with no pause around it
