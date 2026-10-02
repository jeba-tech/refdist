"""Properties the OSF registration promises about the H5 transform library."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refdist.transforms.base import apply  # noqa: E402
from refdist.transforms.rules import TRANSFORMS  # noqa: E402

TEXT = (
    "I don't think the program is ready. I think it is. It's been tested, and it works on my machine, "
    "but they're worried about the color scheme. We've organized a meeting at the center. "
    "Honestly, I believe that we can obtain a new license in order to proceed. The behavior is odd, so we'll see. "
    "She said that it isn't finished. They can't agree, yet the team realized the problem."
)
SEED = 42


@pytest.mark.parametrize("key", list(TRANSFORMS))
def test_intensity_zero_is_byte_identical(key):
    out, sites, _ = apply(TRANSFORMS[key], TEXT, 0.0, "doc", SEED)
    assert out == TEXT and sites == []


@pytest.mark.parametrize("key", list(TRANSFORMS))
def test_every_transform_finds_sites(key):
    assert apply(TRANSFORMS[key], TEXT, 1.0, "doc", SEED)[2] > 0


@pytest.mark.parametrize("key", list(TRANSFORMS))
def test_intensities_are_nested(key):
    applied = [{(s.start, s.end) for s in apply(TRANSFORMS[key], TEXT, p, "doc", SEED)[1]}
               for p in (0.25, 0.5, 1.0)]
    assert applied[0] <= applied[1] <= applied[2]


@pytest.mark.parametrize("key", list(TRANSFORMS))
def test_deterministic(key):
    assert apply(TRANSFORMS[key], TEXT, 0.5, "doc", SEED) == apply(TRANSFORMS[key], TEXT, 0.5, "doc", SEED)


@pytest.mark.parametrize("key", list(TRANSFORMS))
def test_recorded_sites_reproduce_the_edit(key):
    out, sites, _ = apply(TRANSFORMS[key], TEXT, 1.0, "doc", SEED)
    rebuilt, pos = [], 0
    for s in sites:
        assert TEXT[s.start:s.end] == s.original
        rebuilt += [TEXT[pos:s.start], s.replacement]
        pos = s.end
    assert "".join(rebuilt) + TEXT[pos:] == out


def test_t1_expands_correctly():
    out = apply(TRANSFORMS["T1"], TEXT, 1.0, "doc", SEED)[0]
    for want in ("do not think", "It has been tested", "they are worried", "We have organized",
                 "we will see", "is not finished", "cannot agree"):
        assert want in out, want


def test_t3_preserves_length_within_2_percent():
    out = apply(TRANSFORMS["T3"], TEXT, 1.0, "doc", SEED)[0]
    assert abs(len(out.split()) - len(TEXT.split())) / len(TEXT.split()) <= 0.02
    assert ". And it works" in out and ". Yet the team" in out


def test_t5_spelling():
    out = apply(TRANSFORMS["T5"], TEXT, 1.0, "doc", SEED)[0]
    assert all(w in out for w in ("colour", "organised", "centre", "behaviour", "realised"))
    assert "program" in out and "licence" not in out  # ambiguous forms are excluded


def test_t6_inserts_complementiser_before_subject_only():
    text = "I think it works. I know the answer. She said they left, and we hope you agree."
    out = apply(TRANSFORMS["T6"], text, 1.0, "doc", SEED)[0]
    assert "think that it works" in out and "said that they left" in out and "hope that you agree" in out
    assert "know the answer" in out  # object, not a clause: untouched


def test_t2_moves_to_commoner_word():
    from wordfreq import zipf_frequency
    _, sites, _ = apply(TRANSFORMS["T2"], TEXT, 1.0, "doc", SEED)
    assert sites
    for s in sites:
        assert zipf_frequency(s.replacement.lower(), "en") > zipf_frequency(s.original.lower(), "en")
