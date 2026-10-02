"""Guards against a real failure seen during development: a regex word boundary
silently written as a literal backspace character, which makes the pattern
match nothing without raising any error."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_control_or_invisible_characters_in_source():
    invisible = {0x00A0, 0xFEFF, 0x200B, 0xFFFD}
    for f in list((ROOT / "refdist").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        bad = {hex(ord(c)) for c in f.read_text(encoding="utf-8")
               if (ord(c) < 32 and c not in "\n\t\r") or 0x80 <= ord(c) <= 0x9F or ord(c) in invisible}
        assert not bad, f"{f}: {bad}"


def test_clean_repairs_encoding_damage():
    import sys
    sys.path.insert(0, str(ROOT))
    from refdist.corpora.base import clean
    assert clean("a\u0097b &amp;amp; c\u00a0d") == "a\u2014b & c d"


def test_config_imports_and_enforces_independence():
    import sys
    sys.path.insert(0, str(ROOT))
    from refdist import config
    assert config.INDEPENDENT_PREDICTOR["model_id"] not in config._detector_model_ids()
    assert all(len(v["revision"]) == 40 for v in config.REFERENCE_MODELS.values())
