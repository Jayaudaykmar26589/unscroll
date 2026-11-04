"""Translation product tests (offline, deterministic)."""
import os

import pytest

os.environ["UNROLL_OFFLINE"] = "1"

from src.agents import translate as T


def test_exact_phrase():
    r = T.translate("χαῖρε κόσμε")
    assert r["translation"] == "Hello world"
    assert r["conf"] == 1.0 and r["offline"] is True


def test_gloss_mode_marks_unknown():
    r = T.translate("καὶ ὁ μῦθος", mode="gloss")
    assert "καὶ = and" in r["translation"]
    assert "μῦθος = ?" in r["translation"]
    assert r["conf"] == pytest.approx(2 / 3, abs=1e-3)


def test_known_words_full_conf():
    r = T.translate("καὶ ὁ λόγος", mode="gloss")
    assert r["conf"] == 1.0
    assert "λόγος = word" in r["translation"]


def test_multitarget_stub_prefix():
    r = T.translate("χαῖρε κόσμε", target="de")
    assert r["translation"] == "[de-STUB] Hello world"
    assert r["target"] == "de"


def test_invalid_args():
    with pytest.raises(ValueError):
        T.translate("   ")
    with pytest.raises(ValueError):
        T.translate("χαῖρε", target="xx")
    with pytest.raises(ValueError):
        T.translate("χαῖρε", mode="poetic")


def test_batch_readings_skips_stub():
    readings = [{"text": "χαῖρε κόσμε", "conf": 0.9},
                {"text": "[OFFLINE-STUB] x", "conf": 0.0}]
    out = T.translate_readings(readings)
    assert out[0]["translation"] == "Hello world"
    assert out[0]["tconf"] == 1.0
    assert "skipped" in out[1]["translation"]
