"""Harness2-loop tests (offline, no network, tmp lessons)."""
import os

os.environ["UNROLL_OFFLINE"] = "1"

from src.agents import harness as H


def test_score_checks():
    good = {"translation": "Hello world", "conf": 1.0}
    s = H.score_translation(good, "χαῖρε κόσμε")
    assert s["total"] >= 0.85 and s["checks"]["not_echo"]
    echo = {"translation": "χαῖρε κόσμε", "conf": 0.5}
    s2 = H.score_translation(echo, "χαῖρε κόσμε")
    assert s2["checks"]["not_echo"] is False
    assert s2["total"] < s["total"]


def test_improve_offline_trace(tmp_path):
    lp = tmp_path / "lessons.jsonl"
    out = H.improve("χαῖρε κόσμε", budget=1, lessons_path=lp)
    assert out["result"]["translation"] == "Hello world"
    assert out["base_score"] == 1.0
    names = [t["harness"] for t in out["trace"]]
    assert names[0] == "default-v1"
    assert any("+invariant" in n for n in names)
    assert any("+domain" in n for n in names)
    # No improvement over perfect baseline -> no lesson written.
    assert out["lesson"] is None
    assert not lp.exists()


def test_reflect_writes_on_delta(tmp_path):
    lp = tmp_path / "lessons.jsonl"
    base = H.DEFAULT_HARNESS
    weak = {"translation": "χαῖρε κόσμε", "conf": 0.5}
    strong = {"translation": "Hello world", "conf": 1.0}
    sb = H.score_translation(weak, "χαῖρε κόσμε")
    ss = H.score_translation(strong, "χαῖρε κόσμε")
    rec = H.reflect("χαῖρε κόσμε", (base, strong, ss), sb["total"], path=lp)
    assert rec is not None and rec["delta"] > 0
    assert lp.exists()
    loaded = H.load_lessons(lp)
    assert loaded and "lesson" in loaded[0]
