"""Product translation layer: ancient Greek -> modern languages.

Offline-first: exact-glossary + word-fallback stubs make the demo work
with UNROLL_OFFLINE=1 or no key. Live path goes through Spark.
"""
import os

from src.agents import spark

SOURCE_LANG = "grc"

SUPPORTED_TARGETS = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "ta": "Tamil",
}

MODES = ("fluent", "gloss")

# Exact-phrase hits (offline + live fallback).
PHRASE_GLOSSARY = {
    "χαῖρε κόσμε": "Hello world",
    "χαῖρε": "Hail, hello",
    "καὶ": "and",
}

# Word-level fallback for offline gloss/fluent composition.
WORD_GLOSSARY = {
    "χαῖρε": "hail",
    "κόσμε": "world",
    "καὶ": "and",
    "ὁ": "the",
    "ἡ": "the",
    "τό": "the",
    "θεός": "god",
    "ἄνθρωπος": "human",
    "λόγος": "word",
    "βίος": "life",
}

STUB_TAG = "[OFFLINE-STUB]"


def _check_args(text, target, mode):
    if not text or not text.strip():
        raise ValueError("empty text")
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target {target!r}; want one of {sorted(SUPPORTED_TARGETS)}")
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}; want one of {list(MODES)}")


def _offline_en(text, mode):
    clean = text.strip()
    if clean in PHRASE_GLOSSARY:
        return PHRASE_GLOSSARY[clean], 1.0
    words = clean.split()
    if mode == "gloss":
        parts, known = [], 0
        for w in words:
            g = WORD_GLOSSARY.get(w)
            if g is None:
                parts.append(f"{w} = ?")
            else:
                parts.append(f"{w} = {g}")
                known += 1
        conf = known / len(words) if words else 0.0
        return " | ".join(parts), round(conf, 3)
    out, known = [], 0
    for w in words:
        g = WORD_GLOSSARY.get(w)
        if g is None:
            out.append(f"{w}?")
        else:
            out.append(g)
            known += 1
    conf = known / len(words) if words else 0.0
    return " ".join(out), round(conf, 3)


def _system_prompt(target, mode, harness=None):
    lang = SUPPORTED_TARGETS[target]
    if harness is not None:
        return harness.to_system(target, mode)
    if mode == "gloss":
        return (f"Translate ancient Greek to {lang} as a word-by-word gloss. "
                "Reply as JSON {translation, conf, notes}. Mark uncertain words with '?'. "
                "Never echo the Greek as the translation.")
    return (f"Translate ancient Greek to concise fluent {lang}. "
            "Reply as JSON {translation, conf, notes}. Mark uncertain spans with '?'. "
            "Never echo the Greek as the translation.")


def _parse_live(txt):
    import json

    try:
        obj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        return str(obj.get("translation", txt)).strip(), float(obj.get("conf", 0.0))
    except (ValueError, TypeError, AttributeError):
        return txt.strip(), 0.0


def translate(text, target="en", mode="fluent", timeout=120.0, harness=None):
    """Translate one Greek string. Returns a result dict (never raises on API shape)."""
    _check_args(text, target, mode)
    if spark.offline():
        en, conf = _offline_en(text.strip(), mode)
        if target == "en":
            translation = en
        else:
            translation = f"[{target}-STUB] {en}"
        return {
            "source": SOURCE_LANG,
            "input": text.strip(),
            "target": target,
            "mode": mode,
            "translation": translation,
            "conf": conf,
            "offline": True,
            "harness": getattr(harness, "name", "offline-stub"),
        }
    sys = _system_prompt(target, mode, harness)
    txt = spark.call_spark([
        {"role": "system", "content": sys},
        {"role": "user", "content": text.strip()},
    ], timeout=timeout)
    translation, conf = _parse_live(txt)
    return {
        "source": SOURCE_LANG,
        "input": text.strip(),
        "target": target,
        "mode": mode,
        "translation": translation,
        "conf": conf,
        "offline": False,
        "harness": getattr(harness, "name", "default"),
    }


def translate_readings(readings, target="en", mode="fluent", timeout=120.0, harness=None):
    """Batch-translate sheet readings [{text, conf}]. Keeps reading conf alongside."""
    _check_args("x", target, mode)
    out = []
    for r in readings:
        text = r.get("text", "")
        if not text or text.startswith(STUB_TAG):
            out.append({**r, "input": text, "target": target, "mode": mode,
                        "translation": f"{STUB_TAG} skipped (no reading)",
                        "tconf": 0.0, "offline": spark.offline()})
            continue
        try:
            t = translate(text, target=target, mode=mode, timeout=timeout, harness=harness)
        except ValueError:
            out.append({**r, "input": text, "target": target, "mode": mode,
                        "translation": f"{STUB_TAG} skipped (empty)", "tconf": 0.0,
                        "offline": spark.offline()})
            continue
        out.append({**r, "input": text, "target": target, "mode": mode,
                    "translation": t["translation"], "tconf": t["conf"],
                    "offline": t["offline"]})
    return out


def offline_available():
    return True  # stubs always work; live needs key + network


if __name__ == "__main__":
    import json as _j
    import sys as _s
    os.environ.setdefault("UNROLL_OFFLINE", "1")
    _a = _s.argv[1:] or ["χαῖρε κόσμε"]
    print(_j.dumps([translate(t) for t in _a], ensure_ascii=False, indent=1))
