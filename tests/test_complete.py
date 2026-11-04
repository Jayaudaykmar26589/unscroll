import json

import pytest

from src.agents import complete as C


def test_transliteration_and_folding():
    assert C.transliterate("ta spania pantos") == "τα σπανια παντος"
    assert C.transliterate("ho theos") == "ο θεος"
    assert C.transliterate("θεός") == "θεός"
    # Accents, breathings, underdots, final/lunate sigma and η/ω spellings fold together.
    assert C.key_map("ἤδ̣η τὰ ϲπάνια πάντωϲ̣")[0] == C.key_map("εδε τα σπανια παντος")[0]


@pytest.mark.parametrize("fragment, ref", [
    ("Μήτε νέος τις ὢν μελλέτω", "D.L. 10.122"),            # beginning
    ("ρος ἡμᾶς εἶναι τὸν θάνα", "D.L. 10.124"),              # middle, both edge words cut
    ("tes aphthrsias allotrion", "D.L. 10.123"),              # Latin letters with a typo
    ("ta spania pantos kai", "PHerc. Paris. 4"),              # the scroll passage itself
])
def test_attested_lookup(fragment, ref):
    r = C.complete(fragment, use_model=False)
    assert r["status"] == "attested"
    m = r["matches"][0]
    assert m["ref"].startswith(ref)
    assert m["before"] + m["matched"] + m["after"] == m["text"]


def test_short_fragment_is_not_called_certain():
    r = C.complete("ouk hēdion", use_model=False)
    assert r["status"] in ("too_short", "not_found")


def test_unknown_without_model():
    r = C.complete("ὁ δὲ ἀνεξέταστος βίος", use_model=False)
    assert r["status"] == "not_found" and r["matches"] == []


def test_conjecture_rejects_candidates_that_alter_the_fragment():
    def fake(messages):
        return json.dumps({"candidates": [
            {"sentence": "ὁ δὲ ἀνεξέταστος βίος οὐ βιωτὸς ἀνθρώπῳ.", "source": "Plato, Apology 38a", "en": "x"},
            {"sentence": "ὁ δὲ καλὸς βίος ἡδύς ἐστιν.", "source": None, "en": "y"},       # changed the fragment
        ]})
    r = C.complete("ὁ δὲ ἀνεξέταστος βίος", call=fake)
    assert r["status"] == "conjecture"
    assert [c["source"] for c in r["candidates"]] == ["Plato, Apology 38a"]
    assert r["dropped_for_changing_fragment"] == 1
    assert r["candidates"][0]["after"].startswith(" οὐ βιωτὸς")


def test_bad_input():
    with pytest.raises(ValueError):
        C.complete("   ")
    with pytest.raises(ValueError):
        C.complete("1234 !!")
