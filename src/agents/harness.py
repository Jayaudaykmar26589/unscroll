"""Harness2-inspired improvement loop for translation prompts.

Adapted from google-research/harness2 (Apache 2.0): task-level
Propose-Probe-Compose without ground truth, plus a persistent
improvement harness (evidence-backed lessons in lessons.jsonl).

We use 4 of harness2's 8 editable components, renamed for this product:
system prompt, guardrails, memory (lessons), tool guidance. Model weights
stay fixed; only the harness text changes.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.agents import translate as T

ROOT = Path(__file__).resolve().parents[2]
LESSONS_PATH = ROOT / "src" / "agents" / "lessons.jsonl"
MAX_LESSONS = 20


@dataclass
class Harness:
    name: str
    system_prompt: str
    guardrails: str = ""
    memory: list = field(default_factory=list)
    tool_guidance: str = ""

    def to_system(self, target, mode):
        lang = T.SUPPORTED_TARGETS[target]
        parts = [self.system_prompt.format(lang=lang, mode=mode)]
        if self.guardrails:
            parts.append("Guardrails: " + self.guardrails)
        if self.tool_guidance:
            parts.append("Guidance: " + self.tool_guidance)
        if self.memory:
            tips = "; ".join(self.memory[-5:])
            parts.append("Lessons: " + tips)
        return "\n".join(parts)


DEFAULT_HARNESS = Harness(
    name="default-v1",
    system_prompt="Translate ancient Greek to {lang} ({mode}). Reply as JSON {{translation, conf, notes}}.",
    guardrails="Never echo the Greek as the translation. Mark uncertain spans with '?'.",
    tool_guidance="One JSON object only, no prose outside it.",
)


def load_lessons(path=LESSONS_PATH):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-MAX_LESSONS:]


def baseline_with_memory(path=LESSONS_PATH):
    h = Harness(**{**DEFAULT_HARNESS.__dict__, "memory": []})
    for rec in load_lessons(path):
        lesson = rec.get("lesson")
        if lesson:
            h.memory.append(lesson)
    h.memory = h.memory[-5:]
    return h


def propose_invariant_edit(h):
    """Generic translation-quality edit (task-agnostic)."""
    return Harness(
        name=h.name + "+invariant",
        system_prompt=h.system_prompt,
        guardrails=(h.guardrails + " translation must differ from input; conf in 0-1; "
                    "empty translation is a failure.").strip(),
        memory=list(h.memory),
        tool_guidance=(h.tool_guidance + " If unsure, lower conf instead of echoing.").strip(),
    )


def propose_domain_edit(h, task_text):
    """Papyrus-domain edit (ancient Greek fragment specifics)."""
    _ = task_text
    return Harness(
        name=h.name + "+domain",
        system_prompt=h.system_prompt,
        guardrails=(h.guardrails + " Preserve fragment marks ([..], '?'); "
                    "Coptic/Greek block input stays Greek in notes, never in translation.").strip(),
        memory=list(h.memory),
        tool_guidance=(h.tool_guidance + " Gloss unknown words as WORD = ? rather than dropping them.").strip(),
    )


def score_translation(result, input_text):
    """No-ground-truth behavioral checks (0-1 total). Higher is better."""
    tr = (result.get("translation") or "").strip()
    src = (input_text or "").strip()
    checks = {}
    checks["non_empty"] = bool(tr)
    checks["not_echo"] = bool(tr and tr != src)
    try:
        c = float(result.get("conf", 0.0))
        checks["has_conf"] = 0.0 <= c <= 1.0 and c > 0.0
    except (TypeError, ValueError):
        checks["has_conf"] = False
    if "?" in src or "[..]" in src:
        checks["marks_uncertain"] = ("?" in tr) or ("[..]" in tr)
    else:
        checks["marks_uncertain"] = True
    ratio = (len(tr) / len(src)) if src else 0.0
    checks["length_sane"] = 0.2 <= ratio <= 5.0
    weights = {"non_empty": 0.3, "not_echo": 0.3, "has_conf": 0.15,
               "marks_uncertain": 0.1, "length_sane": 0.15}
    total = round(sum(weights[k] for k, v in checks.items() if v), 3)
    return {"total": total, "checks": checks}


def probe(harness, text, target="en", mode="fluent"):
    result = T.translate(text, target=target, mode=mode, harness=harness)
    score = score_translation(result, text)
    return result, score


def compose(scored):
    """scored: list of (harness, result, score). Returns winner + trace rows."""
    ranked = sorted(scored, key=lambda t: (-t[2]["total"], t[0].name))
    winner = ranked[0]
    trace = [{"harness": h.name, "score": s["total"], "checks": s["checks"],
              "translation": r["translation"][:160]} for h, r, s in ranked]
    return winner, trace


def reflect(task_text, winner, base_score, path=LESSONS_PATH, trace=None):
    """Append one evidence-backed lesson (harnessing-level recursion)."""
    delta = round(winner[2]["total"] - base_score, 3)
    if delta <= 0:
        return None
    rec = {
        "task_hash": hashlib.sha256(task_text.encode("utf-8")).hexdigest()[:12],
        "winner": winner[0].name,
        "score": winner[2]["total"],
        "delta": delta,
        "lesson": f"Prefer {winner[0].name.split('+')[-1]} guardrails for fragment translation.",
        "evidence": {"checks": winner[2]["checks"], "translation": winner[1]["translation"][:160]},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def improve(text, target="en", mode="fluent", budget=1, lessons_path=LESSONS_PATH):
    """Task-level recursion: baseline -> propose 2 edits -> probe -> compose.

    budget = recursion rounds (each round probes 2 variants). Bounded to <=3
    to cap live API cost; offline runs are free and deterministic.
    """
    if budget < 1 or budget > 3:
        raise ValueError("budget must be 1..3")
    base = baseline_with_memory(lessons_path)
    base_result, base_score = probe(base, text, target, mode)
    current = base
    full_trace = [{"round": 0, "harness": base.name, "score": base_score["total"]}]
    winner = (base, base_result, base_score)
    for rnd in range(1, budget + 1):
        variants = [propose_invariant_edit(current), propose_domain_edit(current, text)]
        scored = [(current, *probe(current, text, target, mode))]
        for v in variants:
            scored.append((v, *probe(v, text, target, mode)))
        winner, trace = compose(scored)
        for row in trace:
            row["round"] = rnd
        full_trace.extend(trace)
        current = winner[0]
    lesson = reflect(text, winner, base_score["total"], path=lessons_path)
    return {"result": winner[1], "harness": winner[0].name,
            "base_score": base_score["total"], "score": winner[2]["total"],
            "trace": full_trace, "lesson": lesson}


if __name__ == "__main__":
    import sys as _s
    os.environ.setdefault("UNROLL_OFFLINE", "1")
    print(json.dumps(improve(_s.argv[1] if len(_s.argv) > 1 else "χαῖρε κόσμε"),
                     ensure_ascii=False, indent=1))
