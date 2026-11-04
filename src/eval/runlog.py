"""Run IDs + per-slice JSONL log. Stdlib only.

Schema per line: {run_id, z, status, curve_ref, checks, reprompts}
status: ok | reprompted | suspect | failed
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    def __init__(self, root, run_id=None, note=""):
        self.root = Path(root)
        self.run_id = run_id or self.new_id()
        self.path = self.root / "reports" / f"run_{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if note or not self.path.exists():
            self._header(note)

    @staticmethod
    def new_id():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{ts}_{uuid.uuid4().hex[:6]}"

    def _header(self, note):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": self.run_id, "note": note,
                                "type": "header"}) + "\n")

    def log_slice(self, z, status, curve_ref=None, checks=None,
                  reprompts=0):
        assert status in ("ok", "reprompted", "suspect", "failed"), status
        rec = {"run_id": self.run_id, "z": int(z), "status": status,
               "curve_ref": curve_ref, "checks": checks or {},
               "reprompts": int(reprompts)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def read(self):
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
