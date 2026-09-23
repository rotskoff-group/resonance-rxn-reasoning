"""Reject incomplete, duplicate, or old-prompt corpora before either SFT run.

Ported from run_dir('traces')/scripts/validate_no_id.py.
CLI: python -m resonance.pipelines.traces.validate_no_id TRACES_JSONL
"""
import hashlib
import json
import sys
from pathlib import Path
from resonance.pipelines.traces.prompts_stage1 import FINGERPRINT

def validate(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 40000, f"Expected 40000 records, got {len(rows)}"
    assert {r["row_idx"] for r in rows} == set(range(40000)), "Missing/duplicate row indices"
    for r in rows:
        assert r.get("prompt_fingerprint") == FINGERPRINT, "Wrong generation prompt"
        assert r.get("trace", "").strip() and r.get("product", "").strip(), "Empty trace/product"
    result = {"records": len(rows), "source_sha256": hashlib.sha256(raw).hexdigest(),
              "prompt_fingerprint": FINGERPRINT}
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    validate(sys.argv[1])
