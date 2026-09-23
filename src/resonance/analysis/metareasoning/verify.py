"""Check the frozen scripts against MANIFEST.json and, optionally, a run's protocol record.

    python -m resonance.analysis.metareasoning.verify
    python -m resonance.analysis.metareasoning.verify --protocol <run>/outputs/protocol.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parent / "frozen"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", type=Path, help="run protocol JSON mapping script path -> sha256")
    args = ap.parse_args(argv)
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text())["files"]
    bad = [name for name, entry in manifest.items() if sha256(FROZEN / name) != entry["sha256"]]
    for name in bad:
        print(f"MODIFIED {name}: differs from MANIFEST.json", file=sys.stderr)
    if args.protocol:
        record = json.loads(args.protocol.read_text())
        record = record.get("protocol", record)
        recorded = {Path(k).name: v for k, v in record.items() if isinstance(v, str)}
        for name, entry in manifest.items():
            if name in recorded and recorded[name] != entry["sha256"]:
                print(f"DIFFERS {name}: not the version recorded in {args.protocol}", file=sys.stderr)
                bad.append(name)
    print("ok" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
