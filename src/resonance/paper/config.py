"""Load a figure config with its data paths resolved.

Configs under `configs/paper/figures/` spell paths as `${run_dir:<key>}/...` (a
`resonance.paths.RUN_DIRS` key) or `${DATA_ROOT}/...`. `load` substitutes them, so the
same config works wherever `RESONANCE_DATA_ROOT` points.
"""
import json
import re
from pathlib import Path

from resonance.paths import DATA_ROOT, run_dir

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs/paper/figures"
_TOKEN = re.compile(r"\$\{(run_dir:([A-Za-z0-9_]+)|DATA_ROOT)\}")


def _resolve(value):
    if isinstance(value, str):
        return _TOKEN.sub(lambda m: str(run_dir(m.group(2)) if m.group(2) else DATA_ROOT), value)
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    return value


def load(path):
    """Config at `path` (or a file name in CONFIG_DIR) with every token resolved."""
    path = Path(path)
    if not path.exists():
        path = CONFIG_DIR / path
    return _resolve(json.loads(path.read_text()))


def write_resolved(path, out):
    """Write the resolved config to `out` (for tools that take a config file) and return it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(load(path), indent=2) + "\n")
    return out
