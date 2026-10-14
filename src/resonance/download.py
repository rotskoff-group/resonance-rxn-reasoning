"""Fetch the paper's released data into the layout every script expects.

    python -m resonance.download figures        # inputs of every paper figure (~4 GB)
    python -m resonance.download all            # every released original: figure inputs plus the
                                                # USPTO-50k split, teacher traces, S2 data and S3 pairs
    python -m resonance.download figures --from-dir <dir>   # from a local copy of the dataset repo

Files come from the Hugging Face dataset repo `resonance.release.artifacts.DATASET_REPO`,
under `<PAPER>/raw/<run_dir key>/<path>`. Each is placed at
`$RESONANCE_DATA_ROOT/<run_dir key>/<path>` (`resonance.paths`), the path the original run wrote
it to, so the figure code and every stage script find it with `run_dir(key)`. Existing files
are left alone. `--from-dir` takes the directory that contains `<PAPER>/` (a clone or snapshot
of the dataset repo). The tidy tables under `<PAPER>/<config>/` are for
`datasets.load_dataset`; no script reads them.
"""
import argparse
import shutil
from pathlib import Path

from resonance.paths import DATA_ROOT, RUN_DIRS
from resonance.release.artifacts import DATASET_REPO, PAPER

RAW = f"{PAPER}/raw"
# The files `resonance.paper.run_all` reads, as paths under RAW (recorded from a traced run).
FIGURE_INPUTS = Path(__file__).resolve().parent / "release/figure_inputs.txt"


def wanted(which):
    if which == "all":
        return None
    return [line.strip() for line in FIGURE_INPUTS.read_text().splitlines() if line.strip()]


def fetch(which, from_dir=None, revision=None):
    files = wanted(which)
    if from_dir is None:
        from huggingface_hub import snapshot_download
        patterns = [f"{RAW}/*"] if files is None else [f"{RAW}/{f}" for f in files]
        from_dir = snapshot_download(DATASET_REPO, repo_type="dataset", revision=revision,
                                     allow_patterns=patterns)
    raw = Path(from_dir) / RAW
    if not raw.is_dir():
        raise SystemExit(f"{raw} not found: --from-dir must contain {PAPER}/ (a copy of {DATASET_REPO})")
    sources = (sorted(p for p in raw.rglob("*") if p.is_file()) if files is None
               else [raw / f for f in files])
    missing = [p for p in sources if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} files missing from {raw}, e.g. {missing[0]}")
    placed = skipped = 0
    for src in sources:
        key, *rest = src.relative_to(raw).parts
        if key not in RUN_DIRS:
            raise SystemExit(f"{src}: {key!r} is not a resonance.paths.RUN_DIRS key")
        dest = DATA_ROOT / RUN_DIRS[key] / Path(*rest)
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        placed += 1
    print(f"{placed} files placed under {DATA_ROOT} ({skipped} already present)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("which", choices=["figures", "all"])
    ap.add_argument("--from-dir", type=Path, help="local copy of the dataset repo (the directory containing "
                                                  f"{PAPER}/); skips the download")
    ap.add_argument("--revision", help="dataset repo revision, e.g. the iclr2027-v1 tag")
    args = ap.parse_args(argv)
    fetch(args.which, args.from_dir, args.revision)


if __name__ == "__main__":
    main()
