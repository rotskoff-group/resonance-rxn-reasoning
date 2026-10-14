#!/usr/bin/env python3
"""Stage 1a -- generate retrospective reaction traces with Kimi K3.

Prompts are derived from an earlier teacher's prompts with only the reaction ID removed.
The teacher still sees the product and writes to the 1.-5. schema. The published corpus
uses these traces as-is (no reformatting pass).

Both message fields are stored:
    trace     <- message.content            (the 1:1 analogue of the Sonnet `trace` field)
    reasoning <- message.reasoning_content  (K3's raw CoT; Sonnet has no equivalent)
S1 trains on `trace`. `reasoning` is kept for analysis and is NOT trained on -- doing
so would make this a different experiment rather than a teacher swap.

K3 pins temperature/top_p/n and 400s on overrides, so no sampling args are exposed.

    export KIMI_MODAL_API_BASE=...      # or pass --api-base
    python -m resonance.pipelines.traces.generate_traces_kimi --limit 200   # pilot
    python -m resonance.pipelines.traces.generate_traces_kimi               # full run (resumable)

Ported from run_dir('traces')/scripts/generate_traces_kimi.py.
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from openai import APIError, AsyncOpenAI, RateLimitError

from resonance.paths import run_dir
from resonance.pipelines.traces.endpoints import EndpointError, describe, resolve
from resonance.pipelines.traces.prompts_stage1 import FINGERPRINT, SYSTEM_PROMPT, USER_TEMPLATE

RUN       = run_dir("traces")
DATA_FILE = run_dir("uspto") / "train.json"
OUT_FILE  = RUN / "outputs/traces_stage1a_no_reaction_id.jsonl"
LABELS    = run_dir("uspto_classes") / "outputs/uspto_labels_train.json"

MAX_TOKENS  = 8192   # must cover reasoning_content + content together
CONCURRENCY = 16
DEFAULT_ROUTE = "modal"   # the only route measured to sustain a 40k run; see endpoints.py


def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_labels(path: Path) -> dict:
    """row_idx (str) -> {'level1': ..., 'reaction_id': ...}; empty dict if absent."""
    if not path.exists():
        print(f"WARNING: {path} missing; level1_class will be null", flush=True)
        return {}
    return json.load(open(path)).get("labels", {})


def load_done_indices(path: Path) -> set[int]:
    """row_idx values already written, so a resubmit resumes instead of duplicating."""
    done: set[int] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("prompt_fingerprint") != FINGERPRINT:
                raise ValueError(f"{path}: incompatible or missing prompt fingerprint; use a fresh output file")
            if "row_idx" in rec:
                done.add(rec["row_idx"])
    return done


async def process_one(client, sem, row_idx, record, labels, out_f, lock, stats,
                      model, reasoning_effort, max_tokens) -> None:
    reaction_id = record["metadata"]["reaction_id"]
    smiles      = record["original_smiles"]

    # Field defaults are load-bearing: they become literal training text, and the Sonnet arm
    # used exactly these strings.
    starting_material = smiles.get("Starting Material", "") or "N/A"
    reagents          = smiles.get("Reagents", "")          or "None specified"
    conditions        = smiles.get("Reaction Conditions", "") or "None specified"
    product           = smiles.get("End Product", "")        or "N/A"

    prompt = USER_TEMPLATE.format(
        starting_material=starting_material,
        reagents=reagents,
        conditions=conditions,
        product=product,
    )

    lab = labels.get(str(row_idx)) or {}

    async with sem:
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                )
                msg = completion.choices[0].message
                trace     = (msg.content or "").strip()
                reasoning = (getattr(msg, "reasoning_content", None) or "").strip()

                if not trace:
                    # Budget consumed inside the thinking block: no visible answer. Count it
                    # rather than writing an unusable record.
                    async with lock:
                        stats["empty"] += 1
                    return

                usage = completion.usage
                rec = {
                    "prompt_fingerprint": FINGERPRINT,
                    "reaction_id":       reaction_id,
                    "starting_material": starting_material,
                    "reagents":          reagents,
                    "conditions":        conditions,
                    "product":           product,
                    "trace":             trace,
                    "reasoning":         reasoning,
                    "model":             model,
                    "reasoning_effort":  reasoning_effort,
                    "usage": {
                        "input_tokens":  getattr(usage, "prompt_tokens", None),
                        "output_tokens": getattr(usage, "completion_tokens", None),
                    },
                    "row_idx":      row_idx,
                    "level1_class": lab.get("level1"),
                }
                async with lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    stats["done"] += 1
                    if stats["done"] % 100 == 0:
                        elapsed = time.time() - stats["t0"]
                        rate = stats["done"] / elapsed if elapsed else 0.0
                        print(f"  {stats['done']:,} written  ({rate:.2f}/s, "
                              f"{stats['failed']} failed, {stats['empty']} empty)", flush=True)
                return

            except (RateLimitError, APIError) as e:
                if attempt == max_attempts - 1:
                    async with lock:
                        stats["failed"] += 1
                    print(f"  row {row_idx} FAILED after {max_attempts} attempts: {e}", flush=True)
                    return
                delay = min(60.0, 1.0 * (2 ** attempt)) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            except Exception as e:  # noqa: BLE001 - one bad row must not kill the run
                async with lock:
                    stats["failed"] += 1
                print(f"  row {row_idx} FAILED (unexpected): {type(e).__name__}: {e}", flush=True)
                return


async def run(args) -> int:
    try:
        model, api_base, api_key = resolve(args.route, args.api_base, args.model)
    except EndpointError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    data     = load_data(Path(args.data_file))
    labels   = load_labels(Path(args.labels))
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_indices(out_file)
    todo = [i for i in range(len(data)) if i not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Total reactions : {len(data):,}")
    print(f"Already done    : {len(done):,}")
    print(f"To generate     : {len(todo):,}")
    print(f"Endpoint        : {describe(args.route, api_base, model)}")
    print(f"reasoning_effort: {args.reasoning_effort}")
    print(f"Concurrency     : {args.concurrency}")
    print(f"Output          : {out_file}", flush=True)

    if not todo:
        print("Nothing to do.")
        return 0

    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=1800.0)
    sem    = asyncio.Semaphore(args.concurrency)
    lock   = asyncio.Lock()
    stats  = {"done": 0, "failed": 0, "empty": 0, "t0": time.time()}

    with open(out_file, "a") as out_f:
        await asyncio.gather(*[
            process_one(client, sem, i, data[i], labels, out_f, lock, stats,
                        model, args.reasoning_effort, args.max_tokens)
            for i in todo
        ])

    elapsed = time.time() - stats["t0"]
    rate = stats["done"] / elapsed if elapsed else 0.0
    print(f"\n=== Done in {elapsed / 60:.1f} min ===")
    print(f"  written : {stats['done']:,}")
    print(f"  failed  : {stats['failed']:,}")
    print(f"  empty   : {stats['empty']:,}")
    print(f"  rate    : {rate * 3600:.0f}/hr")
    if rate:
        remaining = len(data) - len(done) - stats["done"]
        print(f"  projection for the remaining {remaining:,}: {remaining / rate / 3600:.1f} hr")
    print(f"  output  : {out_file}", flush=True)
    return 1 if (stats["done"] == 0 and todo) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file",   default=str(DATA_FILE))
    ap.add_argument("--labels",      default=str(LABELS))
    ap.add_argument("--output",      default=str(OUT_FILE))
    ap.add_argument("--route", default=DEFAULT_ROUTE,
                    choices=["modal", "native", "openrouter"],
                    help="modal = self-hosted endpoint, no shared-tenant throttle (default).")
    ap.add_argument("--model",    default=None, help="Override the route's default model id.")
    ap.add_argument("--api-base", default=None, help="Override the route's default base URL.")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--max-tokens",  type=int, default=MAX_TOKENS)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "high", "max"],
                    help="K3 thinking budget. Dominant cost knob; output bills at $15/M.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only generate this many NEW records (pilot / smoke tests).")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
