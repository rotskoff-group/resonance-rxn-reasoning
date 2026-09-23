#!/usr/bin/env python3
"""Generate the self-correction conditions for one checkpoint.

WHY TURN 1 IS REBUILT FROM THE ARCHIVED RECORD, not re-derived from the parser.
Every archived results.jsonl already stores the exact `system_prompt` and `task_prompt`
that produced its response. Reconstructing the message list from those two strings makes
the turn-1 context byte-identical to the run whose response we are reviewing -- which is
the whole basis of the paired comparison. Re-deriving through
prepare_data_samples/build_messages would re-introduce the two prompt traps this project
has already been bitten by (`prompt_kwargs` is silently ignored, and S1 was trained on a
bespoke prompt that scores 45.76% vs 29.52% under `rules`). The shape emitted here is
exactly build_messages' shape: [system?] + [user].

Conditions, kinds and wording live in conditions.py. Scoring is score_selfcorrect.py.

  python scripts/selfcorrect_runner.py \
      --model-tag q8_s3 --data data/eval_200_pilot.json --conditions Cn_restate,C1_review_full
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parent / "frozen"))
import conditions as C
from loguru import logger
from vllm import LLM, SamplingParams

RUN = Path(__file__).resolve().parents[1]


def turn1_messages(t1):
    msgs = []
    if t1.get("system_prompt"):
        msgs.append({"role": "system", "content": t1["system_prompt"]})
    msgs.append({"role": "user", "content": t1["task_prompt"]})
    return msgs


def strip_answer_directive(task_prompt):
    """Remove the 'Return the answer between <answer>...' line for the verify turn.

    Leaving it in asks the model for a product and a True/False verdict in the same
    breath; confidence.py in the 30B arm strips it for the same reason.
    """
    keep = [ln for ln in task_prompt.splitlines()
            if "between <answer>" not in ln and "</answer> tags" not in ln]
    return "\n".join(keep).rstrip()


def over_budget(tok, conv, max_model_len, max_tokens):
    """True if this conversation cannot fit its own continuation.

    MEASURED (2026-09-10): ~1.0-1.7% of archived S1/S2/S3 turn-1 responses -- and 16.5%
    of q8_base's -- are runaway generations that hit the original 16,384-token cap, so
    replaying one as an assistant turn overflows a 16k context and vLLM raises rather
    than truncating. Those are exactly the records whose `<think>` never closed. 32k
    covers 100% of items for every 8B checkpoint, but the guard stays: an over-budget
    prompt must degrade to a RECORDED skip, never a silent truncation (which would edit
    the very trace under review) and never a crash 3 hours in. Follows the same rule as
    force_answers() in the 30B arm.
    """
    n = len(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True))
    return n + max_tokens > max_model_len, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    # Explicit paths let this run where the local models.py registry does not exist
    # (e.g. Modal, where the 30B weights live on a volume under /outputs/run_thinking).
    ap.add_argument("--model-path", default="", help="bypass models.py with an explicit path")
    ap.add_argument("--lora-path", default="", help="adapter path; implies enable_lora")
    ap.add_argument("--think-prefilled", choices=["auto", "yes", "no"], default="auto",
                    help="does the chat template prefill <think>? Qwen3-*-Thinking-2507 "
                         "does (0/5000 responses contain the tag); Qwen3-8B does not. "
                         "Only the forced-verify suffix depends on this.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--turn1-tag", default="", help="which archived turn-1 to review (default: --model-tag)")
    ap.add_argument("--conditions", default=",".join(C.CONDITIONS))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--k-fresh", type=int, default=8, help="samples for C5_fresh")
    # Matching C5_fresh to turn 1's own decode makes the resampling null EXACT. The 30B
    # benches all ran T=0.6/top_p=0.95/top_k=20, so pass those here; the 8B turn-1 is
    # greedy and cannot be matched by any sampler, which is why that arm's null is only
    # approximate (and why the sample-0 variant exists).
    ap.add_argument("--fresh-temperature", type=float, default=1.0)
    ap.add_argument("--fresh-top-p", type=float, default=1.0)
    ap.add_argument("--fresh-top-k", type=int, default=-1)
    # Turn-2 and modified-turn-1 sampling. The 0910 arm hardcoded these greedy, which is
    # the single defect this fork exists to fix: a GREEDY turn 2 scored against a SAMPLED
    # resampling null is what inverted that arm's headline conclusions (the null's net went
    # -10.06 pp mismatched vs +2.45 pp matched, i.e. the wrong SIGN, not merely the wrong
    # size). Default to Qwen3-Thinking's recommended decode, which is also what every
    # archived 30B turn-1 used, so turn 1 and turn 2 are drawn from the same sampler.
    ap.add_argument("--turn2-temperature", type=float, default=0.6)
    ap.add_argument("--turn2-top-p", type=float, default=0.95)
    ap.add_argument("--turn2-top-k", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=1,
                    help="samples per item for non-C5 conditions; >1 bounds how much of a "
                         "transition matrix is decode noise rather than intervention effect")
    # 4096 truncated 807/2400 of t30_armA's SHORTEST condition in the 0910 arm. Measured
    # 2026-09-12 the 8B turn-2 cost of that cap is small (0.16-1.60% at cap), but 8192 is
    # free on the collapsed arms, so there is no reason to keep the risk.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-mode", choices=["per_reaction", "shared"], default="per_reaction",
                    help="per_reaction: seed + eval_idx per request; shared: the frozen behaviour")
    ap.add_argument("--max-lora-rank", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-mode", choices=["forced", "think"], default="forced",
                    help="forced: close </think> in id space and read the next token's "
                         "True/False logprobs (needed -- these models always open a think "
                         "block, so an unforced first token is '<think>' every time). "
                         "think: let it reason and parse the verdict from text.")
    args = ap.parse_args()

    if args.model_path:
        class _Spec:
            path = args.model_path
            lora_path = args.lora_path or None
            think_is_prefilled = (args.think_prefilled == "yes")
        spec = _Spec()
        if args.think_prefilled == "auto":
            raise SystemExit("--think-prefilled must be yes/no when --model-path is given: "
                             "the forced-verify suffix differs per model family and "
                             "guessing it silently produces a malformed prompt")
    else:
        import models as M
        spec = M.get(args.model_tag)
    turn1_tag = args.turn1_tag or args.model_tag
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in C.CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; known: {sorted(C.CONDITIONS)}")

    records = json.load(open(args.data))
    if args.limit:
        records = records[: args.limit]
    records = [r for r in records if turn1_tag in r["turn1"]]
    if not records:
        raise SystemExit(f"no records carry an archived turn-1 for {turn1_tag!r}")
    logger.info(f"{len(records)} items | model {args.model_tag} | turn1 {turn1_tag} | {conds}")

    out_dir = Path(args.out_dir or (RUN / "outputs" / args.model_tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_kwargs = dict(model=spec.path, tensor_parallel_size=args.tp, dtype="bfloat16",
                      max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                      seed=args.seed, enable_prefix_caching=True)
    lora_req = None
    if spec.lora_path:
        # arm B was never merged; it is base + adapter. r=128 must be declared up front.
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        lora_req = LoRARequest("arm", 1, spec.lora_path)
        logger.info(f"LoRA: {spec.lora_path} over base {spec.path}")
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    def chat(convs, sp):
        kw = {"lora_request": lora_req} if lora_req else {}
        return llm.chat(convs, sampling_params=sp, **kw)

    def per_reaction(sp, keep, base):
        """One SamplingParams per prompt, seeded by reaction: the frozen runner gave every
        request the same seed, so draw j shared one random stream across all reactions."""
        if args.seed_mode == "shared":
            return sp
        out = []
        for i, _ in keep:
            p = sp.clone()
            p.seed = args.seed + int(base[i][0]["eval_idx"])
            out.append(p)
        return out

    # One sampler for every generated turn, recorded into each record's `decode` key so a
    # table can always state what produced it. Greedy remains reachable with
    # --turn2-temperature 0 (top_p/top_k are ignored by vLLM at T=0).
    turn2_decode = {"temperature": args.turn2_temperature,
                    "top_p": args.turn2_top_p, "top_k": args.turn2_top_k}
    fresh_decode = {"temperature": args.fresh_temperature,
                    "top_p": args.fresh_top_p, "top_k": args.fresh_top_k}

    def _sp(decode, n=1):
        return SamplingParams(n=n, temperature=decode["temperature"],
                              top_p=decode["top_p"], top_k=decode["top_k"],
                              max_tokens=args.max_tokens, seed=args.seed)

    standard = _sp(turn2_decode, n=args.rollouts)
    logger.info(f"decode: turn2/turn1 {turn2_decode} x{args.rollouts} | "
                f"C5_fresh {fresh_decode} x{args.k_fresh} | max_tokens {args.max_tokens}")

    for cond in conds:
        kind, needs_turn1, desc = C.CONDITIONS[cond]
        path = out_dir / f"{cond}.jsonl"
        logger.info(f"=== {cond} ({desc}) -> {path}")

        base = [(r, r["turn1"][turn1_tag]) for r in records]

        if kind == "turn2":
            convs = [C.build_turn2(turn1_messages(t1), t1["response"], cond) for _, t1 in base]
            keep, skipped = _split_budget(tok, base, convs, args)
            outs = chat([c for _, c in keep], per_reaction(standard, keep, base)) if keep else []
            rows = [(base[i][0], base[i][1], list(o.outputs), None)
                    for (i, _), o in zip(keep, outs)]
            rows += skipped

        elif kind == "turn1":
            convs = []
            for _, t1 in base:
                m = turn1_messages(t1)
                # TURN1_SUFFIX generalises what used to be an `if cond == "C6_longer"`
                # branch. C6's text is unchanged, so a re-run stays byte-comparable with
                # outputs_greedy/full5k_*/C6_longer.jsonl; verify that before trusting it.
                sfx = C.turn1_suffix(cond)
                if sfx:
                    m = m[:-1] + [{"role": "user", "content": m[-1]["content"] + sfx}]
                convs.append(m)
            sp = _sp(fresh_decode, n=args.k_fresh) if cond == "C5_fresh" else standard
            keep, skipped = _split_budget(tok, base, convs, args)
            outs = chat([c for _, c in keep], per_reaction(sp, keep, base)) if keep else []
            rows = [(base[i][0], base[i][1], list(o.outputs), None)
                    for (i, _), o in zip(keep, outs)]
            rows += skipped

        elif kind == "verify":
            # True/False logprob, not a sampled word: a one-token sample throws away the
            # margin, and the margin is what an AUROC needs.
            #
            # MEASURED (2026-09-10): asking the question through llm.chat() returns
            # `<think>` as the first token for 198/198 items -- these checkpoints always
            # open a reasoning block, so the verdict never lands at position 0 and every
            # p_true came back None. The fix is confidence.py's `allow_think=False`: close
            # the think block in TOKEN-ID space and prime "Answer:", then read the next
            # token's logprobs. Off-distribution by construction, which is why the flag
            # exists and why the verdict is also parsed from text in the `think` mode.
            from resonance.evaluation.extractors import TASK_PARSERS
            extract, _ = TASK_PARSERS["forward_rxn"]
            convs, preds = [], []
            verify_pos = {r["eval_idx"]: j for j, (r, _) in enumerate(base)}
            for _, t1 in base:
                pred = extract(t1["response"]) or "(no parseable product)"
                preds.append(pred)
                user = strip_answer_directive(t1["task_prompt"]) + \
                    C.VERIFY_TRUE_FALSE.format(pred=pred)
                m = []
                if t1.get("system_prompt"):
                    m.append({"role": "system", "content": t1["system_prompt"]})
                m.append({"role": "user", "content": user})
                convs.append(m)
            keep, skipped = _split_budget(tok, base, convs, args)
            if args.verify_mode == "forced":
                from vllm.inputs import TokensPrompt
                suffix = ("\n\n</think>\n\nAnswer:" if spec.think_is_prefilled
                          else "<think>\n\n</think>\n\nAnswer:")
                sfx_ids = tok.encode(suffix, add_special_tokens=False)
                sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
                prompts = []
                for _, conv in keep:
                    ids = tok.apply_chat_template(conv, tokenize=True,
                                                  add_generation_prompt=True)
                    prompts.append(TokensPrompt(prompt_token_ids=list(ids) + sfx_ids))
                gkw = {"lora_request": lora_req} if lora_req else {}
                outs = llm.generate(prompts, sp, **gkw) if prompts else []
            else:
                # let it reason, then read the verdict out of the text
                sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                    logprobs=20)
                outs = chat([c for _, c in keep], sp) if keep else []
            rows = [(base[i][0], base[i][1], [o.outputs[0]], None)
                    for (i, _), o in zip(keep, outs)]
            rows += skipped

        rows.sort(key=lambda t: t[0]["eval_idx"])
        n = 0
        with open(path, "w") as f:
            for r, t1, outputs, skip_reason in rows:
                i = r["eval_idx"]
                rec = {
                    # keys the shared scorer needs, byte-compatible with workflows.benchmark
                    "idx": r["source_idx"],
                    "system_prompt": t1.get("system_prompt"),
                    "task_prompt": t1.get("task_prompt"),
                    "response": ("" if skip_reason else outputs[0].text),
                    "answer": t1.get("answer"),
                    "answer_type": t1.get("answer_type"),
                    "question_type": t1.get("question_type"),
                    # experiment bookkeeping
                    "eval_idx": r["eval_idx"],
                    "probe_idx": r["probe_idx"],
                    "reaction_class": r["reaction_class"],
                    "condition": cond,
                    "model_tag": args.model_tag,
                    "turn1_tag": turn1_tag,
                    "turn1_prompt_variant": t1.get("prompt_variant"),
                    # The turn-1 suffix is APPENDED to task_prompt when the conversation is
                    # built, but task_prompt above is recorded bare. Record the suffix so a
                    # record says which prompt produced it -- Branch 2's whole claim is that
                    # a prompt changed behaviour, and that is unauditable otherwise.
                    # task_prompt itself stays canonical: score_selfcorrect's copied_sm
                    # regex reads it, and downstream code expects the unmodified text.
                    "turn1_suffix": C.turn1_suffix(cond) or None,
                    "turn1_response": t1["response"],
                    "n_output_tokens": (0 if skip_reason else len(outputs[0].token_ids)),
                    # Needed by the scorer: a prefilled family emits no literal <think>, so
                    # "is this a truncated think block?" cannot be answered from the text.
                    "think_prefilled": spec.think_is_prefilled,
                    "max_tokens": args.max_tokens,
                    "seed_mode": args.seed_mode,
                    "decode": (fresh_decode if cond == "C5_fresh"
                               else {"temperature": 0.0} if kind == "verify"
                               else turn2_decode),
                    "n_rollouts": (args.k_fresh if cond == "C5_fresh"
                                   else 1 if kind == "verify" else args.rollouts),
                    "skipped": skip_reason,
                }
                if not skip_reason and len(outputs) > 1:
                    # C5_fresh's K draws, and --rollouts R for any other condition. The
                    # reported `response` is always outputs[0]; the rest bound decode noise.
                    rec["responses"] = [o.text for o in outputs]
                    rec["n_output_tokens_all"] = [len(o.token_ids) for o in outputs]
                if kind == "verify":
                    rec["verify_pred"] = None if skip_reason else preds[verify_pos[i]]
                    rec["verify_logprobs"] = (None if skip_reason
                                              else _tf_logprobs(outputs[0], tok))
                    rec["verify_mode"] = args.verify_mode
                    rec["verify_verdict_text"] = (None if skip_reason
                                                  else _text_verdict(outputs[0].text))
                f.write(json.dumps(rec) + "\n")
                n += 1
        logger.info(f"wrote {path} ({n} records)")


def _split_budget(tok, base, convs, args):
    """-> ([(base_index, conv)], [skipped_rows]) with skipped rows fully formed."""
    keep, skipped = [], []
    for i, conv in enumerate(convs):
        bad, n = over_budget(tok, conv, args.max_model_len, args.max_tokens)
        if bad:
            skipped.append((base[i][0], base[i][1], None,
                            f"prompt {n} tok + {args.max_tokens} out > {args.max_model_len}"))
        else:
            keep.append((i, conv))
    if skipped:
        logger.warning(f"  {len(skipped)}/{len(convs)} items over budget -> recorded as skipped")
    return keep, skipped


def _text_verdict(text):
    """Last standalone True/False in the completion, or None."""
    import re
    m = re.findall(r"\b(true|false)\b", text.lower())
    return {"true": True, "false": False}[m[-1]] if m else None


def _tf_logprobs(out, tok):
    """P(True) vs P(False) over the first generated token, whitespace/case variants pooled."""
    import math
    lp = (out.logprobs or [None])[0]
    if not lp:
        return None
    acc = {"true": None, "false": None}
    for tid, info in lp.items():
        s = (getattr(info, "decoded_token", None) or tok.decode([tid])).strip().lower()
        if s in acc:
            v = getattr(info, "logprob", None)
            if v is not None:
                acc[s] = v if acc[s] is None else max(acc[s], v)
    t, f = acc["true"], acc["false"]
    if t is None and f is None:
        return {"p_true": None, "logprob_true": None, "logprob_false": None}
    et = math.exp(t) if t is not None else 0.0
    ef = math.exp(f) if f is not None else 0.0
    return {"p_true": (et / (et + ef)) if (et + ef) > 0 else None,
            "logprob_true": t, "logprob_false": f}


if __name__ == "__main__":
    main()
