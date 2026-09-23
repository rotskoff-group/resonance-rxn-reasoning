#!/usr/bin/env python3
"""Condition registry: which conditions exist, and the exact wording of each.

Three tables drive everything:
  CONDITIONS         id -> (kind, needs_turn1, description)
                     kind "turn2"  = [system, user, assistant(turn1), user(instruction)]
                     kind "turn1"  = a single fresh/modified first attempt
                     kind "verify" = single turn ending in a True/False logprob question
  TURN2_INSTRUCTION  id -> the follow-up user message
  TURN1_SUFFIX       id -> text appended to a first-turn prompt

Invariants (see README.md "Four measurement rules" for why):
  * C1 and C2 share byte-identical user text; only the assistant turn differs.
  * Every turn-2 instruction restates the <answer> requirement.
  * Cn_restate must be a no-op. It is the G1 gate and the change-rate floor.
  * Feedback conditions (FEEDBACK_VERDICT) assert a verdict for EVERY item, so each is
    truthful on some items and false on the rest. Branch 4 partitions on turn-1
    correctness to recover both halves from one pass.
  * B2_* modify turn 1, not turn 2. Do not pool them with the review conditions.

Wording here is load-bearing: changing a string invalidates comparability with
outputs_greedy/. scripts/test_prompt_compat.py enforces that.
"""

# --------------------------------------------------------------------------------------
# Turn-2 user instructions
# --------------------------------------------------------------------------------------

REVIEW_NEUTRAL = (
    "Review your work above. If you would keep your answer, restate it; if you would "
    "change it, give the revised answer instead. Either way, put your final answer "
    "between <answer> and </answer> tags."
)

ORACLE_WRONG = (
    "Your answer above is incorrect. Work out where the reasoning went wrong and give "
    "the correct product. Put your final answer between <answer> and </answer> tags."
)

# The three negative-feedback wordings differ only in how strong and how socially sourced
# the error claim is; the task text is held fixed.
MILD_DOUBT = (
    "Please double-check the answer above; there may be an issue with it. Put your final "
    "answer between <answer> and </answer> tags."
)

AUTHORITY_WRONG = (
    "An evaluator marked the answer above as incorrect. Revise it accordingly and give "
    "the correct product. Put your final answer between <answer> and </answer> tags."
)

TOLD_CORRECT = (
    "Your answer above is correct. Verify it and return the final answer. Put your final "
    "answer between <answer> and </answer> tags."
)

# Carries the turn-1 reasoning in the USER turn. The Qwen3 chat template drops <think>
# blocks from non-final assistant turns, so a trace placed in the assistant message never
# reaches the model (verified: outputs/prompt_audit_q8_s3.json). This is the only wording
# the trace conditions add; the instruction that follows it is byte-identical to the
# non-trace condition, so the trace is the single difference.
TRACE_PREAMBLE = (
    "For reference, this was your reasoning for that answer:\n\n{trace}\n\n"
)

RESTATE = (
    "Restate your final answer exactly as given above, unchanged. Put it between "
    "<answer> and </answer> tags."
)

VERIFY_TRUE_FALSE = (
    "\n\nA student proposed the following product for this reaction:\n{pred}\n\n"
    "Is the proposed product correct? Reply with exactly one word, True or False."
)

# --------------------------------------------------------------------------------------
# Turn-1 prompt suffixes (appended to the task prompt of a single-turn attempt)
# --------------------------------------------------------------------------------------
#
# LONGER is byte-identical to the 0910 arm's, so a re-run stays comparable with
# outputs_greedy/. test_prompt_compat.py enforces it.
LONGER = (
    "\n\nWork through this carefully and at length: consider the plausible reaction " 
    "pathways before committing, and check your proposed product against the starting "
    "material and reagents."
)

# Step C, DOWN. Says nothing about length: "be concise" would confound backtracking
# frequency with token budget.
COMMIT_FIRST = (
    "\n\nReason directly to the answer. Commit to your first well-supported conclusion "
    "and do not revisit earlier steps or explore alternatives."
)

# Step D, UP. Inside the rollout, before any answer exists -- distinct from C1's post-hoc
# review.
CONSIDER_ALTERNATIVES = (
    "\n\nBefore finalizing, consider at least two plausible product hypotheses and "
    "actively check whether any earlier inference could be wrong. Revise if a better "
    "interpretation is available."
)

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
# condition -> (kind, needs_turn1, description)
#   kind "turn2"    : [system, user, assistant(turn1), user(instruction)]
#   kind "turn1"    : single turn; a fresh or modified first attempt
#   kind "verify"   : single turn ending in a True/False question, scored by logprob
CONDITIONS = {
    # ---- Branch 1: review and correction ----
    "C1_review_full": ("turn2", True,
                       "review own trace + answer"),
    "C2_review_answer_only": ("turn2", True,
                              "review answer only, trace stripped"),
    "C4_oracle_wrong": ("turn2", True,
                        "told the answer is wrong; repair + cave rate"),
    "Cn_restate": ("turn2", True,
                   "GATE: must be a no-op"),
    "C5_fresh": ("turn1", False,
                 "independent fresh attempt"),
    "C6_longer": ("turn1", False,
                  "longer initial attempt, compute control"),
    "C3_verify": ("verify", True,
                  "detection: is your own answer correct?"),
    # ---- Branch 4: counterfactual correction wording variants ----
    "C4a_mild": ("turn2", True,
                 "mild doubt; weakest negative signal"),
    "C4c_authority": ("turn2", True,
                      "external evaluator says wrong; REPORT SEPARATELY (social compliance)"),
    "C7_told_correct": ("turn2", True,
                        "told the answer is right; affirmation + entrenchment"),
    # ---- Branch 1t: the same instructions with the trace actually in the prompt ----
    "C1t_review_trace": ("turn2", True,
                         "review, trace in the user turn (C1 + the trace)"),
    "C4t_oracle_wrong_trace": ("turn2", True,
                               "told wrong, trace in the user turn (C4 + the trace)"),
    # ---- Branch 2: within-rollout backtracking interventions (turn 1) ----
    "B2_commit_first": ("turn1", False,
                        "DOWN: commit to first conclusion, do not revisit"),
    "B2_consider_alts": ("turn1", False,
                         "UP: two hypotheses, check earlier inferences"),
}

TURN2_INSTRUCTION = {
    "C1_review_full": REVIEW_NEUTRAL,
    "C2_review_answer_only": REVIEW_NEUTRAL,
    "C4_oracle_wrong": ORACLE_WRONG,
    "C4a_mild": MILD_DOUBT,
    "C4c_authority": AUTHORITY_WRONG,
    "C7_told_correct": TOLD_CORRECT,
    "Cn_restate": RESTATE,
    # Byte-identical to their non-trace twins: only the context differs.
    "C1t_review_trace": REVIEW_NEUTRAL,
    "C4t_oracle_wrong_trace": ORACLE_WRONG,
}

# A turn-1 condition absent from this dict is an unmodified fresh attempt, which is what
# makes C5_fresh the null.
TURN1_SUFFIX = {
    "C6_longer": LONGER,
    "B2_commit_first": COMMIT_FIRST,
    "B2_consider_alts": CONSIDER_ALTERNATIVES,
}

# Conditions whose assistant turn is the ANSWER ONLY, trace removed.
STRIP_TRACE = {"C2_review_answer_only"}

# Conditions that inline the turn-1 reasoning into the user turn.
TRACE_IN_USER = {"C1t_review_trace", "C4t_oracle_wrong_trace"}

# id -> the condition it is the trace-carrying twin of.
TRACE_TWIN = {"C1t_review_trace": "C1_review_full",
              "C4t_oracle_wrong_trace": "C4_oracle_wrong"}

# id -> the verdict asserted about turn 1. Branch 4 splits each on turn-1 correctness.
FEEDBACK_VERDICT = {
    "C4_oracle_wrong": False,
    "C4a_mild": False,
    "C4c_authority": False,
    "C7_told_correct": True,
}

# Sources the error claim to an external authority: social compliance, not self-assessment.
REPORT_SEPARATELY = {"C4c_authority"}

# Branch groupings, so a runner or a table never has to hardcode a condition list.
BRANCH1 = ["C1_review_full", "C2_review_answer_only", "C4_oracle_wrong",
           "C5_fresh", "C6_longer", "Cn_restate", "C3_verify"]
BRANCH4 = ["C4a_mild", "C4c_authority", "C7_told_correct"]
BRANCH2 = ["B2_commit_first", "B2_consider_alts"]


def strip_trace(response: str) -> str:
    """Drop the reasoning block, keep what followed.

    The <answer> fallback is required: ~1.2% of responses open <think> and never close it
    while still emitting a valid answer, and splitting on '</think>' alone would return the
    whole trace for those, silently turning C2 into C1.
    """
    if "</think>" in response:
        tail = response.split("</think>", 1)[1].strip()
        if tail:
            return tail
    import re
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if m:
        return f"<answer>{m[-1].strip()}</answer>"
    return response.strip()


def extract_trace(response: str) -> str:
    """The reasoning block, without the <think> tags. '' if there is none.

    Mirrors strip_trace: an unclosed <think> still has usable reasoning, so fall back to
    everything after the opening tag rather than returning nothing.
    """
    if "<think>" not in response:
        return ""
    body = response.split("<think>", 1)[1]
    if "</think>" in body:
        body = body.split("</think>", 1)[0]
    return body.strip()


def build_turn2(messages_turn1, turn1_response, condition):
    """[system, user] + assistant(turn1) + user(instruction).

    For TRACE_IN_USER conditions the reasoning is prepended to the user turn, because the
    chat template discards <think> from a non-final assistant turn. The result is the
    non-trace prompt with the trace inserted and nothing else changed.
    """
    assistant = (strip_trace(turn1_response) if condition in STRIP_TRACE
                 else turn1_response)
    instruction = TURN2_INSTRUCTION[condition]
    if condition in TRACE_IN_USER:
        trace = extract_trace(turn1_response)
        if trace:
            instruction = TRACE_PREAMBLE.format(trace=trace) + instruction
    return list(messages_turn1) + [
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": instruction},
    ]


def turn1_suffix(condition: str) -> str:
    """Suffix appended to the task prompt for a turn-1 condition ('' if unmodified)."""
    return TURN1_SUFFIX.get(condition, "")
