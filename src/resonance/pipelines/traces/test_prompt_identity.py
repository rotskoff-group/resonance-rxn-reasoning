#!/usr/bin/env python3
"""Check frozen Sonnet source and the exact ID-free Kimi transformation.

Ported from run_dir('traces')/scripts/test_prompt_identity.py. The prompts are now
literals (prompts_stage1.py), so the "source" is reconstructed as ID_PREFIX + USER_TEMPLATE
and checked against the frozen Sonnet fingerprint instead of re-parsing the Sonnet script.
CLI: python -m resonance.pipelines.traces.test_prompt_identity
"""
from resonance.pipelines.traces import prompts_stage1 as P

NAMES = ("SYSTEM_PROMPT", "USER_TEMPLATE")

# Present in the Sonnet prompt on purpose. Stage 1a is SUPPOSED to be retrospective; if these
# are gone, someone has changed the experiment rather than fixed a bug.
REQUIRED_IN_USER = (
    "Product: {product}",
    "aim for 150",
    "1. **Reaction type**",
    "5. **Retrosynthetic note**",
)


def main() -> int:
    failures = []

    print(f"source: {P.__file__}")
    for name in NAMES:
        val = getattr(P, name)
        fp = P.FINGERPRINT[name]
        print(f"  {name:16s} len={fp['len']:5d}  sha256[:16]={fp['sha256_16']}")
        if not val.strip():
            failures.append(f"{name} is empty")

    frozen = P.frozen()
    if frozen is None:
        print("\nWARNING: no frozen snapshot; writing one is how drift becomes detectable.")
    else:
        for name in NAMES:
            want, got = frozen.get(name), P.SOURCE_FINGERPRINT[name]
            if want != got:
                failures.append(
                    f"{name} changed vs frozen snapshot: {want} -> {got}. The SONNET script "
                    f"has been edited, which silently changes this experiment. Either revert "
                    f"it or re-freeze deliberately.")
            else:
                print(f"  {name:16s} matches frozen snapshot")

    for needle in REQUIRED_IN_USER:
        if needle not in P.USER_TEMPLATE:
            failures.append(
                f"USER_TEMPLATE no longer contains {needle!r} -- stage 1a is supposed to be "
                f"retrospective; naturalness belongs in a separate reformatting pass")

    if P.FINGERPRINT != P.EXPECTED_FINGERPRINT:
        failures.append("Prompts differ from the published corpus fingerprint")
    if "reaction_id" in P.USER_TEMPLATE or "Reaction ID" in P.USER_TEMPLATE:
        failures.append("Reaction ID is still in the user prompt")

    if failures:
        print("\nPROMPT IDENTITY FAILED -- do not launch generation")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nprompt identity OK (Sonnet source frozen; Kimi omits reaction ID)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
