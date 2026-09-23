"""Stage-1a teacher prompts: the Sonnet-arm prompts with the reaction ID removed.

Ported from run_dir('traces')/scripts/prompts_stage1.py. The original extracted these
constants at import time from the Sonnet generation script and stripped the leading
"Reaction ID: {reaction_id}\n\n" line. Here they are stored as the exact extracted
strings (written with repr(), so every escape is preserved byte-for-byte) and checked
against the fingerprints recorded in every trace row of the published corpus.

The frozen snapshot (prompts_stage1_frozen.json) fingerprints the Sonnet source before
the ID was removed; ID_PREFIX + USER_TEMPLATE must still reproduce it.

CLI: python -m resonance.pipelines.traces.prompts_stage1   (prints the prompts)
"""
import hashlib
import json
from pathlib import Path

FROZEN = Path(__file__).with_name("prompts_stage1_frozen.json")
NAMES = ("SYSTEM_PROMPT", "USER_TEMPLATE")
ID_PREFIX = 'Reaction ID: {reaction_id}\n\n'

# Do not edit these literals: they are the bytes the teacher saw. Change the experiment
# by adding a new prompt module, not by touching these.
SYSTEM_PROMPT: str = 'You are an expert organic chemist with deep knowledge of reaction mechanisms, reagent roles, and synthetic strategy. Given a reaction described by SMILES strings, generate a clear, structured reaction trace that would be useful as training data for a chemistry AI agent focused on retrosynthesis and forward prediction.'
USER_TEMPLATE: str = 'Starting Material: {starting_material}\nReagents: {reagents}\nReaction Conditions: {conditions}\nProduct: {product}\n\nWrite a reaction trace covering:\n1. **Reaction type**: Name and classify this reaction (e.g. Suzuki coupling, reductive amination).\n2. **Transformation**: Which bonds are broken and which are formed; how the carbon skeleton changes.\n3. **Mechanism**: Step-by-step electron-pushing mechanism (concise but chemically precise).\n4. **Reagent roles**: What each reagent/condition does and why it is necessary.\n5. **Retrosynthetic note**: The retrosynthetic disconnection that reveals this as the key step.\n\nUse standard organic chemistry terminology. Be concise — aim for 150–300 words.'

# Recorded in the `prompt_fingerprint` field of every published trace row.
EXPECTED_FINGERPRINT = {'SYSTEM_PROMPT': {'len': 319, 'sha256_16': '535cfdb84475881d'}, 'USER_TEMPLATE': {'len': 695, 'sha256_16': 'c543cca7ceea341d'}}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


_p = {"SYSTEM_PROMPT": SYSTEM_PROMPT, "USER_TEMPLATE": USER_TEMPLATE}
FINGERPRINT = {n: {"len": len(_p[n]), "sha256_16": sha(_p[n])} for n in NAMES}
if FINGERPRINT != EXPECTED_FINGERPRINT:
    raise RuntimeError(f"stage-1a prompts changed: {FINGERPRINT} != {EXPECTED_FINGERPRINT}")

# Fingerprint of the prompts as they appeared in the Sonnet source (with the ID line).
_source = {"SYSTEM_PROMPT": SYSTEM_PROMPT, "USER_TEMPLATE": ID_PREFIX + USER_TEMPLATE}
SOURCE_FINGERPRINT = {n: {"len": len(_source[n]), "sha256_16": sha(_source[n])} for n in NAMES}


def frozen() -> dict | None:
    return json.loads(FROZEN.read_text()) if FROZEN.exists() else None


if __name__ == "__main__":
    for _n in NAMES:
        _v = _p[_n]
        print("=" * 78)
        print(f"{_n}   ({len(_v)} chars, sha256[:16]={sha(_v)})")
        print("=" * 78)
        print(_v)
        print()
