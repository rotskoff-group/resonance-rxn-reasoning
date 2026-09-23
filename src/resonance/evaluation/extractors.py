"""
Evaluation parsers for task-specific answer extraction and matching.

Each task type may use a different response format or require specialised
comparison logic (e.g. SMILES canonicalisation).  Add one
``extract_<task>`` / ``match_<task>`` pair per task type and register them
in ``TASK_PARSERS`` at the bottom of this module.
"""

import re
from rdkit import Chem
from typing import Optional


def extract_forward_rxn(text: str) -> Optional[str]:
    """Extract the SMILES/IUPAC answer from the model output.

    Tries in order:
      1. Content of the last <answer>...</answer> tag
      2. Content of the last <product>...</product> tag
      3. Content of the last <forward_reaction>...</forward_reaction> XML tag (legacy)
      4. Content of the last \\boxed{} expression
      5. Text after </think> block (if present and complete)
    Truncated <think> blocks (no closing tag) return None.
    """
    # 1. <answer> tags
    answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if answer_matches:
        return answer_matches[-1].strip()

    # 2. <product> tags (ForwardReactionPrediction / RL-training prompt)
    product_matches = re.findall(r"<product>\s*(.*?)\s*</product>", text, re.DOTALL)
    if product_matches:
        return product_matches[-1].strip()

    # 3. <forward_reaction> tags (legacy)
    xml_matches = re.findall(
        r"<forward_reaction>\s*(.*?)\s*</forward_reaction>", text, re.DOTALL
    )
    if xml_matches:
        return xml_matches[-1].strip()

    # 4. \boxed{}
    boxed = _extract_boxed_nested(text)
    if boxed is not None:
        return boxed

    # 5. If response has a <think> block, only use text after </think>
    if "<think>" in text:
        if "</think>" not in text:
            # Truncated mid-think — no usable answer
            return None
        after_think = text.split("</think>")[-1].strip()
        return after_think if after_think else None

    # 6. Full response fallback (no think block)
    stripped = text.strip()
    return stripped if stripped else None


def _canonicalize(smiles: str) -> Optional[str]:
    """Return the RDKit canonical SMILES, or None if the string is invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def match_forward_rxn(
    predicted: str, ground_truth: str, answer_type: Optional[str]
) -> bool:
    """Compare predicted and ground-truth forward-reaction answers.

    For SMILES-based answer types (``"smiles"``, ``"atom_mapped"``) both
    strings are canonicalised with RDKit before comparison so that equivalent
    but differently-written SMILES are treated as correct.  Atom-map numbers
    are stripped from atom-mapped SMILES before canonicalisation.

    For ``"iupac"`` (and any unknown type) a case-insensitive string
    comparison is used as a fallback.
    """
    if answer_type in ("smiles", "atom_mapped"):
        # Strip atom-map numbers before canonicalising
        pred_clean = re.sub(r":\d+", "", predicted)
        gt_clean = re.sub(r":\d+", "", str(ground_truth))

        pred_canon = _canonicalize(pred_clean)
        gt_canon = _canonicalize(gt_clean)

        if pred_canon is not None and gt_canon is not None:
            return pred_canon == gt_canon

        # RDKit failed to parse one or both — fall back to string comparison
        return predicted.strip() == str(ground_truth).strip()

    # iupac or unknown: case-insensitive string equality
    return predicted.lower() == str(ground_truth).strip().lower()


def _extract_boxed_nested(text: str) -> Optional[str]:
    """Extract last \\boxed{} content, handling nested braces."""
    results = []
    i = 0
    marker = r"\boxed{"
    while True:
        pos = text.find(marker, i)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start : j - 1])
        i = pos + 1
    return results[-1].strip() if results else None


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract the last <answer>...</answer> tag, falling back to \\boxed{}."""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return _extract_boxed_nested(text)


def match_mol_property(
    predicted: str, ground_truth: str, answer_type: Optional[str]
) -> bool:
    """Match predicted vs ground_truth for molecular property question types."""
    if answer_type in ("integer", "int", "numeric", "number"):
        try:
            return int(predicted) == int(str(ground_truth))
        except (ValueError, TypeError):
            pass
    return predicted.strip().lower() == str(ground_truth).strip().lower()


def _normalize_molecular_formula(formula: str) -> str:
    """Convert LaTeX formula (e.g. C_{15}H_{12}N_2O_2) to plain text (C15H12N2O2)."""
    result = re.sub(r"_\{([^}]*)\}", r"\1", formula)
    result = re.sub(r"_([A-Za-z0-9])", r"\1", result)
    result = result.replace("{", "").replace("}", "").replace("\\", "")
    return result.strip()


def extract_molecular_formula(text: str) -> Optional[str]:
    raw = extract_answer_tag(text)
    if raw is None:
        return None
    return _normalize_molecular_formula(raw)


def match_molecular_formula(
    predicted: str, ground_truth: str, answer_type: Optional[str]
) -> bool:
    return predicted.strip() == str(ground_truth).strip()


_MOL_PROPERTY_TYPES = (
    "atom_count",
    "functional_group_presence",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "heavy_atom_count",
    "ring_count",
    "hba",
    "hbd",
    "rotatable_bonds",
    "stereocenters",
)

def extract_retro_enum(text: str) -> Optional[str]:
    """Extract the reactant SMILES answer from a retrosynthesis response.

    Tries in order:
      1. Content of <retrosynthetic_analysis>...</retrosynthetic_analysis> tags
      2. Content of \\boxed[...] (square-bracket form used by RetroPrompt)
      3. Content of \\boxed{...} (curly-bracket fallback)
      4. Text after </think> block
    """
    matches = re.findall(
        r"<retrosynthetic_analysis>\s*(.*?)\s*</retrosynthetic_analysis>",
        text, re.DOTALL
    )
    if matches:
        return matches[-1].strip()

    sq_matches = re.findall(r"\\boxed\[([^\]]*)\]", text)
    if sq_matches:
        return sq_matches[-1].strip()

    boxed = _extract_boxed_nested(text)
    if boxed is not None:
        return boxed

    if "<think>" in text:
        if "</think>" not in text:
            return None
        after = text.split("</think>")[-1].strip()
        return after if after else None

    stripped = text.strip()
    return stripped if stripped else None


def _canonicalize_set(smiles_str: str) -> Optional[frozenset]:
    """Split a dot-joined or JSON-array SMILES string and return a frozenset of canonical SMILES.

    Accepts both dot-separated format (``SMILES1.SMILES2``) and JSON array
    format (``["SMILES1", "SMILES2"]``) so that model outputs using either
    style are evaluated correctly.

    Returns None if any fragment fails RDKit canonicalization.
    """
    import json as _json

    stripped = smiles_str.strip()
    if stripped.startswith("["):
        try:
            parsed = _json.loads(stripped)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                frags = [s.strip() for s in parsed if s.strip()]
            else:
                raise ValueError
        except Exception:
            frags = [s.strip() for s in stripped.split(".") if s.strip()]
    else:
        frags = [s.strip() for s in stripped.split(".") if s.strip()]

    canon = set()
    for frag in frags:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return None
        canon.add(Chem.MolToSmiles(mol))
    return frozenset(canon) if canon else None


def match_retro_enum(
    predicted: str, ground_truth: str, answer_type: Optional[str]
) -> bool:
    """Exact set-match of reactant SMILES after RDKit canonicalization.

    Accepts dot-separated or JSON-array format for the predicted string.
    Returns True iff the canonical sets are equal.
    """
    pred_set = _canonicalize_set(predicted)
    gt_set = _canonicalize_set(str(ground_truth))
    if pred_set is None or gt_set is None:
        return False
    return pred_set == gt_set



def extract_retro_uspto(text: str) -> Optional[str]:
    """Extract the reactant SMILES for the USPTO-50K retro task.

    Deliberately permissive about the wrapper, so that the number measured is
    retrosynthetic ability rather than output-format compliance. Tries, in order:

      1. <answer>...</answer>            -- what RetroUSPTOPrompt asks for, and the tag
                                            every SFT checkpoint in this project emits
      2. <retrosynthetic_analysis>...</> -- the RetroPrompt convention
      3. \\boxed[...] then \\boxed{...}   -- also RetroPrompt
      4. text after </think>             -- last resort

    Separate from extract_retro_enum rather than an edit to it: retro_enum is already in
    use and must keep its current behaviour.
    """
    m = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if m:
        return m[-1].strip()
    return extract_retro_enum(text)


TASK_PARSERS: dict[str, tuple] = {
    "forward_rxn": (extract_forward_rxn, match_forward_rxn),
    "retro_enum": (extract_retro_enum, match_retro_enum),
    "retro_uspto": (extract_retro_uspto, match_retro_enum),
    "molecular_formula": (extract_molecular_formula, match_molecular_formula),
    **{qt: (extract_answer_tag, match_mol_property) for qt in _MOL_PROPERTY_TYPES},
}
