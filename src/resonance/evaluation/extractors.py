"""
Evaluation parsers for task-specific answer extraction and matching.

Each task type may use a different response format or require specialised
comparison logic (e.g. SMILES canonicalisation).  Add one
``extract_<task>`` / ``match_<task>`` pair per task type and register them
in ``TASK_PARSERS`` at the bottom of this module.
"""

import re

from rdkit import Chem


def extract_forward_rxn(text: str) -> str | None:
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


def _canonicalize(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES, or None if the string is invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def match_forward_rxn(
    predicted: str, ground_truth: str, answer_type: str | None
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


def _extract_boxed_nested(text: str) -> str | None:
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


TASK_PARSERS: dict[str, tuple] = {
    "forward_rxn": (extract_forward_rxn, match_forward_rxn),
}
