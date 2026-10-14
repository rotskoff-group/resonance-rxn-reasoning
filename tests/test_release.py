"""Offline checks of what the release depends on: prompts, scoring, data placement, Fig 2 input.

    uv run python -m pytest -q tests
"""
import json
from pathlib import Path

import pytest

from resonance.evaluation.extractors import extract_forward_rxn, match_forward_rxn
from resonance.paper.export import staged_accuracy
from resonance.predict import messages

DATA = Path(__file__).parent / "data"
# Two USPTO-50k test reactions (one without reagents, one with two) and the exact messages the
# paper's evaluations sent: the 8B S1 benchmark and the 27B S1 evaluation.
RECORDED = json.loads((DATA / "recorded_prompts.json").read_text())


@pytest.mark.parametrize("rec", RECORDED, ids=lambda r: f"row{r['row']}")
@pytest.mark.parametrize("family", ["8b", "27b"])
def test_prompts_match_recorded_evaluation(rec, family):
    assert messages(rec["starting_material"], rec["reagents"], family) == rec[f"messages_{family}"]


def test_no_reagents_wording():
    rec = next(r for r in RECORDED if not r["reagents"])
    assert "Reagents: No reagents" in messages(rec["starting_material"], [], "8b")[1]["content"]


def test_last_answer_block_is_scored():
    text = "<think>try <answer>CCO</answer> no wait</think>\n<answer>OCC</answer>"
    assert extract_forward_rxn(text) == "OCC"
    assert match_forward_rxn("OCC", "CCO", "smiles")          # canonical SMILES, not strings
    assert not match_forward_rxn("CCN", "CCO", "smiles")


def test_answer_inside_unclosed_thinking_counts():
    # Paper scoring rule: the last <answer> tag anywhere, even inside unfinished thinking.
    assert extract_forward_rxn("<think>so <answer>CCO</answer> and then") == "CCO"


def test_download_places_files_at_run_dir_paths(tmp_path, monkeypatch):
    import resonance.download as download
    repo = tmp_path / "repo"
    src = repo / "iclr2027/raw/s3_8b/outputs/benchmark/results.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("{}\n")
    root = tmp_path / "data"
    monkeypatch.setattr(download, "DATA_ROOT", root)
    monkeypatch.setattr(download, "wanted", lambda which: ["s3_8b/outputs/benchmark/results.jsonl"])
    download.fetch("figures", from_dir=repo)
    assert (root / download.RUN_DIRS["s3_8b"] / "outputs/benchmark/results.jsonl").read_text() == "{}\n"


def test_figure_input_list_uses_known_keys():
    from resonance.download import FIGURE_INPUTS
    from resonance.paths import RUN_DIRS
    keys = {line.split("/", 1)[0] for line in FIGURE_INPUTS.read_text().split()}
    assert keys and keys <= set(RUN_DIRS)


def test_fig2_reads_overall_counts():
    correct = {"8b_base": 59, "8b_s1": 1551, "8b_s2": 1946, "8b_s3": 2140, "kimi": 2872, "opus": 3934, "gpt6": 3152}
    fig = staged_accuracy({"overall": {"n": 5000, "correct": correct}})
    bars = [b["correct"] for g in fig["groups"] for b in g["bars"]]
    assert bars == [59, 1551, 1946, 2140, 2872, 3934, 3152]
