# Glossary

Terms used in the paper, the code and the run directories, and how they map onto each other.

## Pipeline

| Term | Meaning |
|---|---|
| **Teacher** | Kimi K3 (`moonshotai/Kimi-K3`, reasoning effort "low"). It writes one reasoning trace per USPTO-50k training reaction, with the recorded product visible. It is also evaluated zero-shot on the test set (57.4%). |
| **Student** | The model being trained: Qwen3-8B (giving **Resonance-8B**) or Qwen3.8-27B (giving **Resonance-27B**). |
| **Base** | The student before any training. 8B: 1.2% (greedy). 27B: 26.3%. |
| **S1** | Stage 1, *teacher SFT*: fine-tune the base model on the 40,000 Kimi K3 traces, each followed by the recorded product in `<answer>` tags. |
| **S2** | Stage 2, *rejection-sampling SFT* (RS-SFT): sample 8 completions per training reaction from S1 (T 0.9), keep the exactly correct ones and fine-tune on them. 8B keeps 47,492 examples (full fine-tune). 27B uses a difficulty-mixed set of 98,731 examples (LoRA r128, one epoch). |
| **S3** | Stage 3, *DPO + NLL*: sample 16 completions per reaction from S2 (T 0.7), pair a random correct completion (chosen) with the incorrect completion whose product is most Tanimoto-similar to the target (rejected), one pair per reaction. Train with DPO (β 0.1) plus 0.2 × the chosen response's NLL, LoRA r128. 8B: 28,747 pairs, 3,000 steps. 27B: 12,079 pairs, one epoch. |
| **skip-S2** | Ablation: the S3 recipe applied directly to S1 (8B: 26,661 pairs; 27B: 13,698 pairs). |
| **No-ID** | The teacher prompt omits the patent reaction ID. Every paper model is in the no-ID lineage; the 27B run id `no-id-b97ef67b9ba65863` is `"no-id-"` + the first 16 hex digits of the traces' sha256. |
| **Gold-answer fallback** | 27B S2 only: for the 1,108 training reactions with no correct S1 sample, the target is the recorded product alone (answer-only). |
| **Difficulty mix** | 27B S2 data: every correct sample from reactions solved in 1 to 7 of 8 tries, plus one or two samples from reactions solved 8/8 ("easy replay", 2:1 mixed to easy). Run name `s2-v5-difficulty-mix-1ep`. |

## Evaluation

| Term | Meaning |
|---|---|
| **USPTO-50k** | 50,000 patent reactions; standard split 40,000 train / 5,000 val / 5,000 test. The *starting material* is the first reactant; the *reagents* are all other reactants. A single-reactant reaction reads "Reagents: No reagents". |
| **Top-1 / Acc@1** | The prediction is the **last `<answer>` block anywhere in the response** (`resonance.evaluation.extractors`, the "lenient" scorer). It is correct if its RDKit canonical SMILES equals the canonical recorded product. Invalid, missing and truncated answers count as wrong and stay in the denominator (N = 5,000). |
| **Strict scorer** | Reads only text after `</think>` (`resonance.stages.common.prediction`). The 27B pipeline stores this flag, but every number in the paper uses the lenient scorer. |
| **Short / long system prompt** | The paper's names for the two evaluation system prompts. In code these are `ForwardRxnPrompt.get_system_prompt("none")` (short) and `("rules")` (long). The 27B scripts use copies in `configs/paper/qwen3_8_27b/prompts.json` with leading and trailing whitespace removed (the inner indentation is kept). Short: the 8B students and the frontier models. Long: the 8B base model, every 27B evaluation, and the 8B S2/S3 data sampling. |
| **Reaction classes** | 11 rule-based classes (appendix table "Reaction classes in USPTO-50k"), with the **benzyl-ether correction**: the classifier's `[OX2]Cc1ccccc1` matched benzyl *alcohols*, and `[OX2;H0]` fixes it. 282 of 5,000 test labels and 2,247 of 40,000 training labels move. Test labels: `resonance.paper.labels`. Training labels: `resonance.pipelines.labels.relabel_train`. |
| **Frontier references** | Kimi K3 (57.4%), Claude Opus 5 (78.7%, 3,934/5,000) and GPT-6-Astra (63.0%), each zero-shot with the short system prompt. Evaluated only; nothing was trained on their test outputs. |

## Reasoning analysis (Resonance-8B, paper §4.3)

| Term | Code id | Meaning |
|---|---|---|
| Fresh resample | `C5_fresh` | An unprompted resample; the baseline for the length prompts. |
| Commit | `B2_commit_first` | "Commit to your first well-supported conclusion..." |
| Consider alternatives | `B2_consider_alts` | "Consider at least two plausible product hypotheses..." |
| Think more | `C6_longer` | "Work through this carefully and at length..." |
| Neutral review | `C1_review_full` | Second turn: "Review your work above..."; the baseline for feedback. |
| Mild doubt | `C4a_mild` | "There may be an issue with it." |
| Told wrong | `C4_oracle_wrong` | "Your answer above is incorrect." |
| Marked wrong | `C4c_authority` | "An evaluator marked the answer above as incorrect." |
| Told correct | `C7_told_correct` | "Your answer above is correct." |
| Plurality@k | | The most frequent canonical product among the first k samples (ties go to the earliest). |
| Backtracking | | Frozen lexical detector (`resonance.analysis.backtrack`, definition 2026-09-12.1) counting phrases such as "wait" or "let me reconsider" in the thinking block, as events per 1,000 output tokens. |
| Per-reaction seed | | Draw *j* of reaction *i* uses seed 42 + *i*. The earlier samples shared one seed across reactions, which correlated draws; `resonance.paper.draw_independence` fails on that. |

## Code

| Term | Meaning |
|---|---|
| `RESONANCE_DATA_ROOT` | The directory holding every run directory; defaults to `./data`. |
| `run_dir(key)` | `RESONANCE_DATA_ROOT / RUN_DIRS[key]` (`resonance.paths`). Every script finds its inputs and outputs this way; see [DATA.md](DATA.md) for what each key holds. |
| `frozen/` | `resonance.analysis.metareasoning.frozen`: verbatim copies of the scripts that produced the metareasoning samples, checked by hash (`MANIFEST.json`). Never edit them. Their model registry (`frozen/models.py`) describes an earlier lineage, not the paper's models. |
| Modal | The 27B stages ran on [Modal](https://modal.com) (8×H200); see `src/resonance/stages/qwen3_8_27b/`. |
