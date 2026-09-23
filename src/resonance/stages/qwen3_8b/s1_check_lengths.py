"""Check that every S1 SFT example fits the training context, with no truncation.

Ported from the inline check in run_dir('traces')/bash_scripts/04_sft_no_id_8gpu.sbatch.

    python -m resonance.stages.qwen3_8b.s1_check_lengths <sft_dataset_dir>
"""
import sys

from datasets import load_from_disk
from transformers import AutoTokenizer


def main() -> None:
    dataset = load_from_disk(sys.argv[1])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    lengths = []
    for batch in dataset.iter(batch_size=256):
        # Column access returns a datasets.Column; the tokenizer requires list[str].
        texts = list(batch["text"])
        lengths.extend(len(ids) for ids in tok(texts, add_special_tokens=False, truncation=False)["input_ids"])
    assert len(lengths) == 40000
    assert max(lengths) <= 2048, f"Refusing truncation: maximum length {max(lengths)} > 2048"
    print(f"All {len(lengths)} examples fit; max tokens={max(lengths)}", flush=True)


if __name__ == "__main__":
    main()
