"""CPU merge of the S3 adapter into its exact S2 parent.

Ported from run_dir('pipeline_27b')/scripts/merge.py.
CLI: python merge.py STAGE_ROOT PARENT_MODEL
"""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,  # ty: ignore[unresolved-import]  (transformers 5.x, Modal image)
)

from resonance.stages.common import checkpoint, write_json

if __name__ == '__main__':
    root, source = Path(sys.argv[1]), str(checkpoint(sys.argv[2]))
    assert (root/'TRAIN_COMPLETE.json').exists()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16, device_map='cpu')
    model = PeftModel.from_pretrained(model, root/'adapter').merge_and_unload(safe_merge=True)
    model.save_pretrained(root/'model', safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(source).save_pretrained(root/'model')
    checkpoint(root/'model')
    write_json(root/'MERGE_COMPLETE.json', {'source': source, 'model': str(root/'model')})
