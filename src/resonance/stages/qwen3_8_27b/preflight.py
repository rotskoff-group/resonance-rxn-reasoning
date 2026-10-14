"""One CPU update on a tiny model of the same class, testing LoRA and DPO+NLL.

Ported from run_dir('pipeline_27b')/scripts/preflight.py. CLI: python preflight.py
"""
import math
import tempfile
from typing import Any, cast

import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3_5Config,  # ty: ignore[unresolved-import]  (transformers 5.x, Modal image)
    Qwen3_5ForConditionalGeneration,  # ty: ignore[unresolved-import]  (transformers 5.x, Modal image)
)
from trl import DPOConfig

from resonance.stages.qwen3_8_27b.dpo_tokens import TokenDPOTrainer

if __name__ == '__main__':
    torch.set_num_threads(2)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {str(i): i for i in range(128)}, unk_token='0')), unk_token='0', pad_token='1', eos_token='2')
    config = Qwen3_5Config(text_config={'vocab_size': 128, 'hidden_size': 32,
        'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 2,
        'num_key_value_heads': 2, 'head_dim': 16, 'layer_types': ['full_attention'],
        'max_position_embeddings': 128, 'pad_token_id': 1, 'eos_token_id': 2},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
            'num_heads': 2, 'out_hidden_size': 32, 'num_position_embeddings': 16})
    model = Qwen3_5ForConditionalGeneration(config)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    ds = Dataset.from_list([{'prompt_ids': [5,6,7], 'chosen_ids': [8,9,2], 'rejected_ids': [10,11,2]}]*4)
    with tempfile.TemporaryDirectory() as tmp:
        args = DPOConfig(output_dir=tmp, use_cpu=True, bf16=False, max_steps=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_length=64, loss_type=['sigmoid','sft'], loss_weights=[1.,.2],
            precompute_ref_log_probs=True, precompute_ref_batch_size=1, beta=.1,
            report_to='none', save_strategy='no', logging_steps=1, remove_unused_columns=False)
        trainer = TokenDPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
            processing_class=tokenizer, peft_config=LoraConfig(r=4,lora_alpha=8,
                target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
        peft_model = cast(Any, trainer.model)   # the PeftModel the trainer wrapped
        before = {n:p.detach().clone() for n,p in peft_model.named_parameters() if p.requires_grad}
        result = trainer.train()
        assert math.isfinite(result.training_loss)
        assert any(not torch.equal(before[n], p) for n,p in peft_model.named_parameters() if n in before)
        trainer.save_model(tmp+'/adapter')
        merged = peft_model.merge_and_unload(safe_merge=True)
        assert merged(input_ids=torch.tensor([[5,6,7]])).logits.shape == (1,3,128)
    print('PREFLIGHT_OK: finite DPO+NLL update, adapter changed, reference precompute and merge passed')
