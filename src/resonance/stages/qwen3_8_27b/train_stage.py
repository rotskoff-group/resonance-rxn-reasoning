"""S2 full text SFT and S3 LoRA DPO, initialized only from a completed predecessor.

Ported from run_dir('pipeline_27b')/scripts/train_stage.py (the version on disk after
S2-V5 launched; its recorded training_config matches this code path).
Run under torchrun with STAGE_ROOT, STAGE_MODEL, STAGE [, STAGE_DATA_ROOT, TRAIN_MODE,
S2_NUM_TRAIN_EPOCHS, S3_NUM_TRAIN_EPOCHS, GRADIENT_ACCUMULATION_STEPS, PIPELINE_VOLUME].
"""
import math
import os
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,  # ty: ignore[unresolved-import]  (transformers 5.x, in the Modal image)
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from resonance.stages.common import checkpoint, write_json


def main():
    started = time.monotonic()
    root = Path(os.environ['STAGE_ROOT'])
    data_root = Path(os.environ.get('STAGE_DATA_ROOT', str(root)))
    source = str(checkpoint(os.environ['STAGE_MODEL']))
    stage = os.environ['STAGE']
    s2_epochs = float(os.environ.get('S2_NUM_TRAIN_EPOCHS', '3'))
    assert math.isfinite(s2_epochs) and s2_epochs > 0
    train_mode = os.environ.get('TRAIN_MODE', 'full')
    assert train_mode in ('full', 'lora')
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '8'))
    expected_world = int(os.environ.get('EXPECTED_WORLD_SIZE', '8'))
    assert world == expected_world, (world, expected_world)
    torch.cuda.set_device(rank)
    set_seed(42)
    ds = load_from_disk(str(data_root / 'dataset'))
    tok = AutoTokenizer.from_pretrained(source)
    tok.pad_token = tok.pad_token or tok.eos_token
    grad_accum = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', '4'))
    assert world * grad_accum == 32
    common: dict[str, Any] = dict(output_dir=str(root/'checkpoints'), per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=10,
        save_strategy='steps', save_steps=500, save_total_limit=2,
        report_to='none', dataloader_num_workers=0, remove_unused_columns=False,
        ddp_timeout=1200, seed=42, disable_tqdm=True)
    if stage == 's2' and train_mode == 'full':
        zero = {'bf16': {'enabled': True}, 'train_micro_batch_size_per_gpu': 1,
            'gradient_accumulation_steps': 4, 'train_batch_size': 32, 'gradient_clipping': 1.0,
            'zero_optimization': {'stage': 3, 'overlap_comm': True, 'contiguous_gradients': True,
                'reduce_bucket_size': 50000000, 'stage3_prefetch_bucket_size': 50000000,
                'stage3_param_persistence_threshold': 100000,
                'stage3_gather_16bit_weights_on_model_save': True}}
        args = TrainingArguments(**common, num_train_epochs=s2_epochs, learning_rate=1e-5,
                                 warmup_steps=50, deepspeed=zero, optim='adamw_torch')
    elif stage == 's2':
        args = TrainingArguments(**common, num_train_epochs=s2_epochs, learning_rate=2e-5,
            warmup_steps=50, optim='adamw_torch', ddp_find_unused_parameters=False)
    else:
        from trl import DPOConfig
        s3_epochs = os.environ.get('S3_NUM_TRAIN_EPOCHS')
        if s3_epochs is not None:
            s3_epochs = float(s3_epochs)
            assert math.isfinite(s3_epochs) and s3_epochs > 0
            budget: dict[str, Any] = dict(max_steps=-1, num_train_epochs=s3_epochs)
        else:
            budget = dict(max_steps=3000)
        # DDP fits the frozen BF16 base on each H200 and avoids ZeRO-3/ref-precompute incompatibility.
        args = DPOConfig(**common, **budget, learning_rate=2e-6,
            max_length=4096, beta=.1, loss_type=['sigmoid', 'sft'], loss_weights=[1., .2],
            precompute_ref_log_probs=True, precompute_ref_batch_size=1,
            sync_ref_model=False, ddp_find_unused_parameters=False)
    load_kwargs = {'device_map': {'': rank}} if (stage == 's3' or train_mode == 'lora') else {}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16,
                                                           attn_implementation='sdpa', **load_kwargs)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    for name, p in model.named_parameters():
        if 'visual' in name:
            p.requires_grad_(False)

    class Commit(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            soft_seconds = int(os.environ.get('TRAIN_SOFT_SECONDS', '72000'))
            stop = torch.tensor(int(time.monotonic()-started > soft_seconds), device='cuda')
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                control.should_training_stop = True
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            dist.barrier()
            if global_rank == 0:
                import modal
                modal.Volume.from_name(os.environ['PIPELINE_VOLUME']).commit()
            dist.barrier()

    if stage == 's2':
        if train_mode == 'lora':
            from peft import LoraConfig, get_peft_model
            targets = [name for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Linear)
                       and 'visual' not in name and not name.endswith('lm_head')]
            assert targets
            model = get_peft_model(model, LoraConfig(
                r=128, lora_alpha=32, target_modules=targets, lora_dropout=.05,
                bias='none', task_type='CAUSAL_LM'))
        def collate(rows):
            batch = tok.pad([{'input_ids': r['input_ids'], 'attention_mask': [1]*len(r['input_ids'])}
                for r in rows], padding=True, pad_to_multiple_of=64, return_tensors='pt')
            if 'labels' in rows[0]:
                labels = torch.full_like(batch['input_ids'], -100)
                for index, row in enumerate(rows):
                    labels[index, :len(row['labels'])] = torch.tensor(row['labels'], dtype=torch.long)
                batch['labels'] = labels
            else:
                batch['labels'] = batch['input_ids'].clone()
                batch['labels'][batch['attention_mask']==0] = -100
            return batch
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate,
                          callbacks=[Commit()])
    else:
        from peft import LoraConfig

        from resonance.stages.qwen3_8_27b.dpo_tokens import TokenDPOTrainer
        # All text Linear modules except the output head; the unused vision tower is excluded.
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and 'visual' not in name and not name.endswith('lm_head')]
        assert targets
        # stage == 's3' here, so `args` is the DPOConfig built above and `ds` the saved Dataset.
        trainer = TokenDPOTrainer(model=model, ref_model=None, args=cast(Any, args), train_dataset=cast(Any, ds),
            processing_class=tok, callbacks=[Commit()], peft_config=LoraConfig(
                r=128, lora_alpha=32, target_modules=targets, lora_dropout=.05,
                bias='none', task_type='CAUSAL_LM'))
    if global_rank == 0:
        import importlib.metadata as meta
        write_json(root/'training_config.json', {'stage': stage, 'train_mode': train_mode,
            'source': source, 'data_root': str(data_root),
            'examples': len(ds), 'effective_batch': 32, 'args': args.to_dict(),
            'packages': {p: meta.version(p) for p in ['torch','transformers','trl','peft','deepspeed']}})
    resume = get_last_checkpoint(str(root/'checkpoints')) if (root/'checkpoints').exists() else None
    if resume:
        assert (Path(resume)/'trainer_state.json').exists(), 'Incomplete checkpoint'
    result = trainer.train(resume_from_checkpoint=resume)
    assert trainer.state.global_step > 0
    complete = trainer.state.global_step >= trainer.state.max_steps
    if not complete:
        if global_rank == 0:
            write_json(root/'PAUSED.json', {'global_step': trainer.state.global_step,
                'epochs': trainer.state.epoch, 'checkpoint': get_last_checkpoint(str(root/'checkpoints'))})
        dist.barrier()
        dist.destroy_process_group()
        return
    assert trainer.state.global_step == trainer.state.max_steps
    dest = root / ('model' if stage == 's2' and train_mode == 'full' else 'adapter')
    trainer.save_model(str(dest))
    if global_rank == 0:
        tok.save_pretrained(dest)
        if stage == 's2' and train_mode == 'full':
            checkpoint(dest)
        else:
            assert (dest/'adapter_model.safetensors').exists()
        write_json(root/'TRAIN_COMPLETE.json', {'stage': stage, 'train_mode': train_mode,
            'source': source,
            'global_step': trainer.state.global_step, 'epochs': trainer.state.epoch,
            'metrics': result.metrics, 'artifact': str(dest)})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
