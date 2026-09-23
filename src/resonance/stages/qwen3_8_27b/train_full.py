"""Full three-epoch SFT with resumable checkpoints and a consolidated final model.

Ported from run_dir('s1_27b')/scripts/train_full.py.
Run under torchrun inside the S1 Modal app with PILOT_OUTPUT, PILOT_MODEL, PILOT_REVISION,
PILOT_VOLUME (the volume committed at each checkpoint).
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import SequentialSampler
from transformers import (AutoTokenizer, Qwen3_5ForConditionalGeneration,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

if __name__ == '__main__':
    OUT = Path(os.environ["PILOT_OUTPUT"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    microbatch = int(os.environ.get("PILOT_MICROBATCH", "1"))
    accumulation = int(os.environ.get("PILOT_ACCUMULATION", "4"))
    checkpointing = os.environ.get("PILOT_CHECKPOINTING", "1") == "1"
    max_steps = int(os.environ.get("PILOT_MAX_STEPS", "100"))
    hourly_cost = float(os.environ.get("PILOT_HOURLY_COST", "39.87216"))
    assert microbatch * accumulation * world_size == 32, "Preserve effective batch size 32"
    torch.cuda.set_device(rank)
    set_seed(42)
    started = time.monotonic()
    if global_rank == 0:
        import importlib.metadata
        (OUT / "packages.json").write_text(json.dumps({p: importlib.metadata.version(p) for p in
            ["torch", "transformers", "accelerate", "deepspeed", "flash-linear-attention", "causal-conv1d", "tilelang"]}, indent=2))
    stats = json.loads(Path("/pilot/data_stats.json").read_text())
    ds_config = {
        "bf16": {"enabled": True}, "train_micro_batch_size_per_gpu": microbatch,
        "gradient_accumulation_steps": accumulation, "train_batch_size": 32,
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": 50000000, "stage3_prefetch_bucket_size": 50000000,
            "stage3_param_persistence_threshold": 100000,
            "stage3_gather_16bit_weights_on_model_save": True},
    }
    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"), num_train_epochs=3,
        per_device_train_batch_size=microbatch, gradient_accumulation_steps=accumulation,
        learning_rate=2e-5, warmup_steps=100, lr_scheduler_type="linear", bf16=True,
        gradient_checkpointing=checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=ds_config, optim="adamw_torch", logging_steps=10, save_strategy="steps", save_steps=500, save_total_limit=2,
        report_to="none", dataloader_num_workers=0, remove_unused_columns=False,
        ddp_timeout=1200, seed=42, disable_tqdm=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["PILOT_MODEL"], revision=os.environ["PILOT_REVISION"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        os.environ["PILOT_MODEL"], revision=os.environ["PILOT_REVISION"], dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    # No image input: optimize every text parameter, freeze the unused vision encoder.
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad_(False)


    def collate(rows):
        batch = tokenizer.pad([{"input_ids": r["input_ids"], "attention_mask": [1]*len(r["input_ids"])}
                               for r in rows], padding=True, pad_to_multiple_of=64, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch


    class CommitCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            dist.barrier()
            if global_rank == 0:
                import modal
                modal.Volume.from_name(os.environ["PILOT_VOLUME"]).commit()
            dist.barrier()

    from transformers.trainer_utils import get_last_checkpoint
    checkpoint_dir = OUT / "checkpoints"
    resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    trainer = Trainer(model=model, args=args, train_dataset=load_from_disk("/pilot/dataset"),
                      data_collator=collate, callbacks=[CommitCheckpoint()])
    result = trainer.train(resume_from_checkpoint=resume)
    assert trainer.state.global_step == 3750, trainer.state.global_step
    trainer.save_model(str(OUT / "model"))
    if global_rank == 0:
        tokenizer.save_pretrained(str(OUT / "model"))
        assert list((OUT / "model").glob("*.safetensors")), "No consolidated weights"
        (OUT / "COMPLETE.json").write_text(json.dumps({"global_step": trainer.state.global_step,
            "epochs": 3, "data": stats, "metrics": result.metrics, "model_path": str(OUT / "model")}, indent=2))
    dist.barrier()
    dist.destroy_process_group()
