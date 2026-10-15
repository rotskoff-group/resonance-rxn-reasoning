"""
Supervised Fine-Tuning (SFT) training script using HuggingFace Transformers and TRL.

This script loads a dataset saved with `datasets.save_to_disk` (e.g. the released S1 and S2
SFT datasets, see docs/DATA.md) and trains a model.
"""

import math
import os
import time
from typing import Any, cast

import hydra
import wandb
from datasets import Dataset, concatenate_datasets, load_from_disk
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from resonance.training.instantiators import instantiate_collator
from resonance.utils import convert_dtype_in_kwargs


class PerplexityCallback(TrainerCallback):
    """Logs perplexity (exp(eval_loss)) to wandb after each evaluation."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        metrics = kwargs.get("metrics", {})
        if "eval_loss" in metrics:
            perplexity = math.exp(metrics["eval_loss"])
            logger.info(f"Perplexity: {perplexity:.4f}")
            if wandb.run is not None:
                wandb.log({"perplexity": perplexity}, step=state.global_step)


@hydra.main(version_base=None, config_path="../configs", config_name="sft")
def main(cfg: DictConfig) -> None:
    """Main function for SFT training workflow.

    Args:
        cfg: Hydra configuration object loaded from sft.yaml
    """
    # Start timing
    start_time = time.time()

    logger.info("Starting SFT training workflow")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset path(s): {cfg.dataset_path}")

    # Initialize wandb on rank 0 only (avoids 8 separate runs under torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.get("wandb") and local_rank == 0:
        logger.info("Initializing wandb")
        wandb.init(
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
            name=cfg.wandb.get("name"),
            tags=cfg.wandb.get("tags"),
            notes=cfg.wandb.get("notes"),
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.tokenizer_name}")
    tokenizer_kwargs = cfg.tokenizer_kwargs if cfg.tokenizer_kwargs else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, **tokenizer_kwargs)
    assert tokenizer is not None, f"Failed to load tokenizer: {cfg.tokenizer_name}"

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load model
    logger.info(f"Loading model: {cfg.model_name}")
    model_kwargs = convert_dtype_in_kwargs(cfg.model_kwargs) if cfg.model_kwargs else {}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    # Load precomputed training dataset(s)
    dataset_paths = cfg.dataset_path
    if isinstance(dataset_paths, (str, ListConfig)) and not isinstance(
        dataset_paths, list
    ):
        dataset_paths = (
            [dataset_paths] if isinstance(dataset_paths, str) else list(dataset_paths)
        )

    logger.info(f"Loading {len(dataset_paths)} training shard(s)")
    train_shards = []
    for path in dataset_paths:
        logger.info(f"Loading train shard: {path}")
        shard = load_from_disk(path)
        assert isinstance(shard, Dataset), (
            f"Expected plain Dataset at {path}, got {type(shard)}"
        )
        train_shards.append(shard)

    assert len(train_shards) > 0, "No training data found"
    train_dataset = (
        concatenate_datasets(train_shards) if len(train_shards) > 1 else train_shards[0]
    )
    logger.info(f"Total training samples: {len(train_dataset)}")
    eval_dataset = None
    if cfg.get("eval_dataset_path"):
        logger.info(
            "Eval dataset path provided, will load eval dataset and compute perplexity during training"
        )
        eval_dataset_paths = cfg.get("eval_dataset_path", [])

        eval_shards = []
        for path in eval_dataset_paths:
            logger.info(f"Loading eval shard: {path}")
            shard = load_from_disk(path)
            assert isinstance(shard, Dataset), (
                f"Expected plain Dataset at {path}, got {type(shard)}"
            )
            eval_shards.append(shard)

        eval_dataset = (
            concatenate_datasets(eval_shards)
            if len(eval_shards) > 1
            else (eval_shards[0] if eval_shards else None)
        )
        if eval_dataset is not None:
            logger.info(f"Total eval samples: {len(eval_dataset)}")

    # Create data collator if specified
    data_collator = None
    if cfg.get("collator") and cfg.collator.get("type"):
        collator_kwargs = cfg.collator.kwargs if cfg.collator.kwargs else {}
        data_collator = instantiate_collator(
            collator_type=cfg.collator.type, tokenizer=tokenizer, **collator_kwargs
        )
    else:
        logger.info("No data collator specified in config")

    # Create training arguments — merge trainer_kwargs (e.g. completion_only_loss)
    # into SFTConfig so TRL picks them up correctly
    logger.info("Setting up training arguments")
    training_args = SFTConfig(**cfg.training_args)

    # Initialize SFT Trainer
    logger.info("Initializing SFT Trainer")
    callbacks: list[TrainerCallback] = (
        [PerplexityCallback()] if eval_dataset is not None else []
    )

    # ── optional LoRA ─────────────────────────────────────────────────────────
    # Gated on a `lora:` block being present, so every existing full-fine-tune config
    # behaves exactly as before. Mirrors how workflows/dpo.py has always done it.
    #
    # target_modules MATTERS ENORMOUSLY on a MoE model. Measured for
    # Qwen3-30B-A3B (128 experts, 48 layers, r=128):
    #     attention only (q/k/v/o) ->  107 M trainable,  1.3 GB fp32 Adam
    #     "all-linear"             -> 6.75 B trainable, 81.0 GB fp32 Adam
    # "all-linear" attaches an adapter to all 128 experts of every layer and so saves
    # almost nothing against the 366 GB of a full fine-tune. It is the default in
    # dpo.py and in this project's DPO configs (harmless on the dense 8B, a trap here),
    # so no default is assumed: target_modules must be given explicitly.
    peft_config = None
    if cfg.get("lora"):
        from peft import LoraConfig

        lora_cfg = cfg.lora
        if not lora_cfg.get("target_modules"):
            raise ValueError(
                "lora.target_modules must be set explicitly. On a MoE model "
                "'all-linear' attaches adapters to every expert (6.75B params on "
                "Qwen3-30B-A3B) and defeats the purpose; use the attention "
                "projections, e.g. [q_proj, k_proj, v_proj, o_proj]."
            )
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 128),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=list(lora_cfg.target_modules),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        logger.info(
            f"LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
            f"targets={peft_config.target_modules}"
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        peft_config=peft_config,
    )
    if peft_config is not None:
        cast(Any, trainer.model).print_trainable_parameters()

    # Train the model
    logger.info("Starting training...")
    resume_ckpt = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
    train_start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    logger.info(
        f"Training completed in {train_time:.2f} seconds ({train_time / 60:.2f} minutes)"
    )

    # Save the final model
    output_dir = cfg.training_args.output_dir
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # End timing and report
    end_time = time.time()
    total_time = end_time - start_time
    logger.info("SFT training workflow completed successfully!")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
