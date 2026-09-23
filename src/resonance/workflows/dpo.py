"""DPO+NLL training workflow (oracle positives, hard negatives).

Uses TRL 1.3.0 multi-loss API:
  loss_type:    ["sigmoid", "sft"]   — DPO + auxiliary NLL on chosen
  loss_weights: [1.0, 0.2]           — 20% NLL anchors oracle positive format

With PEFT (LoRA), ref_model=None means TRL uses the adapter-disabled base model
as the reference — no second model copy in memory.
"""

import os
import time

import hydra
import wandb
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from resonance.data.parsers.sft.dpo_pairs_parser import DPOPairsParser
from resonance.utils import convert_dtype_in_kwargs


@hydra.main(version_base=None, config_path="../configs", config_name="dpo")
def main(cfg: DictConfig) -> None:
    start_time = time.time()

    logger.info("Starting DPO+NLL training workflow (oracle positives, hard negatives)")
    logger.info(f"Model:     {cfg.model_name}")
    logger.info(f"Data path: {cfg.data_path}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cfg.get("wandb") and local_rank == 0:
        wandb.init(
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
            name=cfg.wandb.get("name"),
            tags=cfg.wandb.get("tags"),
            notes=cfg.wandb.get("notes"),
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_kwargs = cfg.tokenizer_kwargs if cfg.tokenizer_kwargs else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    model_kwargs = convert_dtype_in_kwargs(cfg.model_kwargs) if cfg.model_kwargs else {}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 128),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
    )
    logger.info(f"LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading DPO pairs from {cfg.data_path}")
    parser = DPOPairsParser(cfg.data_path)
    records = parser.load_reactions()
    logger.info(f"  {len(records)} DPO pairs loaded")

    train_dataset = Dataset.from_list(records)
    logger.info(f"  Columns: {train_dataset.column_names}")

    # ── Training config ───────────────────────────────────────────────────────
    training_args = DPOConfig(**OmegaConf.to_container(cfg.training_args, resolve=True))
    logger.info(f"loss_type={training_args.loss_type}, "
                f"loss_weights={training_args.loss_weights}, "
                f"beta={training_args.beta}, "
                f"max_length={training_args.max_length}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    # ref_model=None + peft_config → TRL uses adapter-disabled base as reference
    logger.info("Initializing DPOTrainer")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training...")
    trainer.train()

    output_dir = cfg.training_args.output_dir
    logger.info(f"Saving adapter → {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.time() - start_time
    logger.info(f"Done. {elapsed/60:.1f} min — adapter at {output_dir}")


if __name__ == "__main__":
    main()
