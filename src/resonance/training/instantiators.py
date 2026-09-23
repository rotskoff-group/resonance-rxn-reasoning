"""Utility functions for instantiating objects from configuration."""

import os
from typing import Any, Dict, Optional, cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)

from resonance.inference.backends import (
    APIBackend,
    InferenceBackend,
    SharedModelBackend,
    VLLMBackend,
)


def instantiate_collator(
    collator_type: str, tokenizer: PreTrainedTokenizerBase, **kwargs: Any
) -> Optional[Any]:
    """Instantiate a data collator based on type string.

    Args:
        collator_type: Name of the collator class (e.g., 'DataCollatorForLanguageModeling')
        tokenizer: The tokenizer to use with the collator
        **kwargs: Additional keyword arguments to pass to the collator constructor

    Returns:
        Instantiated collator object, or None if type is unrecognized

    Example:
        >>> collator = instantiate_collator(
        ...     "DataCollatorForLanguageModeling",
        ...     tokenizer=tokenizer,
        ...     mlm=False
        ... )
    """
    collator_map = {
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
    }

    if collator_type not in collator_map:
        logger.warning(
            f"Unknown collator type: {collator_type}. "
            f"Available types: {list(collator_map.keys())}"
        )
        return None

    collator_class = collator_map[collator_type]
    logger.info(f"Instantiating {collator_type}")
    return collator_class(tokenizer=tokenizer, **kwargs)


def instantiate_backend(cfg: DictConfig) -> InferenceBackend:
    """Instantiate an InferenceBackend from a Hydra inference config block.

    Args:
        cfg: The ``inference`` sub-config. Must contain ``mode`` and ``model_name``.
             Additional keys are backend-specific (see VLLMBackend / APIBackend).

    Returns:
        An InferenceBackend instance ready for generation.

    Raises:
        ValueError: If the mode is unrecognized or a required API key is missing.
    """

    def _as_dict(key: str) -> Optional[Dict[str, Any]]:
        val = cfg.get(key)
        return (
            cast(Dict[str, Any], OmegaConf.to_container(val, resolve=True))
            if val
            else None
        )

    mode = cfg.mode

    if mode == "vllm":
        return VLLMBackend(
            model_name=cfg.model_name,
            vllm_kwargs=_as_dict("vllm_kwargs"),
            sampling_params_kwargs=_as_dict("sampling_params"),
            lora_path=cfg.get("lora_path"),
        )

    if mode == "api":
        api_key = cfg.get("api_key")
        if not api_key:
            api_base = cfg.get("api_base") or ""
            model_name = cfg.model_name or ""
            if "anthropic" in api_base or "claude" in model_name:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif "google" in api_base or "gemini" in model_name:
                api_key = os.environ.get("GEMINI_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set cfg.inference.api_key, or export ANTHROPIC_API_KEY / "
                "OPENAI_API_KEY / GEMINI_API_KEY in your environment."
            )
        return APIBackend(
            model_name=cfg.model_name,
            api_key=api_key,
            api_base=cfg.get("api_base"),
            request_kwargs=_as_dict("request_kwargs"),
            concurrency=cfg.get("concurrency", 32),
        )

    if mode == "shared":
        return SharedModelBackend(
            sampling_kwargs=_as_dict("sampling_kwargs"),
            use_lora=bool(cfg.get("use_lora", False)),
        )

    raise ValueError(
        f"Unknown inference mode '{mode}'. Must be one of: 'vllm', 'api', 'shared'."
    )
