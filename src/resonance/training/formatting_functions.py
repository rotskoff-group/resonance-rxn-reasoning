"""
Formatting functions for converting dataset samples to model input format.

Data preparation:
  prepare_data_samples  — loads raw data through parser+prompt into sample dicts

Inference functions (return a chat message list):
  build_messages        — system + user messages for chat-completion APIs
"""

from typing import Any

from loguru import logger
from tqdm import tqdm

from resonance.data.parsers.base import Parser
from resonance.data.prompts.base import Prompt


def prepare_data_samples(
    parser: Parser,
    prompt: Prompt,
    representation: str,
    raw_data_list: list[Any],
    system_prompt_kwargs: dict[str, Any] | None = None,
    task_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of sample dicts from raw data.

    Args:
        parser: Parser instance for parsing reactions.
        prompt: Prompt instance for generating prompts.
        representation: Representation type (only 'smiles' is supported).
        raw_data_list: Pre-loaded list of raw reaction data.
        system_prompt_kwargs: Keyword arguments forwarded to prompt.get_system_prompt.
        task_template_kwargs: Keyword arguments forwarded to prompt.get_task_template.

    Returns:
        List of dicts, each containing system_prompt, task_template, task_prompt,
        raw_data, and idx.
    """
    system_prompt_kwargs = system_prompt_kwargs or {}
    task_template_kwargs = task_template_kwargs or {}

    system_prompt = prompt.get_system_prompt(**system_prompt_kwargs)
    task_template = prompt.get_task_template(
        representation=representation, **task_template_kwargs
    )
    samples = []
    logger.info("Preparing data samples...")
    for idx, raw_data in enumerate(tqdm(raw_data_list)):
        task_prompt = parser.parse_reaction(task_template, raw_data)
        samples.append(
            {
                "system_prompt": system_prompt,
                "task_template": task_template,
                "task_prompt": task_prompt,
                "raw_data": raw_data,
                "idx": idx,
            }
        )
    return samples


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """System (if any) and user messages for one sample."""
    messages: list[dict[str, str]] = []
    if sample["system_prompt"]:
        messages.append({"role": "system", "content": sample["system_prompt"]})
    messages.append({"role": "user", "content": sample["task_prompt"]})
    return messages
