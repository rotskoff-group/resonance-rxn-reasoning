"""
Benchmark evaluation script.

This script:
1. Instantiates a parser and prompt, loads data
2. Prepares chat messages for each sample via BenchmarkRunner
3. Dispatches inference to a pluggable InferenceBackend (see utils/inference_backends.py)
4. Saves prompts, responses, and ground-truth answers to a JSONL file
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import hydra
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from resonance.data import parsers, prompts
from resonance.data.parsers.base import Parser
from resonance.data.prompts.base import Prompt
from resonance.training.formatting_functions import build_messages, prepare_data_samples
from resonance.inference.backends import InferenceBackend
from resonance.training.instantiators import instantiate_backend


class BenchmarkRunner:
    """Orchestrates dataset loading, prompt construction, inference, and result saving.

    Args:
        parser: Instantiated Parser used to load and parse data.
        prompt: Instantiated Prompt used to generate system and task prompts.
        backend: InferenceBackend used for model inference.
        representation: Molecular representation type (e.g. "smiles", "iupac", "atom_mapped").
        system_prompt_kwargs: Extra kwargs forwarded to prompt.get_system_prompt.
        task_template_kwargs: Extra kwargs forwarded to prompt.get_task_template.
        parser_kwargs: Extra kwargs forwarded to parser.load_reactions.
    """

    def __init__(
        self,
        parser: Parser,
        prompt: Prompt,
        backend: InferenceBackend,
        representation: str = "smiles",
        system_prompt_kwargs: Optional[Dict[str, Any]] = None,
        task_template_kwargs: Optional[Dict[str, Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.parser = parser
        self.prompt = prompt
        self.backend = backend
        self.representation = representation
        self.system_prompt_kwargs = system_prompt_kwargs or {}
        self.task_template_kwargs = task_template_kwargs or {}
        self.parser_kwargs = parser_kwargs or {}

    def load_samples(self, max_samples: int = 0) -> List[Dict[str, Any]]:
        """Load data and build one sample dict per data point.

        Each dict contains:
            - system_prompt: filled system prompt string
            - task_template: unfilled task template
            - task_prompt: filled user-facing prompt
            - raw_data: original data entry from the parser
            - idx: zero-based index

        Args:
            max_samples: Cap on the number of samples. 0 means no limit.

        Returns:
            List of sample dicts ready for inference.
        """
        logger.info(f"Loading data with representation: {self.representation}")
        raw_data_list = self.parser.load_reactions(
            self.representation, max_samples=max_samples, **self.parser_kwargs
        )
        logger.info(f"Loaded {len(raw_data_list)} samples")

        if max_samples and max_samples > 0:
            raw_data_list = raw_data_list[:max_samples]
            logger.info(f"Capped to {max_samples} samples")

        return prepare_data_samples(
            parser=self.parser,
            prompt=self.prompt,
            representation=self.representation,
            raw_data_list=raw_data_list,
            system_prompt_kwargs=self.system_prompt_kwargs,
            task_template_kwargs=self.task_template_kwargs,
        )

    def run_inference(
        self,
        samples: List[Dict[str, Any]],
        save_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[str]:
        """Build messages for all samples and dispatch to the backend."""
        messages_list = [build_messages(s) for s in samples]
        return self.backend.generate(messages_list, save_callback=save_callback)

    def run(self, output_path: str, max_samples: int = 0) -> None:
        """Load data, run inference, and save results end-to-end.

        Results are written progressively to the output file as each response
        completes, so partial results are preserved if the job is interrupted.

        Args:
            output_path: Path to write the JSONL results file.
            max_samples: Cap on the number of samples. 0 means no limit.
        """
        start = time.time()
        samples = self.load_samples(max_samples=max_samples)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            buffer: List[Dict[str, Any]] = []

            def flush_buffer() -> None:
                for rec in buffer:
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                buffer.clear()

            def save_callback(idx: int, response: str) -> None:
                sample = samples[idx]
                record: Dict[str, Any] = {
                    "idx": sample["idx"],
                    "system_prompt": sample["system_prompt"],
                    "task_prompt": sample["task_prompt"],
                    "response": response,
                }
                raw = sample.get("raw_data", {})
                if isinstance(raw, dict):
                    for key in ("answer", "answer_type", "question_type", "id"):
                        if key in raw:
                            record[key] = raw[key]
                buffer.append(record)
                if len(buffer) >= 10:
                    flush_buffer()

            self.run_inference(samples, save_callback=save_callback)
            if buffer:
                flush_buffer()

        n = len(samples)
        elapsed = time.time() - start
        logger.info(f"Saved {n} results to: {output_path}")
        logger.info(f"Benchmark complete in {elapsed:.2f}s ({elapsed / 60:.2f} min)")


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the benchmark evaluation workflow."""

    # Initialize wandb if configured
    if cfg.get("wandb"):
        logger.info("Initializing wandb")
        wandb.init(
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
            name=cfg.wandb.get("name"),
            tags=cfg.wandb.get("tags"),
            notes=cfg.wandb.get("notes"),
        )
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Validate parser
    if not hasattr(parsers, cfg.parser_name):
        available = [n for n in dir(parsers) if not n.startswith("_") and n != "Parser"]
        raise ValueError(
            f"Parser '{cfg.parser_name}' not found. Available: {available}"
        )

    # Validate prompt
    if not hasattr(prompts, cfg.prompt_name):
        available = [n for n in dir(prompts) if not n.startswith("_") and n != "Prompt"]
        raise ValueError(
            f"Prompt '{cfg.prompt_name}' not found. Available: {available}"
        )

    parser_class = getattr(parsers, cfg.parser_name)
    prompt_class = getattr(prompts, cfg.prompt_name)

    if not issubclass(parser_class, Parser):
        raise TypeError(f"'{cfg.parser_name}' must inherit from Parser base class")
    if not issubclass(prompt_class, Prompt):
        raise TypeError(f"'{cfg.prompt_name}' must inherit from Prompt base class")

    parser_init_kwargs = (
        cast(
            Dict[str, Any], OmegaConf.to_container(cfg.parser_init_kwargs, resolve=True)
        )
        if cfg.get("parser_init_kwargs")
        else {}
    )

    runner = BenchmarkRunner(
        parser=parser_class(cfg.data_path, **parser_init_kwargs),
        prompt=prompt_class(),
        backend=instantiate_backend(cfg.inference),
        representation=cfg.representation,
        system_prompt_kwargs=cast(
            Dict[str, Any],
            OmegaConf.to_container(cfg.system_prompt_kwargs, resolve=True),
        )
        if cfg.get("system_prompt_kwargs")
        else None,
        task_template_kwargs=cast(
            Dict[str, Any],
            OmegaConf.to_container(cfg.task_template_kwargs, resolve=True),
        )
        if cfg.get("task_template_kwargs")
        else None,
        parser_kwargs=cast(
            Dict[str, Any], OmegaConf.to_container(cfg.parser_kwargs, resolve=True)
        )
        if cfg.parser_kwargs
        else None,
    )

    runner.run(
        output_path=cfg.output_path,
        max_samples=cfg.get("max_samples", 0),
    )

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
