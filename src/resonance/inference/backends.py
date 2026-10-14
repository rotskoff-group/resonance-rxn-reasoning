"""Inference backend classes for running model inference during benchmarking.

Two backends are provided:

- VLLMBackend  — uses the vllm Python library for in-process inference on local hardware.
                 Suited for open-source models.
- APIBackend   — calls any OpenAI-compatible /v1/chat/completions endpoint.
                 Uses async concurrent requests for throughput.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, RateLimitError
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class InferenceBackend(ABC):
    """Abstract base class for all inference backends."""

    @abstractmethod
    def generate(
        self,
        messages_list: list[list[dict[str, str]]],
        save_callback: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        """Run inference over a batch of chat message lists.

        Args:
            messages_list: Each element is a list of role/content dicts representing
                           one conversation (e.g. [{"role": "system", ...}, {"role": "user", ...}]).
            save_callback: Optional callable invoked with (original_index, response) as each
                           result completes, enabling progressive saving.

        Returns:
            List of response strings aligned with messages_list.
        """


class VLLMBackend(InferenceBackend):
    """Local inference via the vllm Python library.

    Loads the model in-process and calls llm.chat() for batched generation.

    Args:
        model_name: HuggingFace model ID or local path (base model).
        vllm_kwargs: Forwarded to vllm.LLM (e.g. tensor_parallel_size, dtype, max_model_len).
        sampling_params_kwargs: Forwarded to vllm.SamplingParams (e.g. temperature, max_tokens).
        lora_path: Optional local path to a PEFT LoRA adapter directory. When set,
                   ``enable_lora=True`` is added to vllm_kwargs and a LoRARequest is
                   passed to every chat() call.
    """

    def __init__(
        self,
        model_name: str,
        vllm_kwargs: dict[str, Any] | None = None,
        sampling_params_kwargs: dict[str, Any] | None = None,
        lora_path: str | None = None,
    ):
        self.model_name = model_name
        self.vllm_kwargs = vllm_kwargs or {}
        self.sampling_params_kwargs = sampling_params_kwargs or {}
        self.lora_path = lora_path
        self._llm: Any | None = None
        self._lora_request: Any | None = None

    def _load(self) -> None:
        if self._llm is None:
            kwargs = dict(self.vllm_kwargs)
            if self.lora_path:
                kwargs["enable_lora"] = True
            logger.info(f"Loading vllm model: {self.model_name}")
            self._llm = LLM(model=self.model_name, **kwargs)
            if self.lora_path:
                self._lora_request = LoRARequest("adapter", 1, self.lora_path)
                logger.info(f"LoRA adapter loaded from: {self.lora_path}")

    def generate(
        self,
        messages_list: list[list[dict[str, str]]],
        save_callback: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        self._load()
        assert self._llm is not None
        sampling_params = SamplingParams(**self.sampling_params_kwargs)
        logger.info(f"Running vllm inference on {len(messages_list)} samples...")
        chat_kwargs: dict[str, Any] = {"sampling_params": sampling_params}
        if self._lora_request is not None:
            chat_kwargs["lora_request"] = self._lora_request
        outputs = self._llm.chat(messages_list, **chat_kwargs)
        responses = [o.outputs[0].text for o in outputs]
        if save_callback:
            for idx, response in enumerate(responses):
                save_callback(idx, response)
        return responses


class APIBackend(InferenceBackend):
    """Remote inference via any OpenAI-compatible chat completions endpoint.

    A single backend that works with:
    - OpenAI          (api_base=None, api_key from OPENAI_API_KEY)
    - Anthropic       (api_base="https://api.anthropic.com/v1", api_key from ANTHROPIC_API_KEY)
    - A local vllm server started with `vllm serve` (api_base="http://localhost:8000/v1")

    Args:
        model_name: Model identifier understood by the provider.
        api_key: API key (or any placeholder string for local servers that don't require one).
        api_base: Base URL override. Defaults to the OpenAI endpoint when None.
        request_kwargs: Forwarded to client.chat.completions.create (e.g. temperature, max_tokens).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str | None = None,
        request_kwargs: dict[str, Any] | None = None,
        concurrency: int = 32,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.request_kwargs = request_kwargs or {}
        self.concurrency = concurrency

    async def _call_one(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        messages: list[dict[str, str]],
    ) -> str:
        max_retries = 8
        base_delay = 1.0

        async with semaphore:
            for attempt in range(max_retries):
                try:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **self.request_kwargs,
                    )
                    msg = completion.choices[0].message
                    content = msg.content or ""
                    # Thinking models split their output: the chain of thought arrives in
                    # `reasoning_content` (some gateways call it `reasoning`) and only the
                    # final answer is in `content`. Returning content alone DISCARDED the
                    # CoT -- which we pay for either way, since providers bill reasoning as
                    # completion tokens ($15/MTok on the Kimi endpoint, ~8,200 tok/item
                    # against a 52-character answer).
                    #
                    # Wrapping it in <think></think> and prepending makes an API model's
                    # response structurally identical to a locally-served Qwen3, which emits
                    # those tags natively -- so the same scorers and trace tooling work on
                    # both. Safe for extraction: extract_forward_rxn matches <answer> first,
                    # and its truncation guard only fires on an UNCLOSED <think>, which this
                    # never produces.
                    reasoning = (getattr(msg, "reasoning_content", None)
                                 or getattr(msg, "reasoning", None) or "")
                    if reasoning:
                        return f"<think>\n{reasoning}\n</think>\n\n{content}"
                    return content
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
        return ""  # unreachable

    async def _call_one_indexed(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        messages: list[dict[str, str]],
        idx: int,
    ) -> tuple[int, str]:
        result = await self._call_one(client, semaphore, messages)
        return idx, result

    async def _generate_all(
        self,
        messages_list: list[list[dict[str, str]]],
        save_callback: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        client = AsyncOpenAI(**client_kwargs)

        semaphore = asyncio.Semaphore(self.concurrency)
        logger.info(
            f"Running async API inference on {len(messages_list)} samples "
            f"(model: {self.model_name}, concurrency: {self.concurrency})..."
        )

        results: list[str | None] = [None] * len(messages_list)
        tasks = [
            self._call_one_indexed(client, semaphore, m, i)
            for i, m in enumerate(messages_list)
        ]
        for coro in asyncio.as_completed(tasks):
            idx, response = await coro
            results[idx] = response
            if save_callback:
                save_callback(idx, response)

        return [r or "" for r in results]

    def generate(
        self,
        messages_list: list[list[dict[str, str]]],
        save_callback: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        return asyncio.run(self._generate_all(messages_list, save_callback))
