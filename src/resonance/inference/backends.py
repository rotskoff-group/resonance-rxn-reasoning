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
from contextlib import nullcontext
from typing import Any

import torch
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
        include_reasoning: bool = True,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.request_kwargs = request_kwargs or {}
        self.concurrency = concurrency
        self.include_reasoning = include_reasoning

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
                    if not self.include_reasoning:
                        return content
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


class SharedModelBackend(InferenceBackend):
    """Judge inference on the shared policy model with LoRA adapters optionally disabled.

    Avoids loading a second copy of the base model. The policy model (a PeftModel)
    is shared across training and reward computation. By default adapters are
    disabled so the judge runs on the frozen base weights, giving stable reward
    signals that are independent of the adapters being trained.

    Usage
    -----
    1. Construct with ``SharedModelBackend(sampling_kwargs={...}, use_lora=False)``.
    2. After ``GRPOTrainer`` is initialised, call ``backend.bind(trainer.model, tokenizer)``.
    3. ``generate()`` will then run on the shared model.

    Notes
    -----
    - Not compatible with FSDP-sharded models; use single-GPU or DDP.
    - The full completion batch is processed in one ``model.generate()`` call;
      reduce ``num_generations`` in GRPOConfig if you hit OOM.

    Args:
        sampling_kwargs: Forwarded verbatim to ``model.generate()``
                         (e.g. max_new_tokens, temperature, do_sample).
        use_lora:        If False (default), adapters are disabled during generation
                         so the model acts as the base (judge). If True, adapters
                         remain active (critic mode).
    """

    def __init__(
        self,
        sampling_kwargs: dict[str, Any] | None = None,
        use_lora: bool = False,
    ):
        self._model = None
        self._tokenizer = None
        self._sampling_kwargs: dict[str, Any] = sampling_kwargs or {
            "max_new_tokens": 512
        }
        self._use_lora = use_lora

    def bind(self, model: Any, tokenizer: Any) -> None:
        """Attach the shared policy model and tokenizer after trainer initialisation."""
        self._model = model
        self._tokenizer = tokenizer

    @property
    def is_bound(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def generate(
        self,
        messages_list: list[list[dict[str, str]]],
        save_callback: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        if not self.is_bound:
            raise RuntimeError(
                "SharedModelBackend has no model bound. "
                "Call backend.bind(trainer.model, tokenizer) after GRPOTrainer init."
            )

        tokenizer = self._tokenizer
        assert tokenizer is not None

        # gen_model: the outermost model (preserves FSDP/DDP collective ops in generate())
        gen_model = self._model
        assert gen_model is not None

        # adapter_model: the inner PeftModel that owns disable_adapters(); we peel
        # only wrapper layers (DDP, FSDP) that don't have disable_adapters themselves.
        adapter_model = gen_model
        while not hasattr(adapter_model, "disable_adapters") and hasattr(
            adapter_model, "module"
        ):
            adapter_model = adapter_model.module

        # Tokenize each message list and left-pad to a common length
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        encodings = [
            tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            for msgs in messages_list
        ]
        max_len = max(e.shape[1] for e in encodings)
        padded_ids, attn_masks = [], []
        for e in encodings:
            pad_len = max_len - e.shape[1]
            if pad_len:
                pad = e.new_full((1, pad_len), pad_id)
                padded_ids.append(torch.cat([pad, e], dim=1))
                attn_masks.append(
                    torch.cat(
                        [
                            torch.zeros(1, pad_len, dtype=torch.long),
                            torch.ones(1, e.shape[1], dtype=torch.long),
                        ],
                        dim=1,
                    )
                )
            else:
                padded_ids.append(e)
                attn_masks.append(torch.ones(1, max_len, dtype=torch.long))

        device = next(adapter_model.parameters()).device
        input_ids = torch.cat(padded_ids).to(device)
        attention_mask = torch.cat(attn_masks).to(device)

        gen_kwargs = dict(self._sampling_kwargs)
        gen_kwargs.setdefault("pad_token_id", pad_id)

        was_training = gen_model.training
        gen_model.eval()

        # Disable adapters on the inner PeftModel for judge (base-model) inference.
        # generate() still goes through gen_model so FSDP collective ops are preserved.
        disable_ctx = (
            adapter_model.disable_adapters()
            if (not self._use_lora and hasattr(adapter_model, "disable_adapters"))
            else nullcontext()
        )
        try:
            with torch.no_grad(), disable_ctx:
                output_ids = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
        finally:
            if was_training:
                gen_model.train()

        prompt_len = input_ids.shape[1]
        responses = []
        for i, seq in enumerate(output_ids):
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            responses.append(text)
            if save_callback:
                save_callback(i, text)

        return responses
