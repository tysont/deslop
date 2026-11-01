# ABOUTME: Anthropic Messages API provider.
# ABOUTME: Reads ANTHROPIC_API_KEY from the environment or .env.
from __future__ import annotations

import os
from typing import Any

from deslop.pricing import is_priced, price
from deslop.providers.base import (
    DEFAULT_MODELS,
    CachingProvider,
    Completion,
    NonRetryableError,
    ProviderError,
    Usage,
)


class AnthropicProvider(CachingProvider):
    name = "anthropic"
    default_model = DEFAULT_MODELS["anthropic"]
    cache_namespace = "anthropic"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.cache_only:
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic

        self.client = Anthropic()

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        # Current Claude models take no sampling temperature; depth is set by effort.
        import anthropic
        from anthropic.types import MessageParam, OutputConfigParam

        output_config: OutputConfigParam = {"effort": "low" if temperature < 0.5 else "medium"}
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        try:
            resp = self.client.messages.create(
                model=model,
                max_tokens=8192,
                system=system,
                output_config=output_config,
                messages=messages,
            )
        except (anthropic.AuthenticationError, anthropic.BadRequestError, TypeError) as exc:
            raise NonRetryableError(f"anthropic: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise NonRetryableError("anthropic: request refused by safety classifier")
        text = "".join(block.text for block in resp.content if block.type == "text")
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=price(model, resp.usage.input_tokens, resp.usage.output_tokens),
            calls=1,
            unpriced=0 if is_priced(model) else 1,
        )
        return Completion(text=text, model=model, provider=self.name, usage=usage)
