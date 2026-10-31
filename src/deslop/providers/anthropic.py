# ABOUTME: Anthropic Messages API provider.
# ABOUTME: Reads ANTHROPIC_API_KEY from the environment or .env.
from __future__ import annotations

import os

from deslop.providers.base import (
    DEFAULT_MODELS,
    CachingProvider,
    Completion,
    ProviderError,
    Usage,
    price,
)


class AnthropicProvider(CachingProvider):
    name = "anthropic"
    default_model = DEFAULT_MODELS["anthropic"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic

        self.client = Anthropic()

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        resp = self.client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=price(model, resp.usage.input_tokens, resp.usage.output_tokens),
            calls=1,
        )
        return Completion(text=text, model=model, provider=self.name, usage=usage)
