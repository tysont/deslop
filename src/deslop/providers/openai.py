# ABOUTME: OpenAI chat completions provider and the OpenAI-compatible endpoint variant.
# ABOUTME: Reads OPENAI_API_KEY / DESLOP_COMPATIBLE_BASE_URL (+ _API_KEY) from the environment.
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


class OpenAIProvider(CachingProvider):
    name = "openai"
    default_model = DEFAULT_MODELS["openai"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self.client = OpenAI()

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        kwargs = {}
        if not model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["temperature"] = temperature
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=price(model, in_tok, out_tok),
            calls=1,
        )
        return Completion(text=text, model=model, provider=self.name, usage=usage)


class CompatibleProvider(OpenAIProvider):
    name = "compatible"
    default_model = DEFAULT_MODELS["compatible"]

    def __init__(self, **kwargs):
        CachingProvider.__init__(self, **kwargs)
        base_url = os.environ.get("DESLOP_COMPATIBLE_BASE_URL")
        if not base_url:
            raise ProviderError("DESLOP_COMPATIBLE_BASE_URL is not set")
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url, api_key=os.environ.get("DESLOP_COMPATIBLE_API_KEY", "none")
        )
