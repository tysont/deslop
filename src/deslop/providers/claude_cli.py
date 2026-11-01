# ABOUTME: Provider that shells out to the Claude Code CLI in headless mode (`claude -p`), using
# ABOUTME: the local login; one subprocess per call, billed as subscription calls, API cache shared.
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from deslop.providers.base import (
    DEFAULT_MODELS,
    CachingProvider,
    Completion,
    NonRetryableError,
    ProviderError,
    Usage,
)

BINARY = "claude"
TIMEOUT_SECONDS = 120.0


def _environment() -> dict[str, str]:
    """The CLI uses the local login, never the API key."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


@dataclass
class CliStatus:
    installed: bool
    logged_in: bool | None = None  # None when the CLI could not say
    version: str = ""
    detail: str = ""


def cli_status() -> CliStatus:
    """Whether the Claude Code CLI is on PATH and logged in (`claude auth status`)."""
    path = shutil.which(BINARY)
    if path is None:
        return CliStatus(installed=False, detail=f"{BINARY} is not on PATH")
    try:
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=30, env=_environment()
        ).stdout.strip()
        status = subprocess.run(
            [path, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CliStatus(installed=True, detail=f"{BINARY} did not answer: {exc}")
    try:
        data = json.loads(status.stdout or "{}")
        logged_in = bool(data.get("loggedIn")) if isinstance(data, dict) else None
    except json.JSONDecodeError:
        logged_in = None
    detail = "" if logged_in else (status.stderr or status.stdout).strip()[:200]
    return CliStatus(installed=True, logged_in=logged_in, version=version, detail=detail)


class ClaudeCliProvider(CachingProvider):
    name = "claude-cli"
    default_model = DEFAULT_MODELS["claude-cli"]
    cache_namespace = "anthropic"
    billing = "subscription"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.cache_only:
            return
        self.binary = shutil.which(BINARY)
        if self.binary is None:
            raise ProviderError(f"{BINARY} (Claude Code CLI) is not on PATH")

    def _raw_complete(self, system: str, user: str, model: str, temperature: float) -> Completion:
        cmd = [
            self.binary or BINARY,
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--system-prompt",
            system,
            "--no-session-persistence",
        ]
        try:
            run = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"claude-cli: timed out after {TIMEOUT_SECONDS:.0f}s") from exc
        except OSError as exc:
            raise NonRetryableError(f"claude-cli: could not run {BINARY}: {exc}") from exc
        if run.returncode != 0:
            raise ProviderError(
                f"claude-cli: exit {run.returncode}: {(run.stderr or run.stdout).strip()[:500]}"
            )
        try:
            data = json.loads(run.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"claude-cli: no JSON in output: {run.stdout[:200]}") from exc
        if not isinstance(data, dict):
            raise ProviderError("claude-cli: unexpected JSON output")
        if data.get("is_error"):
            raise ProviderError(f"claude-cli: {str(data.get('result', ''))[:500]}")
        text = str(data.get("result", ""))
        tokens = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(tokens.get("input_tokens", 0) or 0),
            output_tokens=int(tokens.get("output_tokens", 0) or 0),
            cost_usd=0.0,  # a subscription call; logged with cost null
            calls=1,
        )
        return Completion(text=text, model=model, provider=self.name, usage=usage)
