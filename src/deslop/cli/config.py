# ABOUTME: CLI configuration file support: every option can come from deslop.yaml, with
# ABOUTME: explicit flags overriding per-command sections, which override top-level keys.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_FILES = ("deslop.yaml", "deslop.yml", ".deslop.yaml")
_MISSING = object()


class ConfigError(Exception):
    pass


def home() -> Path:
    return Path(os.environ.get("DESLOP_HOME", Path.home() / ".deslop"))


class Config:
    """Resolved settings. `data` is the parsed yaml: top-level keys apply to every command;
    a key named after a command holds that command's overrides."""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self.data = _normalize(data or {})
        self.path = path

    def get(self, command: str, key: str, value: Any, *, default: Any = None) -> Any:
        """The explicit CLI value if given, else the command section, else the top-level
        key, else the default."""
        if value is not None:
            return value
        key = key.replace("-", "_")
        section = self.data.get(command.replace("-", "_"))
        if isinstance(section, dict) and section.get(key, _MISSING) is not _MISSING:
            return section[key]
        top = self.data.get(key, _MISSING)
        if top is not _MISSING and not isinstance(top, dict):
            return top
        return default


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        k = str(key).replace("-", "_")
        out[k] = _normalize(value) if isinstance(value, dict) else value
    return out


def find_config() -> Path | None:
    for name in PROJECT_FILES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    candidate = home() / "config.yaml"
    return candidate if candidate.is_file() else None


def load_config(path: Path | None) -> Config:
    """Load `path`, or the first of ./deslop.yaml, ./deslop.yml, ./.deslop.yaml,
    $DESLOP_HOME/config.yaml. An explicit path that does not exist is an error."""
    if path is None:
        path = find_config()
        if path is None:
            return Config()
    elif not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return Config(data, path)
