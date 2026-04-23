"""Configuration for GreenCompute SDK."""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


GREENCOMPUTE_DIRNAME = ".greencompute"
CONFIG_FILENAME = "config.ini"


@dataclass
class Config:
    """SDK configuration."""

    api_base_url: str
    api_key: str | None


def default_config_path() -> Path:
    override = os.getenv("GREENCOMPUTE_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / GREENCOMPUTE_DIRNAME / CONFIG_FILENAME


def load_file_config(path: Path | None = None) -> Config:
    config_path = path or default_config_path()
    api_base_url = "http://127.0.0.1:8000"
    api_key: str | None = None
    if config_path.exists():
        parser = ConfigParser()
        parser.read(config_path)
        if parser.has_section("api"):
            api_base_url = parser.get("api", "base_url", fallback=api_base_url)
            api_key = parser.get("api", "api_key", fallback=api_key) or api_key
    return Config(api_base_url=api_base_url.rstrip("/"), api_key=api_key)


def save_config(*, api_base_url: str | None = None, api_key: str | None = None, path: Path | None = None) -> Config:
    config_path = path or default_config_path()
    current = load_file_config(config_path)
    parser = ConfigParser()
    parser["api"] = {
        "base_url": (api_base_url or current.api_base_url).rstrip("/"),
        "api_key": api_key if api_key is not None else (current.api_key or ""),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as outfile:
        parser.write(outfile)
    return load_file_config(config_path)


def init_config(*, api_base_url: str | None = None, api_key: str | None = None, path: Path | None = None) -> Config:
    return save_config(api_base_url=api_base_url, api_key=api_key, path=path)


def unset_config(*, api_base_url: bool = False, api_key: bool = False, path: Path | None = None) -> Config:
    config_path = path or default_config_path()
    current = load_file_config(config_path)
    parser = ConfigParser()
    parser["api"] = {
        "base_url": ("http://127.0.0.1:8000" if api_base_url else current.api_base_url).rstrip("/"),
        "api_key": "" if api_key else (current.api_key or ""),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as outfile:
        parser.write(outfile)
    return load_file_config(config_path)


def mask_secret(value: str | None) -> str | None:
    if value is None or value == "":
        return value
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:4]}...{value[-2:]}"


def get_config() -> Config:
    """Load config from file and env vars with env taking precedence."""
    config = load_file_config()
    api_base_url = os.getenv("GREENCOMPUTE_API_URL", config.api_base_url)
    api_key = os.getenv("GREENCOMPUTE_API_KEY", config.api_key or "") or config.api_key
    return Config(api_base_url=api_base_url.rstrip("/"), api_key=api_key)
