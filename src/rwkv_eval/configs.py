from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit


try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class RwkvModel:
    arch_version: Literal["rwkv7", "rwkv7a", "rwkv7b"]
    data_version: str
    param_size: Literal["1.5b", "2.9b", "7.2b", "13.3b"]
    ctx_len: int


@dataclass
class SamplingConfig:
    max_generated_tokens: int
    temp: float
    top_k: int
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    penalty_decay: float
    seed: int


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    url: str
    api_key: str
    model_name: str
    max_num_seqs: int
    ctx_len: int = 4096


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    field: Literal[
        "knowledge",
        "reasoning",
        "maths",
        "coding",
        "instruction_following",
        "agentic",
        "vision",
    ]
    selector: str


def _read_toml(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid TOML {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a TOML table")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _api_key(value: Any) -> str:
    value = _string(value, "api_key")
    match = re.fullmatch(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
    if match:
        value = os.environ.get(match.group(1), "")
    if not value:
        raise ValueError(
            f"missing environment variable {match.group(1)} for api_key" if match else "api_key is empty"
        )
    return value


def _url(value: Any) -> str:
    value = _string(value, "url")
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"invalid HTTP endpoint URL: {value}")
    return value.rstrip("/")


def read_models(path: str | Path) -> tuple[ModelEndpoint, ...]:
    raw = _read_toml(path)
    entries = raw.get("model") or raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected one or more [[model]] entries")
    models = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: model entry {index} must be a table")
        try:
            models.append(
                ModelEndpoint(
                    _url(entry["url"]),
                    _api_key(entry["api_key"]),
                    _string(entry["model_name"], "model_name"),
                    _positive_int(entry["max_num_seqs"], "max_num_seqs"),
                    _positive_int(entry.get("ctx_len", 4096), "ctx_len"),
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}: model entry {index} is missing {error.args[0]}") from error
    return tuple(models)


def read_benchmarks(path: str | Path) -> tuple[BenchmarkSpec, ...]:
    allowed = {"knowledge", "reasoning", "maths", "coding", "instruction_following", "agentic", "vision"}
    specs, seen = [], set()
    for field, selectors in _read_toml(path).items():
        if field not in allowed or not isinstance(selectors, list):
            raise ValueError(f"{path}: invalid benchmark field {field}")
        for selector in selectors:
            selector = _string(selector, "benchmark selector")
            if selector in seen:
                raise ValueError(f"{path}: duplicate benchmark selector {selector}")
            seen.add(selector)
            specs.append(BenchmarkSpec(field, selector))
    if not specs:
        raise ValueError(f"{path}: no benchmarks configured")
    return tuple(specs)
