"""Configuration helpers for API clients and environment management."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, TypeVar
from inspect import Parameter, signature

try:  # Optional dependency for local development.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only triggered when dotenv missing.
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()


class MissingEnvironmentVariable(RuntimeError):
    """Raised when a required environment variable has not been configured."""


_T = TypeVar("_T")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvironmentVariable(
            f"Missing environment variable: {name}. "
            "Create a .env file (see .env.example) or export the variable before running the scripts."
        )
    return value


def _cached_client(factory: Callable[..., _T], env_name: str) -> _T:
    key = _require_env(env_name)

    sig = signature(factory)
    parameters = sig.parameters

    accepts_keyword = False

    api_key_param = parameters.get("api_key")
    if api_key_param is not None:
        accepts_keyword = api_key_param.kind in (
            Parameter.KEYWORD_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        )
    elif any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values()):
        accepts_keyword = True

    if accepts_keyword:
        return factory(api_key=key)


@lru_cache(maxsize=1)
def get_exa_client():
    """Return a cached Exa client configured from ``EXA_API_KEY``."""
    from exa_py import Exa

    return _cached_client(Exa, "EXA_API_KEY")


@lru_cache(maxsize=1)
def get_openai_client():
    """Return a cached OpenAI client configured from ``OPENAI_API_KEY``."""
    from openai import OpenAI

    return _cached_client(OpenAI, "OPENAI_API_KEY")


__all__ = ["MissingEnvironmentVariable", "get_exa_client", "get_openai_client"]
