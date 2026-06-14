import os
from functools import lru_cache
from typing import Any, Callable

from groq import Groq
from langchain_groq import ChatGroq
from pydantic import SecretStr


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
]

FALLBACK_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "rate_limit_exceeded",
    "tokens per minute",
    "tpm",
    "request too large",
    "too many requests",
    "resource_exhausted",
    "service tier",
    "413",
    "429",
    "model_not_found",
    "model not found",
    "does not support",
    "unsupported model",
    "decommissioned",
)

NON_CHAT_MODEL_MARKERS = (
    "whisper",
    "prompt-guard",
    "safeguard",
    "orpheus",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@lru_cache(maxsize=4)
def _available_groq_models(api_key: str | None) -> tuple[str, ...]:
    if not api_key:
        return ()

    try:
        client = Groq(api_key=api_key, timeout=5.0)
        models = client.models.list()
    except Exception as exc:
        print(f"Could not auto-load Groq model list. Using configured fallbacks. Reason: {exc}")
        return ()

    model_ids = []
    for model in getattr(models, "data", []) or []:
        model_id = getattr(model, "id", None)
        active = getattr(model, "active", True)
        if model_id and active is not False:
            model_ids.append(str(model_id))

    return tuple(sorted(model_ids))


def is_fallback_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in FALLBACK_ERROR_MARKERS)


def available_groq_models(api_key_env_names: tuple[str, ...] = ("GROQ_API_KEY",)) -> list[str]:
    return list(_available_groq_models(_first_env(api_key_env_names)))


def _chat_candidate_models(models: list[str]) -> list[str]:
    return [
        model
        for model in models
        if not any(marker in model.lower() for marker in NON_CHAT_MODEL_MARKERS)
    ]


def resolve_groq_models(
    *,
    model_env_name: str,
    fallback_env_name: str | None = None,
    api_key_env_names: tuple[str, ...] = ("GROQ_API_KEY",),
    base_model_env_name: str = "GROQ_MODEL",
    base_fallback_env_name: str = "GROQ_MODEL_FALLBACKS",
    default_model: str = DEFAULT_GROQ_MODEL,
) -> list[str]:
    primary = os.getenv(model_env_name) or os.getenv(base_model_env_name) or default_model
    fallback_models = (
        _split_models(os.getenv(fallback_env_name)) if fallback_env_name else []
    ) or _split_models(os.getenv(base_fallback_env_name))

    models = [primary, *fallback_models, *DEFAULT_GROQ_MODEL_FALLBACKS]

    if _env_bool("GROQ_AUTO_MODEL_FALLBACKS", True):
        models.extend(_chat_candidate_models(available_groq_models(api_key_env_names)))

    return _dedupe(models)


class GroqFallbackRunnable:
    def __init__(
        self,
        chat: "GroqFallbackChat",
        make_runnable: Callable[[ChatGroq], Any],
    ) -> None:
        self.chat = chat
        self.make_runnable = make_runnable

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.chat._invoke_with_fallback(
            lambda llm: self.make_runnable(llm).invoke(*args, **kwargs)
        )


class GroqFallbackChat:
    def __init__(
        self,
        *,
        model_env_name: str,
        fallback_env_name: str | None = None,
        api_key_env_names: tuple[str, ...] = ("GROQ_API_KEY",),
        temperature: float | None = None,
        max_tokens: int | None = None,
        temperature_env_name: str = "GROQ_TEMPERATURE",
        max_tokens_env_name: str = "GROQ_MAX_TOKENS",
    ) -> None:
        self.api_key = _first_env(api_key_env_names)
        self.models = resolve_groq_models(
            model_env_name=model_env_name,
            fallback_env_name=fallback_env_name,
            api_key_env_names=api_key_env_names,
        )
        self.temperature = (
            temperature if temperature is not None else _env_float(temperature_env_name, 0.1)
        )
        self.max_tokens = max_tokens if max_tokens is not None else _env_int(max_tokens_env_name, 4096)

    @property
    def model(self) -> str:
        return self.models[0]

    def _make_llm(self, model: str) -> ChatGroq:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            kwargs["api_key"] = SecretStr(self.api_key)
        return ChatGroq(**kwargs)

    def _invoke_with_fallback(self, call: Callable[[ChatGroq], Any]) -> Any:
        last_error: Exception | None = None

        for index, model in enumerate(self.models):
            try:
                if index > 0:
                    print(f"Trying Groq fallback model: {model}")
                return call(self._make_llm(model))
            except Exception as exc:
                last_error = exc
                if not is_fallback_error(exc) or index >= len(self.models) - 1:
                    raise
                print(f"Groq model {model} failed with a retryable error: {exc}")

        if last_error:
            raise last_error
        raise RuntimeError("No Groq models are configured.")

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke_with_fallback(lambda llm: llm.invoke(*args, **kwargs))

    def with_structured_output(self, *args: Any, **kwargs: Any) -> GroqFallbackRunnable:
        return GroqFallbackRunnable(
            self,
            lambda llm: llm.with_structured_output(*args, **kwargs),
        )


def make_groq_llm(
    *,
    model_env_name: str,
    fallback_env_name: str | None = None,
    api_key_env_names: tuple[str, ...] = ("GROQ_API_KEY",),
    temperature: float | None = None,
    max_tokens: int | None = None,
    temperature_env_name: str = "GROQ_TEMPERATURE",
    max_tokens_env_name: str = "GROQ_MAX_TOKENS",
) -> GroqFallbackChat:
    return GroqFallbackChat(
        model_env_name=model_env_name,
        fallback_env_name=fallback_env_name,
        api_key_env_names=api_key_env_names,
        temperature=temperature,
        max_tokens=max_tokens,
        temperature_env_name=temperature_env_name,
        max_tokens_env_name=max_tokens_env_name,
    )
