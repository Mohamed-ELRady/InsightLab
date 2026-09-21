"""Application and language-model configuration.

The command-line app reads its defaults from ``.env``. The Streamlit app can
instead create a :class:`Settings` value from the provider form in the sidebar
and pass it into the reasoning engine. Keeping those two paths behind the same
object prevents a GUI-only provider from behaving differently to a CLI one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ModelOption:
    """A useful starting model without preventing a custom model ID."""

    id: str
    label: str


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the app needs to present and route one model provider."""

    key: str
    label: str
    key_variable: str
    litellm_prefix: str
    default_model: str
    models: tuple[ModelOption, ...]
    access_note: str
    key_url: str
    free_tier: bool = False
    requires_base_url: bool = False


@dataclass(frozen=True)
class ProviderRoute:
    """One provider in the ordered failover chain, with rotating credentials."""

    provider: str
    model: str
    api_keys: tuple[str, ...] = ()
    base_url: str | None = None
    chatgpt_connected: bool = False

    @property
    def label(self) -> str:
        return PROVIDERS.get(self.provider, PROVIDERS["groq"]).label


# Model IDs are provider-native. ``Settings.model_identifier`` adds the
# LiteLLM routing prefix at the last possible moment. This matters for IDs such
# as Groq's ``openai/gpt-oss-120b`` and OpenRouter's ``openrouter/free``: the
# slash is part of the model ID, not evidence that it is already prefixed.
PROVIDERS: dict[str, ProviderSpec] = {
    "chatgpt": ProviderSpec(
        key="chatgpt",
        label="ChatGPT · via Codex (local)",
        key_variable="",
        litellm_prefix="codex",
        default_model="auto",
        models=(ModelOption("auto", "Account default"),),
        access_note="Uses your plan's Codex allowance, not API billing. Local desktop use only.",
        key_url="",
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        key_variable="GROQ_API_KEY",
        litellm_prefix="groq",
        default_model="openai/gpt-oss-120b",
        models=(
            ModelOption("openai/gpt-oss-120b", "GPT-OSS 120B"),
            ModelOption("openai/gpt-oss-20b", "GPT-OSS 20B · faster"),
            ModelOption("llama-3.3-70b-versatile", "Llama 3.3 70B"),
            ModelOption("groq/compound-mini", "Groq Compound Mini"),
        ),
        access_note="Fast hosted open models with a rate-limited free plan.",
        key_url="https://console.groq.com/keys",
        free_tier=True,
    ),
    "google": ProviderSpec(
        key="google",
        label="Google Gemini",
        key_variable="GOOGLE_API_KEY",
        litellm_prefix="gemini",
        default_model="gemini-3.7-flash",
        models=(
            ModelOption("gemini-3.7-flash", "Gemini 3.7 Flash"),
            ModelOption("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash-Lite"),
        ),
        access_note="Google AI Studio offers free-tier access to selected models.",
        key_url="https://aistudio.google.com/app/apikey",
        free_tier=True,
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter",
        key_variable="OPENROUTER_API_KEY",
        litellm_prefix="openrouter",
        default_model="openrouter/free",
        models=(
            ModelOption("openrouter/free", "Free Models Router · automatic"),
            ModelOption(
                "meta-llama/llama-3.3-70b-instruct:free",
                "Llama 3.3 70B · free",
            ),
            ModelOption("deepseek/deepseek-r1:free", "DeepSeek R1 · free"),
        ),
        access_note=(
            "One key for many models; free routes have lower limits and availability."
        ),
        key_url="https://openrouter.ai/settings/keys",
        free_tier=True,
    ),
    "cerebras": ProviderSpec(
        key="cerebras",
        label="Cerebras",
        key_variable="CEREBRAS_API_KEY",
        litellm_prefix="cerebras",
        default_model="gpt-oss-120b",
        models=(
            ModelOption("gpt-oss-120b", "GPT-OSS 120B"),
            ModelOption("zai-glm-4.7", "GLM 4.7 · preview"),
        ),
        access_note="Very fast inference with rate-limited free access.",
        key_url="https://cloud.cerebras.ai/",
        free_tier=True,
    ),
    "xai": ProviderSpec(
        key="xai",
        label="xAI · Grok",
        key_variable="XAI_API_KEY",
        litellm_prefix="xai",
        default_model="grok-4.3",
        models=(ModelOption("grok-4.3", "Grok 4.3"),),
        access_note="Direct access to Grok. API usage is billed by xAI.",
        key_url="https://console.x.ai/",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        key_variable="OPENAI_API_KEY",
        litellm_prefix="openai",
        default_model="gpt-4.1-mini",
        models=(
            ModelOption("gpt-4.1-mini", "GPT-4.1 mini"),
            ModelOption("gpt-4.1", "GPT-4.1"),
        ),
        access_note="Reliable paid models from OpenAI.",
        key_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic",
        key_variable="ANTHROPIC_API_KEY",
        litellm_prefix="anthropic",
        default_model="claude-sonnet-4-5-20250929",
        models=(
            ModelOption("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
            ModelOption("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ),
        access_note="Paid Claude API access from Anthropic.",
        key_url="https://console.anthropic.com/settings/keys",
    ),
    "custom": ProviderSpec(
        key="custom",
        label="OpenAI-compatible endpoint",
        key_variable="CUSTOM_LLM_API_KEY",
        litellm_prefix="openai",
        default_model="",
        models=(),
        access_note=(
            "Connect a local server or any service with an OpenAI-compatible API."
        ),
        key_url="",
        requires_base_url=True,
    ),
}

# Kept as public aliases for callers that used the original configuration API.
PROVIDER_KEY_VARIABLES = {
    key: provider.key_variable for key, provider in PROVIDERS.items()
}
PROVIDER_DEFAULT_MODELS = {
    key: provider.default_model for key, provider in PROVIDERS.items()
}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_keys(name: str) -> tuple[str, ...]:
    raw = os.getenv(name) or ""
    values = raw.replace("\n", ",").split(",")
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


@dataclass(frozen=True)
class Settings:
    """Resolved application settings, safe to pass into a single run."""

    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    llm_max_output_tokens: int = 1400
    api_key: str | None = None
    api_keys: tuple[str, ...] = ()
    llm_base_url: str | None = None
    chatgpt_connected: bool = False
    fallback_routes: tuple[ProviderRoute, ...] = ()

    workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "runs")
    max_upload_mb: int = 200
    sample_rows: int = 200
    offline: bool = False

    @property
    def provider(self) -> ProviderSpec:
        return PROVIDERS.get(self.llm_provider, PROVIDERS["groq"])

    @property
    def model_identifier(self) -> str:
        """Model string in the ``provider/model`` form expected by CrewAI."""
        model = self.llm_model.strip()
        if not model:
            return ""
        return f"{self.provider.litellm_prefix}/{model}"

    @property
    def configuration_issue(self) -> str | None:
        """Explain why the selected model cannot be called, if applicable."""
        if self.offline:
            return "offline mode is enabled"
        issues = [self._route_issue(route) for route in self.routes]
        if any(issue is None for issue in issues):
            return None
        return issues[0] if issues else "no model provider is configured"

    @staticmethod
    def _route_issue(route: ProviderRoute) -> str | None:
        spec = PROVIDERS.get(route.provider, PROVIDERS["groq"])
        if route.provider == "chatgpt":
            return None if route.chatgpt_connected else "sign in with ChatGPT first"
        if not route.model.strip():
            return "a model ID is required"
        if not route.api_keys:
            return f"{spec.key_variable} is not set"
        if spec.requires_base_url and not route.base_url:
            return "a base URL is required for this endpoint"
        return None

    @property
    def routes(self) -> tuple[ProviderRoute, ...]:
        """Configured providers in exact failover order, primary first."""
        primary_keys = tuple(dict.fromkeys(
            key.strip() for key in ((self.api_key or ""), *self.api_keys) if key.strip()
        ))
        primary = ProviderRoute(
            provider=self.llm_provider, model=self.llm_model,
            api_keys=primary_keys, base_url=self.llm_base_url,
            chatgpt_connected=self.chatgpt_connected,
        )
        return (primary, *self.fallback_routes)

    def candidate_settings(self) -> tuple["Settings", ...]:
        """Expand provider routes into one attempt per API key."""
        if self.offline:
            return ()
        candidates: list[Settings] = []
        for route in self.routes:
            if self._route_issue(route) is not None:
                continue
            keys: tuple[str | None, ...] = (
                (None,) if route.provider == "chatgpt" else tuple(route.api_keys)
            )
            for key in keys:
                candidates.append(replace(
                    self,
                    llm_provider=route.provider,
                    llm_model=route.model,
                    api_key=key,
                    api_keys=(),
                    llm_base_url=route.base_url,
                    chatgpt_connected=route.chatgpt_connected,
                    fallback_routes=(),
                ))
        return tuple(candidates)

    @property
    def llm_available(self) -> bool:
        """True when model calls can actually be made."""
        return self.configuration_issue is None

    def describe_llm(self) -> str:
        """Short human-readable description of the active model setup."""
        if self.offline:
            return "Offline mode · deterministic analysis"
        if issue := self.configuration_issue:
            return f"{self.provider.label} · {issue}"
        fallback_count = max(0, len(self.candidate_settings()) - 1)
        suffix = f" · {fallback_count} fallback(s)" if fallback_count else ""
        return f"{self.provider.label} · {self.llm_model}{suffix}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the environment-backed settings object once per process."""
    provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if provider not in PROVIDERS:
        provider = "groq"

    spec = PROVIDERS[provider]
    model = (os.getenv("LLM_MODEL") or "").strip() or spec.default_model
    api_key = (os.getenv(spec.key_variable) or "").strip() or None
    api_keys = _read_keys("LLM_API_KEYS")
    if api_key and api_key not in api_keys:
        api_keys = (api_key, *api_keys)
    base_url = (os.getenv("LLM_BASE_URL") or "").strip() or None

    workspace_raw = (os.getenv("INSIGHTLAB_WORKSPACE") or "data/runs").strip()
    workspace = Path(workspace_raw)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace

    return Settings(
        llm_provider=provider,
        llm_model=model,
        llm_temperature=_read_float("LLM_TEMPERATURE", 0.2),
        llm_timeout=_read_int("LLM_TIMEOUT", 60),
        llm_max_output_tokens=max(256, _read_int("LLM_MAX_OUTPUT_TOKENS", 1400)),
        api_key=api_key,
        api_keys=api_keys,
        llm_base_url=base_url,
        workspace=workspace,
        max_upload_mb=_read_int("MAX_UPLOAD_MB", 200),
        sample_rows=_read_int("SAMPLE_ROWS", 200),
        offline=_read_bool("INSIGHTLAB_OFFLINE", False),
    )


def with_model_access(
    base: Settings,
    *,
    provider: str,
    model: str,
    api_key: str | None,
    api_keys: tuple[str, ...] | list[str] | None = None,
    base_url: str | None = None,
    chatgpt_connected: bool = False,
    fallback_routes: tuple[ProviderRoute, ...] | list[ProviderRoute] | None = None,
) -> Settings:
    """Return a run-scoped copy populated from the GUI provider form."""
    if provider not in PROVIDERS:
        provider = base.llm_provider
    cleaned_keys = tuple(dict.fromkeys(
        value.strip() for value in (api_keys or ()) if value and value.strip()
    ))
    primary_key = None if provider == "chatgpt" else ((api_key or "").strip() or None)
    if primary_key and primary_key not in cleaned_keys:
        cleaned_keys = (primary_key, *cleaned_keys)
    return replace(
        base,
        llm_provider=provider,
        llm_model=model.strip(),
        api_key=primary_key,
        api_keys=() if provider == "chatgpt" else cleaned_keys,
        llm_base_url=None if provider == "chatgpt" else ((base_url or "").strip().rstrip("/") or None),
        chatgpt_connected=provider == "chatgpt" and chatgpt_connected,
        fallback_routes=tuple(fallback_routes or ()),
    )


def reset_settings_cache() -> None:
    """Drop the cached settings so a later call re-reads the environment."""
    get_settings.cache_clear()
