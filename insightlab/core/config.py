"""Application configuration loaded from the environment.

Every tunable value lives here so that no other module has to reach into
``os.environ`` directly. Values come from a git-ignored ``.env`` file; see
``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


#: API key environment variable for each supported provider.
PROVIDER_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

#: Default model for each supported provider, used when LLM_MODEL is unset.
PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "google": "gemini/gemini-1.5-flash",
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


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    api_key: str | None = None

    workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "runs")
    max_upload_mb: int = 200
    sample_rows: int = 200
    offline: bool = False

    @property
    def model_identifier(self) -> str:
        """Model string in the ``provider/model`` form expected by CrewAI.

        CrewAI routes calls through LiteLLM, which infers the provider from a
        prefix. OpenAI model names are accepted bare, everything else is
        prefixed unless the caller already did so.
        """
        if "/" in self.llm_model:
            return self.llm_model
        if self.llm_provider == "openai":
            return self.llm_model
        return f"{self.llm_provider}/{self.llm_model}"

    @property
    def llm_available(self) -> bool:
        """True when model calls can actually be made."""
        return not self.offline and bool(self.api_key)

    def describe_llm(self) -> str:
        """Short human-readable description of the active model setup."""
        if self.offline:
            return "offline mode (deterministic heuristics, no model calls)"
        if not self.api_key:
            key_var = PROVIDER_KEY_VARIABLES.get(self.llm_provider, "the provider API key")
            return f"no credentials ({key_var} is not set) - falling back to heuristics"
        return f"{self.llm_provider}:{self.llm_model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the settings object once per process."""
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
    if provider not in PROVIDER_KEY_VARIABLES:
        provider = "openai"

    model = (os.getenv("LLM_MODEL") or "").strip() or PROVIDER_DEFAULT_MODELS[provider]
    api_key = (os.getenv(PROVIDER_KEY_VARIABLES[provider]) or "").strip() or None

    workspace_raw = (os.getenv("INSIGHTLAB_WORKSPACE") or "data/runs").strip()
    workspace = Path(workspace_raw)
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace

    return Settings(
        llm_provider=provider,
        llm_model=model,
        llm_temperature=_read_float("LLM_TEMPERATURE", 0.2),
        llm_timeout=_read_int("LLM_TIMEOUT", 60),
        api_key=api_key,
        workspace=workspace,
        max_upload_mb=_read_int("MAX_UPLOAD_MB", 200),
        sample_rows=_read_int("SAMPLE_ROWS", 200),
        offline=_read_bool("INSIGHTLAB_OFFLINE", False),
    )


def reset_settings_cache() -> None:
    """Drop the cached settings so a later call re-reads the environment."""
    get_settings.cache_clear()
