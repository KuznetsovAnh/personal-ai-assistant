"""Runtime configuration loader for the assistant application.

This module centralizes environment-driven settings such as API credentials,
model selection, response limits, and feature flags. It exposes a single
settings object for the rest of the application to consume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _load_dotenv(path: Path | None = None) -> None:
    """Nạp biến từ file .env vào os.environ (chỉ set nếu chưa có trong môi trường).

    App không phụ thuộc python-dotenv nên tự parse file .env ở gốc dự án
    để các biến ASSISTANT_*/EXCHANGERATE_*/TAVILY_*/EXA_* được đọc đúng.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ── Profile definitions ─────────────────────────────────────────────────────
# Each profile maps setting names to their override values.
# Only settings NOT already set via environment variables are applied.

_PROFILE_DEMO_FAST: Dict[str, Any] = {
    "max_response_chars": 4000,
    "llm_temperature": 0.1,
    "llm_max_completion_tokens": 8192,
}

_PROFILE_BALANCED: Dict[str, Any] = {
    "max_response_chars": 4500,
    "llm_temperature": 0.1,
    "llm_max_completion_tokens": 8192,
}

_PROFILE_ACCURATE: Dict[str, Any] = {
    "max_response_chars": 5000,
    "llm_temperature": 0.1,
    "llm_max_completion_tokens": 8192,
}

_PROFILES: Dict[str, Dict[str, Any]] = {
    "demo_fast": _PROFILE_DEMO_FAST,
    "balanced": _PROFILE_BALANCED,
    "accurate": _PROFILE_ACCURATE,
}

# Map setting names → environment variable names for "already set?" check
_SETTING_ENV_MAP: Dict[str, str] = {
    "max_response_chars": "ASSISTANT_MAX_RESPONSE_CHARS",
    "llm_temperature": "ASSISTANT_LLM_TEMPERATURE",
    "llm_max_completion_tokens": "ASSISTANT_LLM_MAX_COMPLETION_TOKENS",
}


@dataclass(slots=True)
class Settings:
    api_key: str = os.getenv("ASSISTANT_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("ASSISTANT_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("ASSISTANT_MODEL", "flash")
    runtime_profile: str = os.getenv("ASSISTANT_RUNTIME_PROFILE", "demo_fast")
    max_response_chars: int = int(os.getenv("ASSISTANT_MAX_RESPONSE_CHARS", "4000"))
    llm_temperature: float = float(os.getenv("ASSISTANT_LLM_TEMPERATURE", "0.1"))
    llm_max_completion_tokens: int = int(os.getenv("ASSISTANT_LLM_MAX_COMPLETION_TOKENS", "8192"))
    memory_max_messages: int = int(os.getenv("ASSISTANT_MEMORY_MAX_MESSAGES", "12"))
    data_dir: Path = Path(os.getenv("ASSISTANT_DATA_DIR", "data"))
    memory_file: Path = Path(os.getenv("ASSISTANT_MEMORY_FILE", "data/memory.json"))
    request_timeout: int = int(os.getenv("ASSISTANT_REQUEST_TIMEOUT", "90"))
    search_general_cache_ttl_sec: int = int(os.getenv("ASSISTANT_SEARCH_GENERAL_CACHE_TTL_SEC", "300"))
    search_time_sensitive_cache_ttl_sec: int = int(os.getenv("ASSISTANT_SEARCH_TIME_SENSITIVE_CACHE_TTL_SEC", "75"))
    search_failed_query_cooldown_sec: int = int(os.getenv("ASSISTANT_SEARCH_FAILED_QUERY_COOLDOWN_SEC", "20"))
    search_empty_evidence_cache_ttl_sec: int = int(os.getenv("ASSISTANT_SEARCH_EMPTY_EVIDENCE_CACHE_TTL_SEC", "45"))
    search_ddg_timeout_sec: int = int(os.getenv("ASSISTANT_SEARCH_DDG_TIMEOUT_SEC", "8"))
    search_ddg_failure_threshold: int = int(os.getenv("ASSISTANT_SEARCH_DDG_FAILURE_THRESHOLD", "5"))
    search_ddg_blocked_signature: str = os.getenv("ASSISTANT_SEARCH_DDG_BLOCKED_SIGNATURE", "TLS close_notify")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    exa_api_key: str = os.getenv("EXA_API_KEY", "")
    exchangerate_api_key: str = os.getenv("EXCHANGERATE_API_KEY", "")

    # ── Persistence (PostgreSQL + Redis) ──────────────────────────────────────
    # use_db = "1"/"true"/"yes" enables DB persistence; otherwise the JSON files
    # under data/ are used. Credentials come from .env (never hardcoded here).
    use_db: bool = os.getenv("ASSISTANT_USE_DB", "0").strip().lower() in ("1", "true", "yes")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5433"))
    postgres_user: str = os.getenv("POSTGRES_USER", "assistant")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_db: str = os.getenv("POSTGRES_DB", "assistant")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    def apply_runtime_profile(self) -> None:
        """Apply profile overrides for settings not already set via env vars."""
        profile_overrides = _PROFILES.get(self.runtime_profile.strip().lower())
        if not profile_overrides:
            return
        for setting_name, value in profile_overrides.items():
            env_var = _SETTING_ENV_MAP.get(setting_name, "")
            if env_var and os.getenv(env_var) is None:
                setattr(self, setting_name, value)


settings = Settings()
settings.apply_runtime_profile()
settings.ensure_directories()
