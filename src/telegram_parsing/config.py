from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


load_dotenv()


@dataclass
class TelegramConfig:
    """Telegram API credentials from https://my.telegram.org/apps"""

    api_id: int = field(default_factory=lambda: int(os.environ["TELEGRAM_API_ID"]))
    api_hash: str = field(default_factory=lambda: os.environ["TELEGRAM_API_HASH"])
    session_name: str = "telegram_parsing_session"


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Supports any OpenAI-compatible API (OpenAI, OpenRouter, LMStudio, etc.)
    and local Ollama.
    """

    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )  # "openai" | "ollama"

    # OpenAI-compatible settings
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str | None = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", None)
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )

    # Ollama settings
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    )


@dataclass
class AppConfig:
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output_file: str = field(
        default_factory=lambda: os.getenv("OUTPUT_FILE", "classified_messages.json")
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "10"))
    )
    max_messages: int | None = field(
        default_factory=lambda: (
            int(v) if (v := os.getenv("MAX_MESSAGES")) else None
        )
    )
