from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Sequence

from telegram_parsing.config import LLMConfig
from telegram_parsing.models import ClassifiedMessage, MessageCategory
from telegram_parsing.telegram_client import RawMessage

logger = logging.getLogger(__name__)

# ── Shared prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a message classifier for Telegram channel messages.
Note: Quiz/poll messages are handled separately. You only classify plain text messages.

For each message you receive, determine its category and extract relevant parts.

Categories:
1. **simple_qa** — The message contains a question and an answer.
2. **other** — The message does not contain a question-answer structure.

Respond with a JSON object (no markdown fences) with these fields:
- "category": one of "simple_qa", "other"
- "question": the extracted question text, or null if category is "other"
- "answers": a list with the extracted answer string (empty list if category is "other"). \
Each answer must be elaborate and detailed. Format answers using Markdown where appropriate: \
use fenced code blocks (```language ... ```) for code snippets, \
LaTeX ($...$) for formulas, **bold**/`inline code` for emphasis, \
and bullet/numbered lists for structured information.
"""

USER_PROMPT_TEMPLATE = """\
Classify the following Telegram message:

---
{text}
---

Respond ONLY with a JSON object.
"""


def _parse_llm_response(raw: str, msg: RawMessage) -> ClassifiedMessage:
    """Parse the JSON returned by the LLM into a ClassifiedMessage."""
    # Strip potential markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse LLM response for message %d, defaulting to 'other'. Response: %s",
            msg.message_id,
            raw[:200],
        )
        return ClassifiedMessage(
            message_id=msg.message_id,
            date=msg.date,
            sender=msg.sender,
            text=msg.text,
            category=MessageCategory.OTHER,
        )

    try:
        category = MessageCategory(data.get("category", "other"))
    except ValueError:
        category = MessageCategory.OTHER

    return ClassifiedMessage(
        message_id=msg.message_id,
        date=msg.date,
        sender=msg.sender,
        text=msg.text,
        category=category,
        question=data.get("question"),
        answers=data.get("answers", []),
    )


# ── Abstract base ────────────────────────────────────────────────────────────


class BaseClassifier(ABC):
    @abstractmethod
    async def classify_batch(
        self, messages: Sequence[RawMessage]
    ) -> list[ClassifiedMessage]:
        ...

    async def classify_one(self, message: RawMessage) -> ClassifiedMessage:
        results = await self.classify_batch([message])
        return results[0]


# ── OpenAI-compatible implementation ─────────────────────────────────────────


class OpenAIClassifier(BaseClassifier):
    """Uses any OpenAI-compatible API (OpenAI, OpenRouter, LMStudio, etc.)."""

    def __init__(self, config: LLMConfig) -> None:
        from openai import AsyncOpenAI

        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self._client = AsyncOpenAI(**kwargs)
        self._model = config.model

    async def classify_batch(
        self, messages: Sequence[RawMessage]
    ) -> list[ClassifiedMessage]:
        import asyncio

        tasks = [self._call_llm(msg) for msg in messages]
        return await asyncio.gather(*tasks)

    async def _call_llm(self, msg: RawMessage) -> ClassifiedMessage:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(text=msg.text),
                    },
                ],
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception:
            logger.exception("LLM call failed for message %d", msg.message_id)
            return ClassifiedMessage(
                message_id=msg.message_id,
                date=msg.date,
                sender=msg.sender,
                text=msg.text,
                category=MessageCategory.OTHER,
            )

        return _parse_llm_response(raw_text, msg)


# ── Ollama implementation ────────────────────────────────────────────────────


class OllamaClassifier(BaseClassifier):
    """Uses a local Ollama instance for classification."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError(
                "Ollama support requires the 'ollama' package. "
                "Install with: uv pip install telegram-parsing[ollama]"
            )

        self._client = AsyncClient(host=config.ollama_host)
        self._model = config.ollama_model

    async def classify_batch(
        self, messages: Sequence[RawMessage]
    ) -> list[ClassifiedMessage]:
        # Ollama runs locally — sequential to avoid overloading
        results: list[ClassifiedMessage] = []
        for msg in messages:
            results.append(await self._call_llm(msg))
        return results

    async def _call_llm(self, msg: RawMessage) -> ClassifiedMessage:
        try:
            resp = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(text=msg.text),
                    },
                ],
                options={"temperature": 0.0},
            )
            raw_text = resp["message"]["content"]
        except Exception:
            logger.exception("Ollama call failed for message %d", msg.message_id)
            return ClassifiedMessage(
                message_id=msg.message_id,
                date=msg.date,
                sender=msg.sender,
                text=msg.text,
                category=MessageCategory.OTHER,
            )

        return _parse_llm_response(raw_text, msg)


# ── Quiz pre-classifier (no LLM needed) ─────────────────────────────────────


def classify_quiz(msg: RawMessage) -> ClassifiedMessage:
    """Directly classify a Telegram quiz/poll — no LLM call required."""
    return ClassifiedMessage(
        message_id=msg.message_id,
        date=msg.date,
        sender=msg.sender,
        text=msg.text,
        category=MessageCategory.MULTI_ANSWER_QA,
        is_quiz=True,
        question=msg.poll_question,
        answers=msg.poll_options,
        correct_answer_index=msg.correct_option_index,
    )


# ── Factory ──────────────────────────────────────────────────────────────────


def create_classifier(config: LLMConfig) -> BaseClassifier:
    if config.provider == "ollama":
        return OllamaClassifier(config)
    return OpenAIClassifier(config)
