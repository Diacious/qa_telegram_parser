from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageCategory(str, Enum):
    """Classification categories for a Telegram message."""

    SIMPLE_QA = "simple_qa"  # Single question with a single answer (text message)
    MULTI_ANSWER_QA = "multi_answer_qa"  # Telegram quiz/poll with multiple options
    OTHER = "other"  # No Q&A structure


@dataclass
class ClassifiedMessage:
    """A single Telegram message with its classification."""

    message_id: int
    date: str  # ISO-8601
    sender: str | None
    text: str
    category: MessageCategory
    is_quiz: bool = False  # True if this came from a Telegram poll/quiz
    question: str | None = None  # Extracted question text (if applicable)
    answers: list[str] = field(default_factory=list)  # Extracted answer(s)
    correct_answer_index: int | None = None  # For quizzes: index of correct answer
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "date": self.date,
            "sender": self.sender,
            "text": self.text,
            "category": self.category.value,
            "is_quiz": self.is_quiz,
            "question": self.question,
            "answers": self.answers,
            "correct_answer_index": self.correct_answer_index,
            "raw_meta": self.raw_meta,
        }
