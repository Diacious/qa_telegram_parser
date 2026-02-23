from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from telethon import TelegramClient
from telethon.tl.functions.messages import SendVoteRequest
from telethon.tl.types import Message, MessageMediaPoll, UpdateMessagePoll

from telegram_parsing.config import TelegramConfig

logger = logging.getLogger(__name__)


@dataclass
class RawMessage:
    """Lightweight container for data pulled from Telegram."""

    message_id: int
    date: str
    sender: str | None
    text: str

    # Quiz / poll fields (populated only for MessageMediaPoll)
    is_quiz: bool = False
    poll_question: str | None = None
    poll_options: list[str] = field(default_factory=list)
    correct_option_index: int | None = None  # index into poll_options


def _extract_poll_text(obj: object) -> str:
    """Extract plain text from a Poll question/answer.

    Telethon >= 1.35 wraps these in ``TextWithEntities``;
    older versions use plain ``str`` or ``bytes``.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # TextWithEntities (has a .text attribute)
    return getattr(obj, "text", str(obj))


def _find_correct_from_results(results, poll_answers) -> int | None:
    """Scan poll result objects for the one marked correct."""
    if not results:
        return None
    for r in results:
        if getattr(r, "correct", False):
            for idx, ans in enumerate(poll_answers):
                if ans.option == r.option:
                    return idx
    return None


def _find_correct_in_vote_response(vote_result, poll_answers) -> int | None:
    """Extract the correct answer index from a SendVoteRequest response.

    Telegram may return the data in different update types depending on
    the client layer, so we try all known shapes.
    """
    updates = getattr(vote_result, "updates", [])
    for update in updates:
        # Shape 1: UpdateMessagePoll (has .results directly)
        if isinstance(update, UpdateMessagePoll):
            found = _find_correct_from_results(
                getattr(update.results, "results", None), poll_answers
            )
            if found is not None:
                return found

        # Shape 2: UpdateEditMessage / UpdateNewMessage (has .message.media)
        msg = getattr(update, "message", None)
        if msg is not None:
            media = getattr(msg, "media", None)
            if isinstance(media, MessageMediaPoll):
                found = _find_correct_from_results(
                    getattr(media.results, "results", None), poll_answers
                )
                if found is not None:
                    return found

        # Shape 3: Any update with .media directly
        media = getattr(update, "media", None)
        if isinstance(media, MessageMediaPoll):
            found = _find_correct_from_results(
                getattr(media.results, "results", None), poll_answers
            )
            if found is not None:
                return found

    return None


async def iter_channel_messages(
    config: TelegramConfig,
    channel: str,
    *,
    max_messages: int | None = None,
) -> AsyncIterator[RawMessage]:
    """Yield every text message and quiz/poll from *channel*.

    Parameters
    ----------
    config:
        Telegram API credentials.
    channel:
        Public channel username (e.g. ``"durov"``) or numeric ID.
    max_messages:
        Optional cap on number of messages to fetch.
    """
    client = TelegramClient(config.session_name, config.api_id, config.api_hash)
    await client.start()

    try:
        entity = await client.get_entity(channel)

        async for msg in client.iter_messages(entity, reverse=False, limit=max_messages):
            if not isinstance(msg, Message):
                continue

            sender_name: str | None = None
            if msg.sender:
                sender_name = getattr(msg.sender, "username", None) or getattr(
                    msg.sender, "first_name", None
                )

            # ── Quiz / Poll message ──────────────────────────────────
            if isinstance(msg.media, MessageMediaPoll):
                poll = msg.media.poll
                question_text = _extract_poll_text(poll.question)
                options = [_extract_poll_text(a.text) for a in poll.answers]

                # Find the correct answer index (only available for quizzes)
                correct_idx: int | None = None
                if poll.quiz and msg.media.results and msg.media.results.results:
                    for r in msg.media.results.results:
                        if r.correct:
                            for idx, ans in enumerate(poll.answers):
                                if ans.option == r.option:
                                    correct_idx = idx
                                    break
                            break

                # If we still don't know the correct answer (haven't voted yet),
                # vote on the first option to make Telegram reveal it.
                if poll.quiz and correct_idx is None and not poll.closed:
                    try:
                        vote_result = await client(
                            SendVoteRequest(
                                peer=entity,
                                msg_id=msg.id,
                                options=[poll.answers[0].option],
                            )
                        )
                        # Extract correct answer from the vote response.
                        # Telegram returns UpdateMessagePoll with .results,
                        # or sometimes UpdateMessageID / UpdateEditMessage.
                        correct_idx = _find_correct_in_vote_response(
                            vote_result, poll.answers
                        )
                    except Exception:
                        logger.warning(
                            "Could not auto-vote on quiz %d to reveal correct answer",
                            msg.id,
                        )

                yield RawMessage(
                    message_id=msg.id,
                    date=msg.date.isoformat(),
                    sender=sender_name,
                    text=msg.text or "",
                    is_quiz=True,
                    poll_question=question_text,
                    poll_options=options,
                    correct_option_index=correct_idx,
                )
                continue

            # ── Regular text message ─────────────────────────────────
            if not msg.text:
                continue

            yield RawMessage(
                message_id=msg.id,
                date=msg.date.isoformat(),
                sender=sender_name,
                text=msg.text,
            )
    finally:
        await client.disconnect()
