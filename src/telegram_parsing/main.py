from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from telegram_parsing.classifier import classify_quiz, create_classifier
from telegram_parsing.config import AppConfig
from telegram_parsing.models import ClassifiedMessage, MessageCategory
from telegram_parsing.telegram_client import RawMessage, iter_channel_messages

console = Console()
logger = logging.getLogger("telegram_parsing")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and classify Telegram channel messages.",
    )
    parser.add_argument(
        "channel",
        help="Public channel username (e.g. 'durov') or numeric chat ID.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file path (overrides OUTPUT_FILE env var).",
    )
    parser.add_argument(
        "-n",
        "--max-messages",
        type=int,
        default=None,
        help="Maximum number of messages to fetch (overrides MAX_MESSAGES env var).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of messages to classify in parallel (overrides BATCH_SIZE env var).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def _print_summary(results: list[ClassifiedMessage]) -> None:
    counts = {cat: 0 for cat in MessageCategory}
    for r in results:
        counts[r.category] += 1

    table = Table(title="Classification Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for cat, count in counts.items():
        table.add_row(cat.value, str(count))

    table.add_row("[bold]Total[/bold]", f"[bold]{len(results)}[/bold]")
    console.print(table)


async def run(args: argparse.Namespace) -> None:
    config = AppConfig()

    # CLI overrides
    if args.output:
        config.output_file = args.output
    if args.max_messages is not None:
        config.max_messages = args.max_messages
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    classifier = create_classifier(config.llm)

    # ── Phase 1: Fetch messages ──────────────────────────────────────────
    console.print(f"\n[bold]Fetching messages from [cyan]{args.channel}[/cyan]...[/bold]")
    console.print(
        "[dim]First run will ask you to log in to Telegram (phone + code).[/dim]\n"
    )

    quiz_messages: list[RawMessage] = []
    text_messages: list[RawMessage] = []

    async for msg in iter_channel_messages(
        config.telegram, args.channel, max_messages=config.max_messages
    ):
        if msg.is_quiz:
            quiz_messages.append(msg)
        else:
            text_messages.append(msg)

    total = len(quiz_messages) + len(text_messages)
    if total == 0:
        console.print("[yellow]No messages found in channel.[/yellow]")
        return

    console.print(
        f"Fetched [green]{total}[/green] messages "
        f"([cyan]{len(quiz_messages)}[/cyan] quizzes, "
        f"[cyan]{len(text_messages)}[/cyan] text).\n"
    )

    # ── Phase 2: Classify ────────────────────────────────────────────────
    results: list[ClassifiedMessage] = []

    # Quizzes are classified directly — no LLM needed
    for qm in quiz_messages:
        results.append(classify_quiz(qm))

    # Text messages go through the LLM
    if text_messages:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Classifying text messages…", total=len(text_messages))

            for i in range(0, len(text_messages), config.batch_size):
                batch = text_messages[i : i + config.batch_size]
                classified = await classifier.classify_batch(batch)
                results.extend(classified)
                progress.advance(task, advance=len(batch))

    # Sort by message_id to restore chronological order
    results.sort(key=lambda r: r.message_id)

    # ── Phase 3: Write output ────────────────────────────────────────────
    output_data = [r.to_dict() for r in results]
    out, ext = (config.output_file or "classified_messages.json").rsplit(".", 1)
    output_file = f"{out}_{args.channel}.{ext}"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    console.print(
        f"\n[green]Results saved to [bold]{output_file}[/bold][/green]"
    )
    _print_summary(results)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
