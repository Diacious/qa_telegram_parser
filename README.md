# Telegram Channel Parser

Parses every message from a Telegram channel and classifies each one into:

| Category | Description |
|---|---|
| `simple_qa` | Single question with a single answer |
| `multi_answer_qa` | Question with multiple answers/options |
| `other` | No question-answer structure |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Telegram API credentials from [my.telegram.org/apps](https://my.telegram.org/apps)
- An LLM provider (OpenAI-compatible API **or** local [Ollama](https://ollama.com))

## Setup

```bash
# Install dependencies
uv sync

# Copy & fill in your credentials
cp .env.example .env
# Edit .env with your values
```

## Usage

```bash
# Parse a public channel (first run asks for Telegram login)
uv run telegram-parse durov

# Limit to 50 messages
uv run telegram-parse durov -n 50

# Custom output file
uv run telegram-parse durov -o results.json

# Verbose logging
uv run telegram-parse durov -v
```

### Using Ollama (local LLM)

```bash
# Install with Ollama support
uv sync --extra ollama

# Set in .env
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:8b

uv run telegram-parse durov
```

### Using a custom OpenAI-compatible API

Set these in your `.env`:

```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=google/gemini-2.0-flash-001
```

## Output

Results are saved as a JSON array. Each entry:

```json
{
  "message_id": 12345,
  "date": "2025-01-15T10:30:00+00:00",
  "sender": "username",
  "text": "What is Python? Python is a programming language.",
  "category": "simple_qa",
  "question": "What is Python?",
  "answers": ["Python is a programming language."],
  "raw_meta": {}
}
```
