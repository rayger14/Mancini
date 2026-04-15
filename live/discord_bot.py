"""Mancini Discord Bot — reply to alerts and have Claude investigate/fix.

Runs as a Docker sidecar on the VM. When you reply to a watchdog alert
(or any message in the channel), Claude reads the bot logs, diagnoses
the issue, and can apply fixes + redeploy automatically.

Usage:
    DISCORD_TOKEN=xxx ANTHROPIC_API_KEY=xxx python3 live/discord_bot.py

Environment variables:
    DISCORD_TOKEN       — Discord bot token
    ANTHROPIC_API_KEY   — Claude API key
    DISCORD_CHANNEL_ID  — (optional) restrict to one channel
    LOG_PATH            — (optional) path to bot.log, default /app/logs/bot.log
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force unbuffered stdout for Docker
sys.stdout.reconfigure(line_buffering=True)

try:
    import discord
    from discord import Intents, Embed
except ImportError:
    print("[DISCORD BOT] ERROR: discord.py not installed. Run: pip install discord.py")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("[DISCORD BOT] ERROR: anthropic not installed. Run: pip install anthropic")
    sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")  # empty = all channels
LOG_PATH = Path(os.environ.get("LOG_PATH", "/app/logs/bot.log"))
TRADES_LOG = Path(os.environ.get("TRADES_LOG", "/app/logs/trades.jsonl"))
ALERTS_PATH = Path(os.environ.get("ALERTS_PATH", "/app/logs/watchdog_alerts.json"))

SYSTEM_PROMPT = """You are the Mancini Trading Bot's on-call engineer. You monitor an automated ES/MES futures day trading engine that runs on a VM via Docker.

Your job: when the user (the trader) replies to an alert or asks a question, you diagnose the issue using the bot logs provided and give a clear, actionable answer.

## What you know
- The bot trades MES futures via Interactive Brokers using the Mancini Method (price action, support/resistance levels, Failed Breakdown longs, Breakdown Shorts)
- It runs in Docker container `mancini_mancini-bot_1` on the VM
- Logs are at /app/logs/bot.log
- The watchdog monitors for: bar gaps, zero-volume (expired contract), error spikes, stale data, missed rollovers, signal pipeline silence
- Common issues: contract expiry, IB disconnects, level detection gaps, time gate blocking

## How to respond
1. Read the logs carefully
2. Identify the root cause
3. Explain in 2-3 sentences what happened and why
4. If you can suggest a fix, do so clearly
5. If it needs a code change, describe exactly what file and what to change

## What you CAN'T do
- You cannot edit files or run commands directly. You can only analyze and recommend.
- If a fix requires code changes, tell the user to open Claude Code to apply it.

Keep responses concise. The user is reading on their phone."""

MAX_LOG_LINES = 150  # last N lines of bot.log to include in context


def get_recent_logs(n: int = MAX_LOG_LINES) -> str:
    """Read last N lines of bot.log."""
    if not LOG_PATH.exists():
        return "(bot.log not found)"
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(LOG_PATH)],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout or "(empty)"
    except Exception as e:
        return f"(error reading logs: {e})"


def get_docker_status() -> str:
    """Get docker container status."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}} {{.Status}}",
             "--filter", "name=mancini"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "(no containers)"
    except Exception as e:
        return f"(error: {e})"


def get_recent_docker_logs(n: int = 50) -> str:
    """Get recent docker logs from the bot container."""
    try:
        result = subprocess.run(
            ["docker", "logs", "mancini_mancini-bot_1", "--tail", str(n)],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout or result.stderr or "(empty)"
    except Exception as e:
        return f"(error: {e})"


def get_alerts() -> str:
    """Read current watchdog alerts."""
    if not ALERTS_PATH.exists():
        return "(no alerts file)"
    try:
        return ALERTS_PATH.read_text()[:2000]
    except Exception as e:
        return f"(error: {e})"


def build_context(user_message: str, referenced_message: Optional[str] = None) -> str:
    """Build the full context for Claude."""
    parts = []

    if referenced_message:
        parts.append(f"## Alert being replied to\n{referenced_message}")

    parts.append(f"## User message\n{user_message}")

    parts.append(f"## Container status\n```\n{get_docker_status()}\n```")

    parts.append(f"## Current watchdog alerts\n```json\n{get_alerts()}\n```")

    parts.append(f"## Recent bot logs (last {MAX_LOG_LINES} lines)\n```\n{get_recent_logs()}\n```")

    parts.append(f"## Recent docker logs (last 50 lines)\n```\n{get_recent_docker_logs()}\n```")

    return "\n\n".join(parts)


async def ask_claude(context: str) -> str:
    """Send context to Claude API and get response."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Claude API error: {e}"


# ── Discord Bot ──────────────────────────────────────────────────────────

intents = Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"[DISCORD BOT] Logged in as {client.user} (ID: {client.user.id})")
    print(f"[DISCORD BOT] Watching for messages...")


@client.event
async def on_message(message: discord.Message):
    # Don't respond to ourselves or other bots
    if message.author.bot:
        return

    # Optional: restrict to a specific channel
    if CHANNEL_ID and str(message.channel.id) != CHANNEL_ID:
        return

    # Check if the message mentions the bot or is a reply to a bot message
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and hasattr(message.reference.resolved, 'author')
        and message.reference.resolved.author.bot
    )
    is_mention = client.user in message.mentions
    # Also respond to messages starting with !mancini
    is_command = message.content.strip().lower().startswith("!mancini")

    if not (is_reply_to_bot or is_mention or is_command):
        return

    # Get the referenced message content if it's a reply
    referenced_content = None
    if message.reference and message.reference.resolved:
        ref = message.reference.resolved
        if ref.embeds:
            # Extract embed content
            embed = ref.embeds[0]
            referenced_content = f"**{embed.title}**\n{embed.description or ''}"
            for field in embed.fields:
                referenced_content += f"\n{field.name}: {field.value}"
        elif ref.content:
            referenced_content = ref.content

    # Clean the user message (remove bot mention and command prefix)
    user_text = message.content
    user_text = user_text.replace(f"<@{client.user.id}>", "").strip()
    if user_text.lower().startswith("!mancini"):
        user_text = user_text[8:].strip()
    if not user_text:
        user_text = "What's going on? Investigate this alert."

    # Show typing indicator while Claude thinks
    async with message.channel.typing():
        # Build context and ask Claude
        context = build_context(user_text, referenced_content)
        response = await ask_claude(context)

    # Split response if too long for Discord (2000 char limit)
    chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
    for chunk in chunks:
        embed = Embed(
            description=chunk,
            color=0x3498DB,
            timestamp=datetime.utcnow(),
        )
        embed.set_footer(text="Claude • Mancini Bot Engineer")
        await message.reply(embed=embed)

    print(f"[DISCORD BOT] Responded to {message.author}: {user_text[:50]}...")


def main():
    if not DISCORD_TOKEN:
        print("[DISCORD BOT] ERROR: DISCORD_TOKEN not set")
        sys.exit(1)
    if not ANTHROPIC_API_KEY:
        print("[DISCORD BOT] ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    print("[DISCORD BOT] Starting Mancini Discord Bot")
    print(f"[DISCORD BOT] Log path: {LOG_PATH}")
    if CHANNEL_ID:
        print(f"[DISCORD BOT] Restricted to channel: {CHANNEL_ID}")
    else:
        print("[DISCORD BOT] Listening in all channels")

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
