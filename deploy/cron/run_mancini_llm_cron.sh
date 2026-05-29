#!/bin/bash
# Cron entry point for the daily Mancini LLM extraction + Discord brief.
#
# Schedule (host crontab):
#   30 17 * * *  — primary (5:30pm ET, after Mancini's ~4pm post)
#   30 20 * * *  — backup  (8:30pm ET, in case Mancini publishes late)
#
# Sequence:
#   1. Source host .env for SUBSTACK_COOKIE, ANTHROPIC_API_KEY, WATCHDOG_WEBHOOK
#   2. Run LLM extractor inside the bot container — writes
#      data/mancini_plan_<tomorrow>.json
#   3. Run the Discord brief poster inside the bot container — reads the
#      plan JSON and posts a rich embed to $WATCHDOG_WEBHOOK. Idempotent
#      via state file so the backup run doesn't double-post.
#
# Container name: mancini-mancini-bot-1 (compose v2 dash naming).
# If you rename the container, update CONTAINER below.

set -a
. /home/ubuntu/mancini/.env
set +a

CONTAINER="mancini-mancini-bot-1"
DATE_STR=$(TZ=America/New_York date -d "tomorrow" +%Y-%m-%d)
PLAN_FILE_IN_CONTAINER="/app/data/mancini_plan_${DATE_STR}.json"

# 1. Extract
docker exec \
  -e SUBSTACK_COOKIE="$SUBSTACK_COOKIE" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  "$CONTAINER" \
  python3 live/mancini_llm_extract.py
EXTRACT_EXIT=$?

if [ "$EXTRACT_EXIT" -ne 0 ]; then
  # Extraction failed — surface to Discord as a plain error message.
  if [ -n "$WATCHDOG_WEBHOOK" ]; then
    MSG=":rotating_light: **Mancini LLM extraction FAILED** (exit=$EXTRACT_EXIT) for ${DATE_STR}. Check logs."
    PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$MSG")
    curl -s -H "Content-Type: application/json" -d "$PAYLOAD" "$WATCHDOG_WEBHOOK" >/dev/null 2>&1 || true
  fi
  exit "$EXTRACT_EXIT"
fi

# 2. Post brief (idempotent — won't repost if already done for this date)
docker exec \
  -e WATCHDOG_WEBHOOK="$WATCHDOG_WEBHOOK" \
  "$CONTAINER" \
  python3 live/mancini_llm_summary.py \
    --plan-file "$PLAN_FILE_IN_CONTAINER"
POST_EXIT=$?

exit "$POST_EXIT"
