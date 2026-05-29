#!/bin/bash
# Cron entry point for the daily Substack scraper (regex-heuristic levels).
#
# Schedule (host crontab):
#   0 17 * * *  — primary (5pm ET, after Mancini's ~4pm post)
#   0 20 * * *  — backup  (8pm ET, in case Mancini publishes late)
#
# Sources host .env so SUBSTACK_COOKIE is in scope, then exec the
# scraper inside the bot container with -e injection.
#
# Container name: mancini-mancini-bot-1 (compose v2 dash naming).

set -a
. /home/ubuntu/mancini/.env
set +a

CONTAINER="mancini-mancini-bot-1"

exec docker exec -e SUBSTACK_COOKIE="$SUBSTACK_COOKIE" "$CONTAINER" \
     python3 live/substack_compare.py
