# Mancini Trading Engine — Claude Code Instructions

## Calendar & Date Discipline

**ALWAYS verify today's date before reasoning about trading:** run `TZ=America/New_York date "+%Y-%m-%d %A %Z"` at the start of any date-sensitive task. Do not trust the date from earlier conversation context — it may be stale.

**Reference clock for this project**: all trading dates are **America/New_York (ET)**. Substack posts, IB bracket orders, session boundaries, cron jobs — every timestamp in this codebase normalizes to ET.

**Trading week**:
- Globex opens Sunday 6:00pm ET, closes Friday 5:00pm ET
- One-hour daily break: 5:00pm-6:00pm ET (no trading)
- Weekend = no ES futures (Friday 17:00 → Sunday 18:00 ET)
- "Trading day" runs 18:00 ET previous calendar day → 16:59 ET current calendar day
- Mondays are skipped by `--skip-mondays` (per Optuna v2; backtest convention)

**Mancini post timing & naming**:
| When Mancini posts | Post titled | Bot loads it for |
|---|---|---|
| Sunday ~4pm ET | "Monday Plan" | Monday session (starts Sun 6pm ET) |
| Monday ~4pm ET | "Tuesday Plan" | Tuesday session (starts Mon 6pm ET) |
| Friday ~4pm ET | "Monday Plan" (covers weekend) | Monday session next week |

So the post posted "today evening" describes "tomorrow's trading day." Plan files are named `mancini_plan_<trading_date>.json` where trading_date = the day the session ENDS on (Mon 6pm Sun → Mon 17:00 Mon = trading_date Monday).

**Disambiguation rules — when I (Claude) say "today":**
- ALWAYS resolve to the actual ET calendar date by running `date`. Do not assume.
- "Today's trade" = a trade that fired during the session ending today's date (which started 6pm yesterday ET)
- "Today's plan" = the plan file named for today (loaded from yesterday evening's Mancini post)
- "Tomorrow's plan" = the plan file for tomorrow's date (Mancini posts it this evening)

**Common date confusion patterns to avoid**:
1. Conversation context says "today is May 17" but it's now May 18 → ALWAYS re-check via `date`
2. The plan file `mancini_plan_2026-05-19.json` describes Tuesday May 19 trading, sourced from a Monday May 18 evening post
3. ES bar timestamps in UTC: subtract 4-5h for ET (4h EDT, 5h EST)

## Git Workflow

### Branching
- **Never commit directly to `main`**. Always create a feature branch first.
- Branch naming: `feat/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`
- Examples: `feat/optuna-v2-params`, `fix/ib-bracket-fill`, `refactor/regime-filter`

### Commits
- Commit early and often — after each logical unit of work, not at the end.
- Each commit should be atomic: one concern per commit (don't mix bug fixes with features).
- Write clear commit messages: imperative mood, explain WHY not just WHAT.
- Always run `python3 -m pytest tests/ -v` before committing. Never commit failing tests.
- Stage specific files (`git add file1.py file2.py`), never `git add -A` or `git add .`
- Never commit: `.env`, credentials, `.parquet` data files, `__pycache__/`, `.venv*/`

### Pull Requests
- Create a PR for every feature branch before merging to main.
- PR title: short, under 70 characters.
- PR body must include: Summary (what changed and why), Test plan, Pattern breakdown if backtest results changed.
- Always include backtest results (before/after) when changing strategy params or logic.
- Use `gh pr create` to create PRs.

### PR Reviews
When reviewing a PR (own or others):
- Check that all tests pass
- Verify no hardcoded credentials or data paths leaked
- Confirm backtest results are reproducible
- Look for: off-by-one errors in bar indexing, timezone issues (ET vs UTC), float comparison bugs
- Verify changes don't break position recovery or runner management in live mode

## Code Standards

### Python
- Python 3.9+ compatible (system python on macOS)
- Use `python3` and `python3 -m pip` (no `pip` alias)
- Type hints on public functions
- Use loguru for logging, never print() in production code
- Pydantic for config/settings validation

### Trading-Specific
- All prices are in points (not dollars). Convert with contract spec.
- Timestamps must be timezone-aware (US/Eastern for display, UTC for storage).
- Never assume bar order from aggregated stats — always trace chronological price path.
- Level-based stops (not fixed points) for FB entries.
- Test with both RTH-only and full-session data.

### Testing
- Run: `python3 -m pytest tests/ -v`
- 91 tests passing, 10 nautilus skipped
- When adding new pattern logic, add matching test cases
- Use realistic price data in tests (ES ~5000-7000 range, not 100.0)

## Project Layout
```
config/          Settings, levels, contract specs
core/            Pattern detection, signals, indicators, regime filter
strategy/        Entry/exit/risk/position management, ManciniLongStrategy
backtest/        BacktestRunner (authoritative), Optuna, analysis scripts
live/            IB bridge, IB runner, dashboard, retrospective, cron jobs
tests/           pytest suite
data/            Parquet files, live trade logs, Optuna results
```

## VM Deployment
- VM: 152.70.113.24, user `ubuntu`, key `~/.ssh/oracle_bullmachine`
- Bot runs in Docker: `mancini_mancini-bot_1`
- Deploy: scp files -> `docker build --no-cache` -> stop/rm/run container
- Always check for open positions before restarting: `docker logs ... | grep ENTRY`
- Data and logs mounted from `/home/ubuntu/mancini/{data,logs}` to `/app/{data,logs}`

## Mancini Substack Plan Loading
- **Cookie format matters**: `SUBSTACK_COOKIE` env var must be sent as `Cookie: substack.sid={value}` (the `live/substack_compare.py` wrapper handles this automatically since the 2026-05-18 fix). Sending the raw value without the `substack.sid=` name serves only paywall preview (~2.5KB body) instead of full paid content (~30KB+). Silently breaks LLM extraction.
- **Posts published ~4pm ET** the day BEFORE the trading session they describe. The "May 19 Plan" is posted Monday May 18 evening for Tuesday May 19 trading. ES globex session starts 6pm ET, so the plan must be loaded BEFORE then.
- **Cron schedule (host crontab — times are UTC!)**: the VM host runs UTC and Ubuntu cron schedules in system time; a `TZ=` line in the crontab does NOT change firing times (lesson of 2026-06-11 — jobs ran 4h early in ET for a week, publishing yesterday's post as tomorrow's plan). Canonical schedule in `deploy/cron/host_crontab.txt`: scraper+extractor rounds at 21:00/21:30, 22:00/22:30, 00:00/00:30 UTC (= 17:00–20:30 EDT).
- **Stale-post validation (since #48)**: `live/mancini_llm_extract.py` parses the plan date from the post title ("June 12th Plan" is authoritative) and refuses to write a plan whose post doesn't describe the target trading date — early/duplicate cron runs write a `stale_post` stub instead and never overwrite an `extract_status=ok` plan. The Discord brief poster stays silent on stale stubs.
- **Trading date convention**: target date is the next ET *trading* day, skipping weekends (`next_trading_date()`): Friday/Saturday runs target Monday, matching Mancini's Friday-evening "Monday Plan" post. The cron shell wrapper, extractor default, and brief-poster default all share this rule.
- **Verifying a fetch worked**: body length should be 15,000-50,000+ chars. If it's 2,000-3,000 chars, the cookie is broken (paywall preview only). Check `data/mancini_plan_<date>.json` — a healthy plan has 5+ `planned_setups` and 1+ `danger_zones`; a paywalled one has 0-1 setups.
- **Two extraction systems run in parallel**:
  - `live/substack_compare.py` (heuristic regex) writes `data/mancini_levels_<date>.json` — read by engine when `use_mancini_levels=True` (currently False).
  - `live/mancini_llm_extract.py` (Claude Opus) writes `data/mancini_plan_<date>.json` — read by engine when `use_mancini_llm_plan=True` (currently True).
