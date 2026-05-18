# Mancini Trading Engine — Claude Code Instructions

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
- **Cron schedule (host crontab)**:
  - `0 17 * * *` — primary scraper (5pm ET)
  - `30 17 * * *` — primary LLM extractor (5:30pm ET)
  - `0 20 * * *` — backup scraper (8pm ET, in case Mancini publishes late)
  - `30 20 * * *` — backup LLM extractor (8:30pm ET)
- **Trading date convention**: `run_mancini_llm_cron.sh` uses `TZ=America/New_York date -d "tomorrow"` to determine the trading_date for the plan file. So evening cron at 17:30 writes `mancini_plan_<next_day>.json`. Aligns correctly when running before midnight ET.
- **Verifying a fetch worked**: body length should be 15,000-50,000+ chars. If it's 2,000-3,000 chars, the cookie is broken (paywall preview only). Check `data/mancini_plan_<date>.json` — a healthy plan has 5+ `planned_setups` and 1+ `danger_zones`; a paywalled one has 0-1 setups.
- **Two extraction systems run in parallel**:
  - `live/substack_compare.py` (heuristic regex) writes `data/mancini_levels_<date>.json` — read by engine when `use_mancini_levels=True` (currently False).
  - `live/mancini_llm_extract.py` (Claude Opus) writes `data/mancini_plan_<date>.json` — read by engine when `use_mancini_llm_plan=True` (currently True).
