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
