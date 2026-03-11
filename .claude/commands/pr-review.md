Review the current branch's changes against main and provide a thorough PR review.

## Steps

1. Run `git diff main...HEAD` to see all changes in this branch
2. Run `git log main..HEAD --oneline` to see commit history
3. Run `python3 -m pytest tests/ -v` to verify all tests pass
4. Check for:
   - Breaking changes to live bot (ib_runner.py, ib_bridge.py)
   - Untested code paths in strategy/core changes
   - Hardcoded values that should be params
   - Timezone bugs (ET vs UTC)
   - Missing phantom/near-miss tracking for new signal types
   - Position recovery compatibility
   - Off-by-one bar indexing errors
5. If strategy params or pattern logic changed, run a quick 1yr backtest to compare before/after
6. Provide a summary: APPROVE, REQUEST CHANGES, or NEEDS DISCUSSION with specific feedback
