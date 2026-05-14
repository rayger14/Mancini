"""Structural guards on Failed Breakdown detection.

Regression tests built from real losing trades:
- Trade 2742 (2026-05-03 Sun Globex): FB Long off a self-made intraday low,
  8 bars after Sunday gap-up open. Stopped out -29.25pts.
- Trade 1147 (2026-05-01 NY lunch): FB Long off a 4-bar-old intraday low
  inside a 5pt chop range. Slow death loss.

Both setups defended INTRADAY_LOW levels via two different paths:
  1. _scan_for_deep_sell_recovery (Path 2 in update()) — fires retroactively
     on any INTRADAY_LOW where price is already above. Has no actual
     "deep sell" precondition despite the name.
  2. _scan_for_level_sweep (Path 4 in update()) — treats INTRADAY_LOW as a
     high-quality level eligible for the no-elevator level-sweep path.

Mancini's framework requires either:
  (a) a significant prior-session level (PDL/MHL/cluster/Mancini-posted), or
  (b) a preceding elevator-down sell that traps shorts.

Both paths should refuse to bypass the elevator requirement for INTRADAY_LOW
levels that were not preceded by a real flush.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from config.levels import Level, LevelStore, LevelType
from config.settings import StrategyParams
from core.patterns import FailedBreakdown
from tests.conftest import make_bars


# Trade 2742 bar sequence — Sunday Globex 5/3/2026 18:00-18:10 ET
# Friday close ~7247.75, then gap up to 7275.25 open, chop 7267-7279, FB fired bar #12
TRADE_2742_BARS = [
    # (open, high, low, close)  — comments label live bars
    (7247.25, 7248.00, 7246.25, 7247.75),  # bar #1: Fri 16:59 last RTH minute
    (7275.25, 7279.50, 7272.25, 7274.75),  # bar #2: Sun 18:00 Globex open (gap +28)
    (7274.75, 7275.50, 7271.00, 7273.25),  # bar #3
    (7273.50, 7273.75, 7267.75, 7268.75),  # bar #4: session low forms at 7267.75
    (7268.75, 7270.00, 7267.25, 7268.25),  # bar #5
    (7268.50, 7271.50, 7267.75, 7271.50),  # bar #6
    (7271.00, 7271.75, 7267.25, 7269.00),  # bar #7
    (7269.00, 7270.50, 7268.50, 7270.25),  # bar #8
    (7270.50, 7270.75, 7268.75, 7269.75),  # bar #9
    (7269.75, 7270.75, 7269.00, 7269.50),  # bar #10
    (7269.25, 7269.50, 7268.50, 7269.25),  # bar #11
    (7269.00, 7269.50, 7267.75, 7268.25),  # bar #12: live FB triggered here
]

# Trade 1147 bar sequence — 2026-05-01 13:00-13:13 NY lunch chop
# 5pt range 7269.75-7276.00, no preceding elevator down
TRADE_1147_BARS = [
    (7270.25, 7273.25, 7270.25, 7273.00),  # 12:54
    (7273.00, 7274.50, 7272.75, 7274.00),  # 12:55
    (7273.75, 7274.75, 7272.00, 7272.25),  # 12:56
    (7272.25, 7273.50, 7272.00, 7272.75),  # 12:57
    (7273.00, 7275.00, 7272.50, 7274.25),  # 12:58
    (7274.25, 7274.25, 7270.00, 7270.25),  # 12:59: dip to 7270
    (7270.00, 7272.75, 7269.75, 7272.25),  # 13:00: sweep low 7269.75, recovers
    (7272.50, 7274.50, 7272.25, 7274.25),  # 13:01
    (7274.25, 7276.00, 7273.00, 7275.75),  # 13:02: live FB triggered here
]


@pytest.fixture
def prod_like_params() -> StrategyParams:
    """Strategy params matching the deployed live config: deep_sell_recovery
    and level_sweep_fb both enabled. This is what fires the FB on tonight's
    bar pattern.
    """
    return StrategyParams(
        allow_deep_sell_recovery=True,
        allow_level_sweep_fb=True,
    )


def _run_fb_no_elevator(bars, level_price: float, level_type: LevelType,
                       params: StrategyParams) -> list:
    """Run a bar sequence through FB detector with the given seeded level
    and no elevator event. Returns list of (bar_idx, signal) for any signal
    emitted.
    """
    fb = FailedBreakdown(params)
    df = make_bars(bars, start=datetime(2026, 5, 3, 18, 0))

    store = LevelStore()
    base_time = df.index[0]
    store.add(Level(
        price=level_price,
        level_type=level_type,
        created_at=base_time,
        confirmed_at=base_time,
    ))

    signals = []
    for i in range(len(df)):
        result = fb.update(
            bar_idx=i,
            timestamp=df.index[i],
            high=float(df["high"].iat[i]),
            low=float(df["low"].iat[i]),
            close=float(df["close"].iat[i]),
            level_store=store,
            elevator_event=None,
        )
        if result is not None:
            signals.append((i, result))
    return signals


class TestNoElevatorIntradayLowRejected:
    """An INTRADAY_LOW alone should not be enough to fire an FB.

    Without an elevator-down preceding the sweep, the level-sweep fast path
    only applies to high-conviction prior-session levels. Self-formed
    intraday lows do not qualify.
    """

    def test_trade_2742_intraday_low_no_elevator_rejected(self, prod_like_params):
        """Tonight's losing trade should not trigger an FB."""
        signals = _run_fb_no_elevator(
            TRADE_2742_BARS,
            level_price=7267.75,
            level_type=LevelType.INTRADAY_LOW,
            params=prod_like_params,
        )
        assert signals == [], (
            f"INTRADAY_LOW + no elevator should not fire FB; "
            f"got {len(signals)} signal(s) at bars {[s[0] for s in signals]}"
        )

    def test_trade_1147_intraday_low_no_elevator_rejected(self, prod_like_params):
        """5/1 NY-lunch losing trade should not trigger an FB either."""
        signals = _run_fb_no_elevator(
            TRADE_1147_BARS,
            level_price=7270.00,
            level_type=LevelType.INTRADAY_LOW,
            params=prod_like_params,
        )
        assert signals == [], (
            f"INTRADAY_LOW + no elevator should not fire FB; "
            f"got {len(signals)} signal(s) at bars {[s[0] for s in signals]}"
        )


class TestRealDeepSellStillFires:
    """Regression: Path 2 (deep_sell_recovery) is meant to retroactively
    fire on legitimate crash bottoms — when an actual sustained sell drove
    price down to an INTRADAY_LOW, then recovery proved the bottom.

    Tightening Path 2 to require a real drop must NOT block these cases.
    """

    def test_real_deep_sell_into_intraday_low_still_fires(self, prod_like_params):
        """Synthetic: 25pt sell from 7295 down to 7270, recovery to 7280.
        Level at 7270 (INTRADAY_LOW). Path 2 should retro-fire.
        """
        # Phase 1: sell from 7295 down to 7270 (25pt drop over 12 bars)
        prices = []
        p = 7295.0
        for _ in range(12):
            o = p
            c = p - 2.0  # ~2 pts per bar
            prices.append((o, max(o, c) + 0.5, min(o, c) - 0.5, c))
            p = c
        # Phase 2: recover above 7270 and hold
        p = 7270.5
        for _ in range(8):
            o = p
            c = p + 1.5
            prices.append((o, max(o, c) + 0.5, min(o, c) - 0.5, c))
            p = c

        signals = _run_fb_no_elevator(
            prices,
            level_price=7270.0,
            level_type=LevelType.INTRADAY_LOW,
            params=prod_like_params,
        )
        assert len(signals) >= 1, (
            f"Real ~25pt deep sell into INTRADAY_LOW + recovery should "
            f"still trigger Path 2 retroactively; got 0 signals"
        )
