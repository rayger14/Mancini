"""Tests for the Mode 1 Red FB-long block gate in SignalAggregator.

Drawn directly from Mancini's May 8 2025 post:
  "On Mode 1 red days, there often won't be a failed breakdown until
   before the close or in the early evening and one just has to wait
   patiently all day for the sell to complete."

Behavior under test:
  * FB longs are BLOCKED when mode1_red_active=True AND time < cutoff
    (default 15:00 ET).
  * FB longs are ALLOWED past the cutoff (Mancini's "near the close").
  * Non-FB signals (LR, shorts) are unaffected.
  * When use_mode1_detection is False, the gate is a no-op.

Today's production loss case (2026-06-05 13:50 ET FB long @ 7474,
stopped out −$650) is encoded explicitly in test_real_world_2026_06_05.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from types import SimpleNamespace

import pytest

from core.signals import SignalAggregator, SignalType
from config.settings import StrategyParams


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeLevel:
    price: float = 7474.0


@dataclass
class _FakePattern:
    timestamp: datetime
    direction: str = "long"
    entry_price: float = 7474.0
    level: object = field(default_factory=_FakeLevel)


def _make_aggregator(
    *,
    use_mode1: bool = True,
    cutoff_hour: int = 15,
    cutoff_minute: int = 0,
) -> SignalAggregator:
    params = StrategyParams(
        use_mode1_detection=use_mode1,
        mode1_red_fb_long_block_until_hour=cutoff_hour,
        mode1_red_fb_long_block_until_minute=cutoff_minute,
    )
    agg = SignalAggregator(strategy_params=params)
    return agg


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------


class TestMode1RedBlocksFBLongs:
    def test_blocks_fb_long_when_red_active_before_cutoff(self):
        agg = _make_aggregator()
        agg.mode1_red_active = True
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 13, 50))
        reason = agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN)
        assert reason is not None
        assert "mode1_red" in reason
        assert "wait patiently" in reason  # Mancini quote in the reason

    def test_allows_fb_long_when_red_inactive(self):
        agg = _make_aggregator()
        agg.mode1_red_active = False
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 13, 50))
        assert agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN) is None

    def test_allows_fb_long_past_cutoff_even_when_red(self):
        """Mancini: 'won't be a failed breakdown until before the close.'
        Past the cutoff, FBs are allowed."""
        agg = _make_aggregator(cutoff_hour=15, cutoff_minute=0)
        agg.mode1_red_active = True
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 15, 30))
        assert agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN) is None

    def test_exact_cutoff_time_is_allowed(self):
        agg = _make_aggregator(cutoff_hour=15, cutoff_minute=0)
        agg.mode1_red_active = True
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 15, 0))
        assert agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN) is None

    def test_one_minute_before_cutoff_still_blocked(self):
        agg = _make_aggregator(cutoff_hour=15, cutoff_minute=0)
        agg.mode1_red_active = True
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 14, 59))
        assert agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN) is not None

    def test_short_signal_not_blocked(self):
        """The gate only blocks FB longs (Mancini's specific rule).
        Shorts are filtered by separate gates."""
        agg = _make_aggregator()
        agg.mode1_red_active = True
        pat = _FakePattern(
            timestamp=datetime(2026, 6, 5, 13, 50),
            direction="short",
        )
        assert agg._check_mode1_red_gate(pat, SignalType.BREAKDOWN_SHORT) is None

    def test_level_reclaim_not_blocked(self):
        """Only FAILED_BREAKDOWN is blocked. LR is a different mechanic."""
        agg = _make_aggregator()
        agg.mode1_red_active = True
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 13, 50))
        assert agg._check_mode1_red_gate(pat, SignalType.LEVEL_RECLAIM) is None


# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------


class TestUseMode1DetectionSwitch:
    def test_gate_is_noop_when_use_mode1_detection_false(self):
        agg = _make_aggregator(use_mode1=False)
        agg.mode1_red_active = True  # even if forced
        pat = _FakePattern(timestamp=datetime(2026, 6, 5, 13, 50))
        assert agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN) is None


# ---------------------------------------------------------------------------
# Production scenario
# ---------------------------------------------------------------------------


class TestRealWorldProductionScenario:
    def test_real_world_2026_06_05(self):
        """The actual losing trade: 2026-06-05, 13:50 ET, FB long @ 7474,
        stopped at 7441.50 for −$650. After Mancini's rule is enforced,
        this trade is blocked at the qualify_signal layer.

        Today's session DID develop into Mode 1 Red — RTH open 7548,
        close 7368, 220-pt range, closed near the low. By 13:50 the
        Mode1Detector would have flagged it Red. With the gate active,
        the FB long signal is rejected before order placement.
        """
        agg = _make_aggregator()
        agg.mode1_red_active = True
        pat = _FakePattern(
            timestamp=datetime(2026, 6, 5, 13, 50),
            entry_price=7474.0,
        )
        reason = agg._check_mode1_red_gate(pat, SignalType.FAILED_BREAKDOWN)
        assert reason is not None
        assert "mode1_red" in reason
