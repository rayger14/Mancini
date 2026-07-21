"""Full-state position snapshot: restarts must restore EVERY field.

Trade 781 (2026-07-21): a mid-trade deploy restart reconstructed the
position from the trade log — 6 fields of ~15. trade_id logged as 0,
and any field nobody explicitly copied silently defaulted. The fix is
snapshot-don't-reconstruct: persist the whole TradePosition on every
bar; recovery loads it and only VERIFIES against the venue. The
round-trip test iterates dataclass fields so a future field can't
silently join the bug class.
"""
import dataclasses
import math
from collections import deque

from strategy.exit_manager import ExitPhase, TradePosition


def _full_position() -> TradePosition:
    p = TradePosition(
        entry_price=7505.75, stop_price=7502.87, target_1=7535.5,
        target_2=7550.5, total_contracts=2, remaining_contracts=1,
        phase=ExitPhase.AFTER_T1, highest_price_since_entry=7535.75,
        lowest_price_since_entry=7488.5, realized_pnl_pts=29.75,
        direction="long", prior_day_low=7473.25, prior_day_high=7538.0,
        t1_hit=True, t2_hit=False, is_double_dip=True,
    )
    p.bar_history = deque([(7530.0, 7522.0), (7534.5, 7528.25)], maxlen=30)
    return p


class TestSnapshotRoundTrip:
    def test_every_field_survives(self):
        orig = _full_position()
        restored = TradePosition.from_snapshot(orig.to_snapshot())
        for f in dataclasses.fields(TradePosition):
            a, b = getattr(orig, f.name), getattr(restored, f.name)
            if f.name == "bar_history":
                assert list(a) == list(b), f.name
            else:
                assert a == b, f.name

    def test_json_safe_through_dumps_loads(self):
        import json
        p = TradePosition(entry_price=7500.0, stop_price=7490.0,
                          target_1=7510.0, target_2=7520.0,
                          total_contracts=2, remaining_contracts=2)
        # mid-trade tracker values must round-trip exactly through real
        # JSON (not get re-seeded by __post_init__ on restore)
        p.lowest_price_since_entry = 7493.25
        p.highest_price_since_entry = 7508.5
        blob = json.dumps(p.to_snapshot())
        restored = TradePosition.from_snapshot(json.loads(blob))
        assert restored.lowest_price_since_entry == 7493.25
        assert restored.highest_price_since_entry == 7508.5

    def test_phase_enum_by_name(self):
        p = _full_position()
        snap = p.to_snapshot()
        assert snap["phase"] == "AFTER_T1"

    def test_from_snapshot_rejects_garbage(self):
        assert TradePosition.from_snapshot(None) is None
        assert TradePosition.from_snapshot({}) is None
        assert TradePosition.from_snapshot({"entry_price": "x"}) is None
