"""Tests for the 'missed setup while in a position' alert dedup key.

When the bot is already in a trade (often a collection-mode experiment) and a
new signal fires, it can't take it (single-position), but it should ALERT +
RECORD it — the missed setup may be more legit than the trade we hold. The
alert dedups per setup so it fires once, not on every bar we stay in position.
"""
from types import SimpleNamespace
from live.ib_runner import blocked_alert_key


def _sig(name, level_price=None, entry=0.0):
    lvl = SimpleNamespace(price=level_price) if level_price is not None else None
    return SimpleNamespace(
        signal_type=SimpleNamespace(name=name),
        pattern=SimpleNamespace(level=lvl),
        entry_price=entry,
    )


def test_key_uses_type_and_level():
    assert blocked_alert_key(_sig("FAILED_BREAKDOWN", 7491.0)) == "FAILED_BREAKDOWN@7491"


def test_same_setup_same_key_for_dedup():
    # Two bars of the same setup (level drifts a fraction) → same key → dedup.
    k1 = blocked_alert_key(_sig("FAILED_BREAKDOWN", 7491.25))
    k2 = blocked_alert_key(_sig("FAILED_BREAKDOWN", 7490.75))
    assert k1 == k2 == "FAILED_BREAKDOWN@7491"


def test_different_type_or_level_distinct_keys():
    assert blocked_alert_key(_sig("FAILED_BREAKDOWN", 7491.0)) != \
           blocked_alert_key(_sig("LEVEL_RECLAIM", 7491.0))
    assert blocked_alert_key(_sig("FAILED_BREAKDOWN", 7491.0)) != \
           blocked_alert_key(_sig("FAILED_BREAKDOWN", 7459.0))


def test_falls_back_to_entry_when_no_level():
    assert blocked_alert_key(_sig("FAILED_BREAKDOWN", None, entry=7548.0)) == \
           "FAILED_BREAKDOWN@7548"
