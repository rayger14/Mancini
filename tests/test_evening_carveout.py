"""Evening fast-reclaim carve-out (2026-07-24).

NON_ACCEPTANCE longs trade 18:00-22:00 ET as full production; everything
else in that window stays blocked. Flag off = legacy blanket block.
"""
from datetime import time
from types import SimpleNamespace

from config.settings import RiskParams
from strategy.risk_manager import RiskManager


def _signal(conf_name, direction="long"):
    conf = SimpleNamespace(name=conf_name)
    return SimpleNamespace(
        pattern=SimpleNamespace(confirmation=conf),
        direction=direction,
    )


def _rm(flag):
    return RiskManager(risk_params=RiskParams(evening_allow_non_acceptance=flag))


class TestEveningCarveout:
    def test_non_acceptance_long_passes_when_flag_on(self):
        check = _rm(True)._check_not_evening_block(
            time(19, 30), _signal("NON_ACCEPTANCE"))
        assert check.passed
        assert "carve-out" in check.reason

    def test_acceptance_still_blocked_when_flag_on(self):
        check = _rm(True)._check_not_evening_block(
            time(19, 30), _signal("ACCEPTANCE"))
        assert not check.passed

    def test_non_acceptance_short_still_blocked(self):
        check = _rm(True)._check_not_evening_block(
            time(19, 30), _signal("NON_ACCEPTANCE", direction="short"))
        assert not check.passed

    def test_flag_off_blocks_everything(self):
        check = _rm(False)._check_not_evening_block(
            time(19, 30), _signal("NON_ACCEPTANCE"))
        assert not check.passed

    def test_outside_window_unaffected(self):
        check = _rm(False)._check_not_evening_block(
            time(23, 0), _signal("ACCEPTANCE"))
        assert check.passed

    def test_missing_signal_blocked_not_crashing(self):
        check = _rm(True)._check_not_evening_block(time(19, 30), None)
        assert not check.passed
