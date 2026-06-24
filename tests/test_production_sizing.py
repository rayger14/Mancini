"""The production size must enable Mancini's full 75/15/10 (level-to-level).

At 4 contracts, floor(4*0.15)=0 — the T2 tranche rounds away and the bot
collapses to 75% + runner (2 rungs). Mancini's method is 3 rungs: lock 75% at
the first level up, more (15%) at the second, ride a 10% runner. That needs
at least 7 contracts; we size full-conviction to 8.
"""
from __future__ import annotations

import math

from live.ib_runner import PRODUCTION_EXIT, PRODUCTION_RISK


def test_production_full_size_enables_all_three_tranches():
    n = PRODUCTION_EXIT.default_contracts
    # the position cap must not clip the full conviction size
    assert PRODUCTION_RISK.max_position_contracts >= n, (
        "max_position_contracts would clip full-size below default_contracts"
    )
    t1 = math.floor(n * PRODUCTION_EXIT.t1_exit_fraction)
    t2 = math.floor(n * PRODUCTION_EXIT.t2_exit_fraction)
    runner = n - t1 - t2
    assert t1 >= 1 and t2 >= 1 and runner >= 1, (
        f"75/15/10 needs three tranches; at {n} contracts got "
        f"T1={t1} T2={t2} runner={runner}"
    )


def test_full_size_is_eight():
    assert PRODUCTION_EXIT.default_contracts == 8
