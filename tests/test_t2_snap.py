"""T2 snap-to-real-level: the arithmetic T2 (3R) often hangs 1-3pts ABOVE the
real supply where price actually turns (622: T2 7581 vs Mancini rung 7578,
price topped 7579; 07-06: T2 7603.25 vs prior high, session top 7602.25).

Snap T2 DOWN onto the nearest real level below it — priority order:
  1. Mancini's published target rungs (his ladder)
  2. engine-detected resistance levels
  3. no level within tolerance -> keep the arithmetic T2

Gated by t2_snap_to_level_tol_pts (0.0 = off, the default).
"""
from core.signals import snap_t2_to_real_level


def test_snaps_to_mancini_rung():
    # the 622 case: math says 7581, his ladder has 7578 (3pt below, tol 4)
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7569.75, mancini_targets=[7550.0, 7578.0, 7597.0],
        engine_levels=[], tol=4.0) == 7578.0


def test_mancini_rung_beats_engine_level():
    # both in range: his rung wins even when the engine level is nearer to t2
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7560.0, mancini_targets=[7578.0],
        engine_levels=[7580.0], tol=4.0) == 7578.0


def test_falls_back_to_engine_level():
    # no rung in range -> nearest engine resistance below t2
    assert snap_t2_to_real_level(
        t2=7603.25, t1=7580.0, mancini_targets=[7550.0, 7640.0],
        engine_levels=[7601.5, 7597.0], tol=4.0) == 7601.5


def test_keeps_math_when_nothing_in_tolerance():
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7560.0, mancini_targets=[7550.0],
        engine_levels=[7570.0], tol=4.0) == 7581.0


def test_never_snaps_at_or_below_t1():
    # candidate inside tol but at/below T1 is not a valid second target
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7578.5, mancini_targets=[7578.0],
        engine_levels=[], tol=4.0) == 7581.0


def test_ignores_levels_above_t2():
    # only snap DOWN — a level above the math target is not a snap candidate
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7560.0, mancini_targets=[7583.0],
        engine_levels=[7585.0], tol=6.0) == 7581.0


def test_tol_zero_is_off():
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7560.0, mancini_targets=[7578.0],
        engine_levels=[7580.0], tol=0.0) == 7581.0


def test_exact_t2_level_allowed():
    # a rung exactly at t2 counts (delta 0 <= tol)
    assert snap_t2_to_real_level(
        t2=7578.0, t1=7560.0, mancini_targets=[7578.0],
        engine_levels=[], tol=4.0) == 7578.0


def test_handles_empty_and_none_sources():
    assert snap_t2_to_real_level(
        t2=7581.0, t1=7560.0, mancini_targets=None,
        engine_levels=None, tol=4.0) == 7581.0
