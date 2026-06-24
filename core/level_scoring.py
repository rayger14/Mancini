"""Level Quality Score (LQS) — continuous scoring for level-based trading decisions.

Every level gets scored 0-100 based on:
  Factor 1: Level Origin (0-40 pts) — PDL=40, MANCINI=35, MHL=30, etc.
  Factor 2: Structural Confirmation (0-25 pts) — mancini_confirmed, multi-touch, etc.
  Factor 3: Recency & Context (0-20 pts) — age, proximity to price, Mancini conviction
  Factor 4: Market Regime (0-15 pts) — VIX, term structure, SMA position

LQS drives:
  70-100: Trade aggressively (100% size, R:R 1.0, non-acceptance OK)
  50-69:  Trade normally (75% size, R:R 1.3)
  30-49:  Trade cautiously (50% size, R:R 1.5, clear acceptance required)
  15-29:  Shadow only (log phantom, don't trade)
  0-14:   Skip entirely
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional

from loguru import logger

from config.levels import Level, LevelType
from config.settings import StrategyParams


# Origin scores by LevelType.
# PDL = 40 (institutional order flow, 100% WR live).
# CUSTOM is treated as MANCINI_CALLED = 35 (curated by expert).
# MHL = 30 (proved significance via 20+ pt rally).
# INTRADAY_LOW = 25 (crash bottom with confirmed recovery).
# HORIZONTAL_SR = 15 (structural but weaker).
# SWING_LOW = 5 (noise unless confirmed).
# CLUSTER_LOW = 0 (0% WR live, excluded from FB).
_ORIGIN_SCORES: dict[LevelType, int] = {
    LevelType.PRIOR_DAY_LOW: 45,    # Highest quality — 100% WR live, institutional anchor
    LevelType.PRIOR_DAY_HIGH: 45,
    LevelType.CUSTOM: 30,           # Mancini-called levels (lower than PDL unless confirmed)
    LevelType.MULTI_HOUR_LOW: 35,   # Proved significance via 20+ pt rally
    LevelType.MULTI_HOUR_HIGH: 35,
    LevelType.INTRADAY_LOW: 25,     # Crash bottom with confirmed recovery
    LevelType.HORIZONTAL_SR: 12,    # Structural but weaker than time-based
    LevelType.SWING_LOW: 5,         # Noise unless confirmed by other factors
    LevelType.SWING_HIGH: 5,
    LevelType.VWAP: 5,
    LevelType.CLUSTER_LOW: 0,       # Base 0, but CAN reach 30 via confirmation (8+ touches = shelf)
    LevelType.CLUSTER_HIGH: 0,
}


class LevelQualityScorer:
    """Computes Level Quality Score (LQS) for each level.

    Called by SignalAggregator when qualifying FB/BD signals.
    Uses level metadata, market context, and session context.

    Parameters
    ----------
    strategy_params : StrategyParams
        Strategy configuration (contains LQS thresholds).
    """

    def __init__(self, strategy_params: StrategyParams) -> None:
        self.strategy_params = strategy_params

    def compute_lqs(
        self,
        level: Level,
        market_data: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Compute Level Quality Score (0-100) for a level.

        Parameters
        ----------
        level : Level
            The level being scored.
        market_data : dict, optional
            Market snapshot from ``fetch_market_snapshot()`` (vix, vix_term_structure, etc.).
        session_context : dict, optional
            Session context with keys: session_date, current_price, session_high,
            session_low, bar_count.

        Returns
        -------
        int
            LQS score clamped to [0, 100].
        """
        score = 0
        score += self._origin_score(level)
        score += self._confirmation_score(level)
        score += self._recency_score(level, session_context)
        score += self._regime_score(market_data)
        return min(100, max(0, score))

    def get_trade_params(self, lqs: int) -> Dict[str, Any]:
        """Return trade parameters driven by LQS.

        Parameters
        ----------
        lqs : int
            Level Quality Score (0-100).

        Returns
        -------
        dict
            Keys: size_factor (float), min_rr (float), acceptance_mode (str).
        """
        # Tiers calibrated from backtest data:
        # - PDL (origin 45) + yesterday (8) = 53 → trade aggressively
        # - MHL (origin 35) + today (10) = 45 → trade normally
        # - Mancini-confirmed PDL = 45+15+8 = 68 → trade aggressively
        # - SWING_LOW (5) + today (10) = 15 → shadow only
        # - CLUSTER_LOW (0) + today (10) = 10 → shadow only
        if lqs >= self.strategy_params.lqs_full_size_threshold:  # default 55
            return {
                "size_factor": 1.0,
                "min_rr": 1.0,
                "acceptance_mode": "non_acceptance_ok",
            }
        elif lqs >= self.strategy_params.lqs_min_trade_threshold:  # default 25
            return {
                "size_factor": 0.75,
                "min_rr": 1.3,
                "acceptance_mode": "any",
            }
        elif lqs >= self.strategy_params.lqs_shadow_threshold:  # default 10
            return {
                "size_factor": 0.0,
                "min_rr": 0.0,
                "acceptance_mode": "shadow_only",
            }
        else:
            return {
                "size_factor": 0.0,
                "min_rr": 0.0,
                "acceptance_mode": "skip",
            }

    # ------------------------------------------------------------------
    # Sub-scorers (separate methods for testability)
    # ------------------------------------------------------------------

    def _origin_score(self, level: Level) -> int:
        """Factor 1: Level Origin (0-40 points).

        Where did this level come from? PDL=40, CUSTOM/Mancini=35, MHL=30, etc.
        """
        return _ORIGIN_SCORES.get(level.level_type, 0)

    def _confirmation_score(self, level: Level) -> int:
        """Factor 2: Structural Confirmation (0-25 points).

        How has price behaved around this level?
        - Mancini-confirmed (label contains 'mancini' or type is CUSTOM with engine match): +15
        - Multi-touch (8+ touches): +10
        - Multi-touch (4-7 touches): +5
        - Validated rally (20+ pts from this low): +10
        - Tested and held (prior sweep survived): +5
        """
        score = 0

        # Mancini-confirmed: ONLY when both engine AND Mancini agree on a level
        # (set by mancini_overlay when engine level is within 3 pts of Mancini call)
        # A CUSTOM (Mancini-only) level does NOT get this bonus — it needs engine validation.
        mancini_confirmed = getattr(level, "mancini_confirmed", False)
        if mancini_confirmed and level.level_type != LevelType.CUSTOM:
            # Engine level confirmed by Mancini = highest conviction
            score += 15
        elif mancini_confirmed and level.level_type == LevelType.CUSTOM:
            # Mancini-only level — lower bonus since no engine validation
            score += 5

        # Multi-touch scoring — Mancini's "shelf of lows" concept
        # A shelf is a horizontal zone tested many times = institutional interest
        # 8+ touches on a CLUSTER_LOW transforms noise into a real shelf (Mancini: 4354 mentions)
        if level.touch_count >= 15:
            score += 15  # Monster shelf — "ATM machine level"
        elif level.touch_count >= 8:
            score += 10  # Real shelf — Mancini would trade this
        elif level.touch_count >= 4:
            score += 5   # Moderate structure

        # Validated rally: 20+ pt rally from this low proves demand
        if level.rally_from_low_pts >= 20.0:
            score += 10

        # Tested and held: battle-tested level
        if level.tested_and_held:
            score += 5

        # Cross-source confluence: independent sources (engine + Mancini +
        # pivot) agreeing on the same price is strong confirmation and filters
        # each source's noise. A bare swing that 3 sources land on becomes
        # tradeable; a lone pivot does not.
        source_count = getattr(level, "source_count", 1)
        if source_count >= 3:
            score += 10
        elif source_count >= 2:
            score += 5

        # Cap at 30 (raised from 25 to let elite shelves score higher)
        return min(30, score)

    def _recency_score(
        self,
        level: Level,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Factor 3: Recency & Context (0-20 points).

        How relevant is this level RIGHT NOW?
        - Today's session level: +10
        - Yesterday's level: +8
        - 2-3 days old: +5
        - 5+ days old: +2
        - Level near session low (within 10 pts): +5
        - Level far from price (50+ pts away): 0
        """
        score = 0
        if session_context is None:
            return score

        # Age-based scoring
        session_date = session_context.get("session_date")
        if session_date is not None and level.origin_date is not None:
            if isinstance(session_date, str):
                try:
                    session_date = date.fromisoformat(session_date)
                except (ValueError, TypeError):
                    session_date = None
            elif isinstance(session_date, datetime):
                session_date = session_date.date()

            if session_date is not None:
                age_days = (session_date - level.origin_date).days
                if age_days <= 0:
                    score += 10  # Today's level
                elif age_days == 1:
                    score += 8   # Yesterday
                elif age_days <= 3:
                    score += 5   # 2-3 days old
                else:
                    score += 2   # 5+ days old

        # If no origin_date, use created_at timestamp
        elif session_date is not None and level.created_at is not None:
            try:
                level_date = level.created_at.date() if isinstance(level.created_at, datetime) else None
                if level_date is not None:
                    if isinstance(session_date, str):
                        session_date = date.fromisoformat(session_date)
                    elif isinstance(session_date, datetime):
                        session_date = session_date.date()
                    if session_date is not None:
                        age_days = (session_date - level_date).days
                        if age_days <= 0:
                            score += 10
                        elif age_days == 1:
                            score += 8
                        elif age_days <= 3:
                            score += 5
                        else:
                            score += 2
            except (ValueError, TypeError, AttributeError):
                pass

        # Proximity to session low (price is near this level = about to test)
        current_price = session_context.get("current_price")
        session_low = session_context.get("session_low")
        if current_price is not None and level.price > 0:
            distance = abs(current_price - level.price)
            if session_low is not None and abs(level.price - session_low) <= 10:
                score += 5
            elif distance >= 50:
                pass  # 0 points for far-away levels

        # Cap at 20
        return min(20, score)

    def _regime_score(self, market_data: Optional[Dict[str, Any]] = None) -> int:
        """Factor 4: Market Regime (0-15 points).

        What's the market doing?
        - VIX > 25 (high fear): +10
        - VIX 20-25 (moderate): +5
        - VIX < 20: 0
        - VIX term structure inverted (>1.0): +5
        """
        if market_data is None:
            return 0

        score = 0

        # VIX level
        vix = market_data.get("vix")
        if vix is not None:
            if vix > 25:
                score += 10
            elif vix >= 20:
                score += 5

        # VIX term structure: vix9d/vix > 1.0 = short-term fear elevated
        vix_ts = market_data.get("vix_term_structure")
        if vix_ts is not None and vix_ts > 1.0:
            score += 5

        # Cap at 15
        return min(15, score)
