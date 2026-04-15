"""Intraday Price Action Context — Mancini-faithful trend detection.

Reads intraday direction from pure price action (no indicators):
- Swing structure (HH/HL = bullish, LH/LL = bearish)
- Recovery quality (weak bounces = bearish, strong = bullish)
- Session position (where is price in today's range)
- Elevator event status (sharp flush = POST_SELL_SETUP, allow FB Longs)

Critical invariant: POST_SELL_SETUP always overrides BEARISH_PRESSURE for longs.
This is Mancini's "two siblings" — elevator down → FB → squeeze.

States:
  NEUTRAL          — both directions allowed (default)
  BEARISH_PRESSURE — active selling / slow grind down, suppress FB Longs
  BULLISH_PRESSURE — active buying / grind up, suppress BD Shorts
  POST_SELL_SETUP  — elevator completed, FB Long is the money trade (ALWAYS allow)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional


class IntradayState(Enum):
    """Intraday directional context state."""

    NEUTRAL = auto()
    BEARISH_PRESSURE = auto()
    BULLISH_PRESSURE = auto()
    POST_SELL_SETUP = auto()


class IntradayContextTracker:
    """Track intraday price action context bar-by-bar.

    Uses 4 components to vote on direction:
      A. Session position (where is price in range)
      B. Swing structure (HH/HL vs LH/LL, double weight)
      C. Recovery quality (bounce magnitude after swing lows)
      D. Elevator event status (priority override)
    """

    def __init__(
        self,
        swing_order: int = 5,
        min_swing_pts: float = 3.0,
        weak_bounce_pts: float = 5.0,
        bounce_lookback: int = 3,
        elevator_recency_bars: int = 30,
        session_pos_bearish: float = 0.2,
        session_pos_bullish: float = 0.8,
        bearish_threshold: int = 3,
        bullish_threshold: int = 3,
    ):
        self._swing_order = swing_order
        self._min_swing_pts = min_swing_pts
        self._weak_bounce_pts = weak_bounce_pts
        self._bounce_lookback = bounce_lookback
        self._elevator_recency_bars = elevator_recency_bars
        self._session_pos_bearish = session_pos_bearish
        self._session_pos_bullish = session_pos_bullish
        self._bearish_threshold = bearish_threshold
        self._bullish_threshold = bullish_threshold

        # Internal state
        self._state = IntradayState.NEUTRAL
        self._high_buffer: list[float] = []
        self._low_buffer: list[float] = []
        self._swing_highs: list[tuple[int, float]] = []  # (bar_idx, price)
        self._swing_lows: list[tuple[int, float]] = []
        self._recent_bounces: list[float] = []
        self._tracking_bounce: bool = False
        self._bounce_base: float = 0.0
        self._bounce_peak: float = 0.0
        self._bar_count: int = 0

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._state = IntradayState.NEUTRAL
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._recent_bounces.clear()
        self._tracking_bounce = False
        self._bounce_base = 0.0
        self._bounce_peak = 0.0
        self._bar_count = 0

    @property
    def state(self) -> IntradayState:
        return self._state

    def update(
        self,
        bar_idx: int,
        high: float,
        low: float,
        close: float,
        elevator_event: object = None,
        elevator_active: bool = False,
        session_high: float = 0.0,
        session_low: float = 0.0,
    ) -> IntradayState:
        """Process one bar and return the current intraday state.

        Parameters
        ----------
        bar_idx : int
        high, low, close : float
        elevator_event : ElevatorEvent or None
            Completed elevator event (has end_idx).
        elevator_active : bool
            True if elevator is currently in progress (velocity still falling).
        session_high, session_low : float
            Session extremes for position calculation.
        """
        self._bar_count += 1
        self._high_buffer.append(high)
        self._low_buffer.append(low)

        # Detect swings
        self._detect_swings(bar_idx)

        # Track bounce quality
        self._track_bounce(close)

        # Compute state
        self._state = self._resolve_state(
            bar_idx, close, elevator_event, elevator_active,
            session_high, session_low,
        )
        return self._state

    def _detect_swings(self, bar_idx: int) -> None:
        """Detect swing highs and lows using a simple order-based method.

        A swing high is confirmed when we have `order` bars after the peak
        that are all lower. Same logic inverted for swing lows.
        No lookahead — we can only confirm a swing `order` bars after it happens.
        """
        order = self._swing_order
        n = len(self._high_buffer)
        if n < 2 * order + 1:
            return

        # Check if the bar at position (n - 1 - order) is a swing high
        # It needs `order` bars before and `order` bars after
        check_idx = n - 1 - order
        candidate_high = self._high_buffer[check_idx]
        candidate_low = self._low_buffer[check_idx]

        # Swing high: highest in the window
        is_swing_high = True
        for i in range(check_idx - order, check_idx + order + 1):
            if i == check_idx:
                continue
            if 0 <= i < n and self._high_buffer[i] >= candidate_high:
                is_swing_high = False
                break

        if is_swing_high:
            swing_bar = bar_idx - order
            # Check minimum swing size vs last swing low
            if self._swing_lows:
                swing_size = candidate_high - self._swing_lows[-1][1]
                if swing_size >= self._min_swing_pts:
                    self._swing_highs.append((swing_bar, candidate_high))
                    if len(self._swing_highs) > 5:
                        self._swing_highs = self._swing_highs[-5:]
            else:
                self._swing_highs.append((swing_bar, candidate_high))

        # Swing low: lowest in the window
        is_swing_low = True
        for i in range(check_idx - order, check_idx + order + 1):
            if i == check_idx:
                continue
            if 0 <= i < n and self._low_buffer[i] <= candidate_low:
                is_swing_low = False
                break

        if is_swing_low:
            swing_bar = bar_idx - order
            # Check minimum swing size vs last swing high
            if self._swing_highs:
                swing_size = self._swing_highs[-1][1] - candidate_low
                if swing_size >= self._min_swing_pts:
                    # Record bounce from previous swing low before starting new one
                    self._record_bounce()
                    self._swing_lows.append((swing_bar, candidate_low))
                    if len(self._swing_lows) > 5:
                        self._swing_lows = self._swing_lows[-5:]
                    # Start tracking bounce from this new low
                    self._tracking_bounce = True
                    self._bounce_base = candidate_low
                    self._bounce_peak = candidate_low
            else:
                self._record_bounce()
                self._swing_lows.append((swing_bar, candidate_low))
                self._tracking_bounce = True
                self._bounce_base = candidate_low
                self._bounce_peak = candidate_low

    def _track_bounce(self, close: float) -> None:
        """Track bounce magnitude after each swing low."""
        if not self._tracking_bounce:
            return

        if close > self._bounce_peak:
            self._bounce_peak = close

        # A new swing low was detected → record the bounce from the previous one
        # This is handled in _detect_swings when a new swing low appears.
        # Here we just track the running peak.

    def _record_bounce(self) -> None:
        """Record the current bounce magnitude and reset."""
        if self._tracking_bounce and self._bounce_peak > self._bounce_base:
            bounce_mag = self._bounce_peak - self._bounce_base
            self._recent_bounces.append(bounce_mag)
            if len(self._recent_bounces) > self._bounce_lookback:
                self._recent_bounces = self._recent_bounces[-self._bounce_lookback:]
        self._tracking_bounce = False
        self._bounce_peak = 0.0
        self._bounce_base = 0.0

    def _get_swing_structure(self) -> str:
        """Determine swing structure from last 2 confirmed swing highs and lows.

        Returns 'LH_LL' (bearish), 'HH_HL' (bullish), or 'neutral'.
        """
        if len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return "neutral"

        sh1, sh2 = self._swing_highs[-2][1], self._swing_highs[-1][1]
        sl1, sl2 = self._swing_lows[-2][1], self._swing_lows[-1][1]

        lower_highs = sh2 < sh1
        lower_lows = sl2 < sl1
        higher_highs = sh2 > sh1
        higher_lows = sl2 > sl1

        if lower_highs and lower_lows:
            return "LH_LL"
        if higher_highs and higher_lows:
            return "HH_HL"
        return "neutral"

    def _get_recovery_quality(self) -> str:
        """Assess recovery quality from recent bounce magnitudes.

        Returns 'weak', 'strong', or 'neutral'.
        """
        if not self._recent_bounces:
            return "neutral"
        avg_bounce = sum(self._recent_bounces) / len(self._recent_bounces)
        if avg_bounce < self._weak_bounce_pts:
            return "weak"
        return "strong"

    def _resolve_state(
        self,
        bar_idx: int,
        close: float,
        elevator_event: object,
        elevator_active: bool,
        session_high: float,
        session_low: float,
    ) -> IntradayState:
        """Resolve intraday state from all components.

        Priority system:
          1. Active elevator → BEARISH_PRESSURE (no knife catching)
          2. Completed elevator (recent) → POST_SELL_SETUP (the money trade)
          3. Vote from A + B + C → threshold check
        """
        # Priority 1: Active elevator — don't knife catch
        if elevator_active:
            return IntradayState.BEARISH_PRESSURE

        # Priority 2: Completed elevator — POST_SELL_SETUP
        if elevator_event is not None:
            end_idx = getattr(elevator_event, "end_idx", None)
            if end_idx is not None and bar_idx - end_idx <= self._elevator_recency_bars:
                return IntradayState.POST_SELL_SETUP

        # Need minimum bars before voting (let session establish)
        if self._bar_count < 2 * self._swing_order + 5:
            return IntradayState.NEUTRAL

        # Priority 3: Component votes
        bearish_votes = 0
        bullish_votes = 0

        # A: Session position
        session_range = session_high - session_low
        if session_range > 2.0:
            position = (close - session_low) / session_range
            if position < self._session_pos_bearish:
                bearish_votes += 1
            elif position > self._session_pos_bullish:
                bullish_votes += 1

        # B: Swing structure (double weight — primary detector)
        structure = self._get_swing_structure()
        if structure == "LH_LL":
            bearish_votes += 2
        elif structure == "HH_HL":
            bullish_votes += 2

        # C: Recovery quality
        recovery = self._get_recovery_quality()
        if recovery == "weak":
            bearish_votes += 1
        elif recovery == "strong":
            bullish_votes += 1

        # Threshold check
        if bearish_votes >= self._bearish_threshold:
            return IntradayState.BEARISH_PRESSURE
        if bullish_votes >= self._bullish_threshold:
            return IntradayState.BULLISH_PRESSURE

        return IntradayState.NEUTRAL

    def get_swing_snapshot(self) -> dict:
        """Return current swing structure for logging."""
        return {
            "structure": self._get_swing_structure(),
            "swing_highs": [(idx, round(price, 2)) for idx, price in self._swing_highs[-5:]],
            "swing_lows": [(idx, round(price, 2)) for idx, price in self._swing_lows[-5:]],
            "recovery_quality": self._get_recovery_quality(),
            "recent_bounces": [round(b, 2) for b in self._recent_bounces[-3:]],
            "state": self._state.name,
        }

