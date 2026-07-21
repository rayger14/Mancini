"""Level-to-level exit management (75/15/10 split with prior-day-low runner trail).

Mancini's actual method (from 500 Substack posts):
  Entry → full position, stop below sweep low
  Target 1 (R1) → exit 75%, move stop to several pts under breakeven
  Target 2 (R2) → exit 15%, leave 10% runner, initiate trailing-stop methodology
  Runner → trail under prior day's RTH low *or* below the most recent
           structural swing low (whichever is higher / ratchets up)
  Runner carries overnight/multi-day until that trail is taken out.

Two Mancini quotes drive the post-T2 behaviour implemented here:
  2025-08-05: "lock in 75% profits at the first level, leave a 25% runner,
    then lock in more at second level up, and let a 10% runner go and
    initiate the trailing stop methodology."
  2025-05-14: "I lock in 75% profits at the first level up, leaving a 25%
    runner. I then lock in more at the 2nd level up, leaving a 10% runner
    to trail, at this point, I typically move my stop up, often to below
    wherever structure is."
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional, Tuple

from config.settings import (
    ExitParams,
    ESContractSpec,
    StrategyParams,
    DEFAULT_EXIT,
    DEFAULT_CONTRACT,
    DEFAULT_STRATEGY,
)


class ExitPhase(Enum):
    """Current phase of the exit management."""

    INITIAL = auto()      # Full position, initial stop
    AFTER_T1 = auto()     # 75% exited, stop near breakeven
    AFTER_T2 = auto()     # Runner only, trailing under prior day's low
    CLOSED = auto()       # All positions closed


@dataclass
class ExitAction:
    """Describes an exit action to take."""

    contracts_to_close: int
    exit_price: float
    new_stop: float
    new_phase: ExitPhase
    reason: str


@dataclass
class TradePosition:
    """Tracks an open trade's position and exit state."""

    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    total_contracts: int
    remaining_contracts: int
    phase: ExitPhase = ExitPhase.INITIAL
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = float("inf")
    realized_pnl_pts: float = 0.0
    direction: str = "long"  # "long" or "short"
    # Prior day's RTH low — set by ib_runner at EOD for runner trailing.
    # When set, the runner stop trails under this level instead of fixed pts.
    prior_day_low: float = 0.0
    # Prior day's RTH high — for short runner trailing
    prior_day_high: float = 0.0
    # Track whether T1/T2 were hit (for accurate exit_reason on final close)
    t1_hit: bool = False
    t2_hit: bool = False
    is_double_dip: bool = False  # True if this is a double-dip re-entry
    # Rolling history of recent (high, low) bars used by the post-T2
    # structure-based runner trail. Populated by ExitManager.update() each
    # bar. Sized to params.structure_trail_lookback_bars so the deque keeps
    # a fixed-window view of the most recent structure.
    bar_history: Deque[Tuple[float, float]] = field(default_factory=deque)

    def to_snapshot(self) -> dict:
        """JSON-safe dict of EVERY field — the restart-survival contract.
        Enums by name, deque as list; +/-inf survive via string sentinels
        (json.dumps(allow_nan) is not guaranteed at every call site)."""
        import dataclasses as _dc
        out = {}
        for f in _dc.fields(self):
            v = getattr(self, f.name)
            if f.name == "phase":
                v = v.name
            elif f.name == "bar_history":
                v = [list(t) for t in v]
            elif isinstance(v, float) and v == float("inf"):
                v = "inf"
            elif isinstance(v, float) and v == float("-inf"):
                v = "-inf"
            out[f.name] = v
        return out

    @classmethod
    def from_snapshot(cls, snap):
        """Rebuild from to_snapshot() output; None on anything malformed —
        callers fall back to log-based reconstruction."""
        import dataclasses as _dc
        from collections import deque as _deque
        if not isinstance(snap, dict) or "entry_price" not in snap:
            return None
        try:
            kwargs = {}
            for f in _dc.fields(cls):
                if f.name not in snap:
                    continue
                v = snap[f.name]
                if f.name == "phase":
                    v = ExitPhase[v]
                elif f.name == "bar_history":
                    v = _deque([tuple(t) for t in v], maxlen=30)
                elif v == "inf":
                    v = float("inf")
                elif v == "-inf":
                    v = float("-inf")
                elif f.name != "direction" and isinstance(
                        getattr(cls, "__dataclass_fields__")[f.name].default,
                        (int, float)) and isinstance(v, str):
                    return None
                kwargs[f.name] = v
            if not isinstance(kwargs.get("entry_price"), (int, float)):
                return None
            obj = cls(**kwargs)
            # __post_init__ re-seeds the price trackers from entry — restore
            # the snapshotted values after construction.
            for k in ("highest_price_since_entry", "lowest_price_since_entry"):
                if k in kwargs:
                    setattr(obj, k, kwargs[k])
            return obj
        except (KeyError, TypeError, ValueError):
            return None

    def __post_init__(self):
        self.highest_price_since_entry = self.entry_price
        self.lowest_price_since_entry = self.entry_price

    @property
    def is_open(self) -> bool:
        return self.remaining_contracts > 0 and self.phase != ExitPhase.CLOSED


class ExitManager:
    """Manages the 75/15/10 exit strategy for a single trade.

    After T1 hit, stop moves to several pts under breakeven.
    After T2 hit (or at EOD), runner trails under prior day's low.
    """

    def __init__(
        self,
        params: ExitParams = DEFAULT_EXIT,
        contract: ESContractSpec = DEFAULT_CONTRACT,
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
    ):
        self.params = params
        self.contract = contract
        self.strategy_params = strategy_params

        # When mancini_exit_scaling is enabled, override the exit fractions
        # from StrategyParams so the 75/15/10 split is used instead of
        # whatever ExitParams has (which may be a different split).
        if strategy_params.mancini_exit_scaling:
            self._t1_exit_fraction = strategy_params.mancini_t1_exit_pct
            self._t2_exit_fraction = strategy_params.mancini_t2_exit_pct
            self._runner_fraction = strategy_params.mancini_runner_pct
        else:
            self._t1_exit_fraction = params.t1_exit_fraction
            self._t2_exit_fraction = params.t2_exit_fraction
            self._runner_fraction = params.runner_fraction

    def create_position(
        self,
        entry_price: float,
        stop_price: float,
        target_1: float,
        target_2: float,
        contracts: int,
        direction: str = "long",
        is_double_dip: bool = False,
    ) -> TradePosition:
        """Create a new trade position."""
        return TradePosition(
            entry_price=entry_price,
            stop_price=stop_price,
            target_1=target_1,
            target_2=target_2,
            total_contracts=contracts,
            remaining_contracts=contracts,
            direction=direction,
            is_double_dip=is_double_dip,
        )

    def update_prior_day_low(self, position: TradePosition, prior_day_low: float) -> Optional[ExitAction]:
        """Update the runner's trailing stop to under the prior day's low.

        Called by ib_runner at EOD. Mancini: "Move stop under daily low at
        end of day." This only applies to runners (AFTER_T1 or AFTER_T2).
        """
        if not position.is_open:
            return None
        if position.phase not in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
            return None

        position.prior_day_low = prior_day_low
        buffer = self.params.runner_prior_day_low_buffer_pts

        if position.direction == "long":
            new_stop = prior_day_low - buffer
            # Only ratchet up, never down
            if new_stop > position.stop_price:
                position.stop_price = new_stop
                return ExitAction(
                    contracts_to_close=0,
                    exit_price=0.0,
                    new_stop=new_stop,
                    new_phase=position.phase,
                    reason=f"Runner trail to prior day low {prior_day_low:.2f}",
                )
        else:
            new_stop = prior_day_low + buffer  # For shorts, trail above prior day's HIGH
            # For short runners, we'd use prior_day_high — but use prior_day_low
            # as fallback if high not set
            if position.prior_day_high > 0:
                new_stop = position.prior_day_high + buffer
            if new_stop < position.stop_price:
                position.stop_price = new_stop
                return ExitAction(
                    contracts_to_close=0,
                    exit_price=0.0,
                    new_stop=new_stop,
                    new_phase=position.phase,
                    reason=f"Runner trail to prior day high {new_stop:.2f}",
                )
        return None

    def update(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Evaluate current bar against position and return an exit action if triggered.

        Priority: stop loss > T1 > T2 > trail.
        Direction-aware: long checks low<=stop, short checks high>=stop.
        """
        if not position.is_open:
            return None

        # Update tracking
        if high > position.highest_price_since_entry:
            position.highest_price_since_entry = high
        if low < position.lowest_price_since_entry:
            position.lowest_price_since_entry = low

        # Append this bar to the rolling history used by the structure-based
        # runner trail. Sized to structure_trail_lookback_bars so older bars
        # naturally roll off.
        self._record_bar(position, high, low)

        # 1. Check stop loss (highest priority)
        if position.direction == "short":
            if high >= position.stop_price:
                return self._stop_out(position)
        else:
            if low <= position.stop_price:
                return self._stop_out(position)

        # 2. Phase-specific logic
        if position.direction == "short":
            if position.phase == ExitPhase.INITIAL:
                return self._check_t1_short(position, low, close)
            elif position.phase == ExitPhase.AFTER_T1:
                return self._check_t2_short(position, low, close)
            elif position.phase == ExitPhase.AFTER_T2:
                return self._check_trail_short(position, high, low, close)
        else:
            if position.phase == ExitPhase.INITIAL:
                return self._check_t1(position, high, close)
            elif position.phase == ExitPhase.AFTER_T1:
                return self._check_t2(position, high, close)
            elif position.phase == ExitPhase.AFTER_T2:
                return self._check_trail(position, high, low, close)

        return None

    # ------------------------------------------------------------------
    # Phase handlers (long)
    # ------------------------------------------------------------------

    def _stop_out(self, position: TradePosition) -> ExitAction:
        """Close entire position at stop."""
        contracts = position.remaining_contracts
        if position.direction == "short":
            pnl = (position.entry_price - position.stop_price) * contracts
        else:
            pnl = (position.stop_price - position.entry_price) * contracts
        position.realized_pnl_pts += pnl
        position.remaining_contracts = 0

        if position.phase in (ExitPhase.AFTER_T1, ExitPhase.AFTER_T2):
            if position.t2_hit:
                reason = "Runner stopped after T1+T2"
            elif position.t1_hit:
                reason = "Runner stopped after T1"
            else:
                reason = "Trailing stop hit"
        else:
            reason = "Stop loss hit"

        position.phase = ExitPhase.CLOSED
        return ExitAction(
            contracts_to_close=contracts,
            exit_price=position.stop_price,
            new_stop=0.0,
            new_phase=ExitPhase.CLOSED,
            reason=reason,
        )

    def _check_t1(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 1 is reached → exit 75%, stop to under breakeven.

        Mancini: "Lock in 75% profits at first level up. Move stop several
        points under break-even. Will not let the entire trade go back red."
        """
        if high >= position.target_1:
            # math.floor (NOT round) so a 2-contract trade closes 1 at T1,
            # not 2 — Python's banker's rounding turned 2*0.75=1.5 into 2,
            # which closed the entire position and bypassed the runner.
            contracts_to_exit = math.floor(
                position.total_contracts * self._t1_exit_fraction
            )
            # Guard tiny sizes: with 1 contract, floor(0.75)=0 would skip the
            # exit entirely. Fall back to closing the 1 contract — there's no
            # runner to preserve when starting from 1.
            if contracts_to_exit == 0 and position.total_contracts == 1:
                contracts_to_exit = 1
            contracts_to_exit = min(contracts_to_exit, position.remaining_contracts)

            if contracts_to_exit <= 0:
                return None

            pnl = (position.target_1 - position.entry_price) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Mancini: "several points under break-even"
            # breakeven_buffer_pts is negative (e.g., -3.0) = below breakeven
            new_stop = position.entry_price + self.params.breakeven_buffer_pts
            # If prior_day_low is set and is lower, use that (wider stop for runner)
            if position.prior_day_low > 0:
                pdl_stop = position.prior_day_low - self.params.runner_prior_day_low_buffer_pts
                new_stop = min(new_stop, pdl_stop)  # use the wider (lower) stop

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T1
            position.t1_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_1,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T1,
                reason=f"Target 1 hit ({position.target_1:.2f})",
            )
        return None

    def _check_t2(
        self, position: TradePosition, high: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 2 is reached → scale down another 15%, leaving the runner.

        Mancini 2025-08-05: "lock in 75% profits at the first level, leave a 25%
        runner, then lock in more at second level up, and let a 10% runner go
        and initiate the trailing stop methodology."

        Concretely, with the default 75/15/10 split, T1 already exited 75%. At
        T2 we exit another ``t2_exit_fraction`` (15% of TOTAL contracts), so
        only the runner (10%) remains. We always preserve at least
        ``runner_fraction`` of the original size — never oversell — and
        gracefully fall back to a phase-only transition if rounding would
        leave nothing left.

        After T2, the runner stop migrates to (in priority order):
          1. Prior-day low / structural swing low (whichever ratchets up)
          2. Fallback fixed-distance trailing if neither is available.
        """
        # Fallback intraday trail (ratchet up) if no prior_day_low set
        # For runners (AFTER_T1), use base trailing_stop_pts without aggressive tightening
        # Double-dip entries use wider trail (dd_trail_pts_after_t1) to give room
        if position.prior_day_low <= 0:
            if position.is_double_dip:
                trail_pts = self.strategy_params.dd_trail_pts_after_t1
            else:
                trail_pts = self.params.trailing_stop_pts
            new_trail = position.highest_price_since_entry - trail_pts
            if new_trail > position.stop_price:
                position.stop_price = new_trail

        if high >= position.target_2:
            # Mancini's T2 scale-down: sell another `t2_exit_fraction` of the
            # ORIGINAL total. With 4 contracts and t2_exit_fraction=0.15 this
            # rounds to 1 contract — combined with the 3-contract T1 that
            # leaves a 1-contract runner (~25% of size, but the smallest
            # tradeable runner for a tiny base). With bigger sizes the math
            # cleans up: e.g. 20 contracts → 15 T1 + 3 T2 + 2 runner.
            t2_target_exit = math.floor(
                position.total_contracts * self._t2_exit_fraction
            )
            runner_floor = max(
                1, math.floor(position.total_contracts * self._runner_fraction)
            )
            # Never oversell — always preserve at least the runner floor.
            max_sellable = max(0, position.remaining_contracts - runner_floor)
            contracts_to_exit = min(t2_target_exit, max_sellable)

            if contracts_to_exit <= 0:
                # Nothing left to scale (already at runner size). Still
                # transition phase so the structure trail kicks in.
                position.phase = ExitPhase.AFTER_T2
                position.t2_hit = True
                # Initialise structure-trail stop on phase transition so the
                # 25%→10% slice carries the same structural trail going
                # forward (if it never sells, the phase-only T2 still gets
                # the better trail logic).
                struct_stop = self._compute_structure_trail_stop(position)
                if struct_stop is not None and struct_stop > position.stop_price:
                    position.stop_price = struct_stop
                return None

            pnl = (position.target_2 - position.entry_price) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Runner stop after T2. Priority:
            #   1. Prior-day low (if set) — Mancini's overnight trail.
            #   2. Structure trail (if enabled) — Mancini's "below wherever
            #      structure is". Often returns None at the moment T2 fires
            #      because no swing has formed yet; subsequent bars in
            #      `_check_trail` will ratchet it up as structure emerges.
            #   3. Legacy fixed-distance trail — only used when both PDL
            #      and structure trail are unavailable (e.g. backtest before
            #      the EOD hook fires and structure_trail_enabled=False).
            # The ratchet-up guard ensures we never lower the stop below the
            # post-T1 level.
            new_stop = position.stop_price  # ratchet-up baseline
            struct_stop = self._compute_structure_trail_stop(position)
            if struct_stop is not None:
                new_stop = max(new_stop, struct_stop)
            if position.prior_day_low > 0:
                pdl_stop = position.prior_day_low - self.params.runner_prior_day_low_buffer_pts
                new_stop = max(new_stop, pdl_stop)
            elif not self.params.structure_trail_enabled and struct_stop is None:
                # Last-resort fallback: structure trail is OFF and no PDL —
                # use the legacy fixed-distance trail.
                fallback = self._compute_trail_stop(position, high)
                new_stop = max(new_stop, fallback)

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T2
            position.t2_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_2,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T2,
                reason=f"Target 2 hit ({position.target_2:.2f})",
            )
        return None

    def _check_trail(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Update trailing stop for the long runner after T2.

        Priority (highest stop wins — we only ratchet up, never down):
          1. Structure trail (preferred when enabled): most recent swing low
             minus a small buffer. Per Mancini 2025-05-14: "I typically move
             my stop up, often to below wherever structure is."
          2. Prior-day low (set by EOD hook) minus the runner buffer.
          3. Legacy fixed-distance fallback — only used when BOTH the
             structure trail is disabled AND no PDL is set. With structure
             trail enabled, we deliberately do NOT use the fixed-distance
             trail because it would clamp the stop above swing-based levels
             and defeat the purpose of structural trailing.
        """
        # 1) Structure-based trail — preferred post-T2 behaviour.
        if self.params.structure_trail_enabled:
            struct_stop = self._compute_structure_trail_stop(position)
            if struct_stop is not None and struct_stop > position.stop_price:
                position.stop_price = struct_stop
            # When structure trail is enabled, it is authoritative for the
            # runner — we never fall through to the fixed-distance trail.
            # If no swing has formed yet, we simply hold the current stop
            # (typically the post-T1 / post-T2 baseline) until structure
            # emerges. PDL ratchets continue to happen via update_prior_day_low.
            return None

        # 2) Prior-day low: Mancini's overnight trail — only updated at EOD,
        #    so we don't ratchet it per-bar here. The update_prior_day_low()
        #    EOD hook handles ratcheting to a higher PDL on subsequent days.
        if position.prior_day_low > 0:
            return None

        # 3) Fixed-distance fallback (structure trail OFF, no PDL): the
        #    legacy behaviour preserved for backwards compatibility with
        #    backtests that disable structure trailing.
        if position.is_double_dip:
            trail_pts = self.strategy_params.dd_trail_pts_after_t1
        else:
            trail_pts = self.params.trailing_stop_pts
        new_trail = position.highest_price_since_entry - trail_pts
        if new_trail > position.stop_price:
            position.stop_price = new_trail
        return None

    def _compute_trail_stop(self, position: TradePosition, high: float) -> float:
        """Compute fallback trailing stop distance based on profit (long)."""
        profit = high - position.entry_price
        trail_pts = self.params.trailing_stop_pts

        for threshold, tighter_trail in self.params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter_trail

        return high - trail_pts

    # ------------------------------------------------------------------
    # Short-side phase handlers
    # ------------------------------------------------------------------

    # Short-side T1/T2 use the same math.floor semantics as longs (see
    # the long _check_t1/_check_t2 above for the rationale).
    def _check_t1_short(
        self, position: TradePosition, low: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 1 is reached for short (price drops to target)."""
        if low <= position.target_1:
            contracts_to_exit = math.floor(
                position.total_contracts * self._t1_exit_fraction
            )
            if contracts_to_exit == 0 and position.total_contracts == 1:
                contracts_to_exit = 1
            contracts_to_exit = min(contracts_to_exit, position.remaining_contracts)

            if contracts_to_exit <= 0:
                return None

            pnl = (position.entry_price - position.target_1) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            # Stop to several pts above breakeven (for shorts, above = tighter)
            # Use short-specific buffer (wider than longs to survive post-T1 bounces)
            short_buf = getattr(self.params, "short_breakeven_buffer_pts", self.params.breakeven_buffer_pts)
            new_stop = position.entry_price - short_buf
            if position.prior_day_high > 0:
                pdh_stop = position.prior_day_high + self.params.runner_prior_day_low_buffer_pts
                new_stop = max(new_stop, pdh_stop)  # use the wider (higher) stop for short

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T1
            position.t1_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_1,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T1,
                reason=f"Target 1 hit ({position.target_1:.2f})",
            )
        return None

    def _check_t2_short(
        self, position: TradePosition, low: float, close: float
    ) -> Optional[ExitAction]:
        """Check if Target 2 is reached for short."""
        # Fallback intraday trail (ratchet down) if no prior_day_high set.
        # Delayed activation: only trail once price makes a new low below T1,
        # confirming the trend continues. This prevents the trail from
        # ratcheting the stop down during the initial T1 drop, then getting
        # clipped by the post-T1 bounce.
        if position.prior_day_high <= 0 and low < position.target_1:
            new_trail = position.lowest_price_since_entry + self.params.trailing_stop_pts
            if new_trail < position.stop_price:
                position.stop_price = new_trail

        if low <= position.target_2:
            runner_contracts = max(
                1, math.floor(position.total_contracts * self._runner_fraction)
            )
            contracts_to_exit = position.remaining_contracts - runner_contracts

            if contracts_to_exit <= 0:
                position.phase = ExitPhase.AFTER_T2
                return None

            pnl = (position.entry_price - position.target_2) * contracts_to_exit
            position.realized_pnl_pts += pnl
            position.remaining_contracts -= contracts_to_exit

            if position.prior_day_high > 0:
                new_stop = position.prior_day_high + self.params.runner_prior_day_low_buffer_pts
                new_stop = min(new_stop, position.stop_price)
            else:
                new_stop = self._compute_trail_stop_short(position, low)
                new_stop = min(new_stop, position.stop_price)

            position.stop_price = new_stop
            position.phase = ExitPhase.AFTER_T2
            position.t2_hit = True

            return ExitAction(
                contracts_to_close=contracts_to_exit,
                exit_price=position.target_2,
                new_stop=new_stop,
                new_phase=ExitPhase.AFTER_T2,
                reason=f"Target 2 hit ({position.target_2:.2f})",
            )
        return None

    def _check_trail_short(
        self, position: TradePosition, high: float, low: float, close: float
    ) -> Optional[ExitAction]:
        """Update trailing stop for short runner."""
        if position.prior_day_high > 0:
            pass  # Trail updated at EOD only
        else:
            # Fallback: intraday trailing with base trailing_stop_pts (no tightening for runners).
            # Only activate once price is below T1 (confirming trend continuation).
            if low < position.target_1:
                new_trail = position.lowest_price_since_entry + self.params.trailing_stop_pts
                if new_trail < position.stop_price:
                    position.stop_price = new_trail
        return None

    def _compute_trail_stop_short(self, position: TradePosition, low: float) -> float:
        """Compute fallback trailing stop for short (above lowest low)."""
        profit = position.entry_price - low
        trail_pts = self.params.trailing_stop_pts

        for threshold, tighter_trail in self.params.trailing_tighten_thresholds:
            if profit >= threshold:
                trail_pts = tighter_trail

        return low + trail_pts

    # ------------------------------------------------------------------
    # Structure-based trail helpers (Mancini 2025-05-14)
    # ------------------------------------------------------------------

    def _record_bar(self, position: TradePosition, high: float, low: float) -> None:
        """Append the latest bar to the position's rolling history.

        Keeps ``structure_trail_lookback_bars`` worth of (high, low) tuples,
        sized once at first insertion. The deque is bounded by ``maxlen`` so
        old bars roll off automatically and the structure detector always
        sees a fixed-window view of recent price action.
        """
        lookback = max(1, int(self.params.structure_trail_lookback_bars))
        if position.bar_history.maxlen != lookback:
            # First insertion (or lookback changed) — rebuild the deque with
            # the configured maxlen, preserving any earlier history.
            position.bar_history = deque(position.bar_history, maxlen=lookback)
        position.bar_history.append((float(high), float(low)))

    def _find_recent_swing_low(self, position: TradePosition) -> Optional[float]:
        """Locate the most recent significant swing low in bar history.

        A swing low is a bar whose ``low`` is the minimum within a
        ``structure_trail_swing_order`` window on each side. We scan the
        history newest-first and return the first qualifying low.

        Returns ``None`` if there isn't enough history yet (need
        ``2*order + 1`` bars at minimum) or no swing low exists in the
        window.
        """
        order = max(1, int(self.params.structure_trail_swing_order))
        bars = list(position.bar_history)
        n = len(bars)
        if n < 2 * order + 1:
            return None

        # Scan from newest-confirmable candidate (n-1-order) back to oldest
        # confirmable (order). A bar at index i is a swing low if every bar
        # within `order` on either side has a low >= bars[i].low.
        for i in range(n - 1 - order, order - 1, -1):
            candidate_low = bars[i][1]
            is_swing = True
            for j in range(i - order, i + order + 1):
                if j == i:
                    continue
                if bars[j][1] < candidate_low:
                    is_swing = False
                    break
            if is_swing:
                return candidate_low
        return None

    def _compute_structure_trail_stop(
        self, position: TradePosition
    ) -> Optional[float]:
        """Compute the structure-based runner stop for a long position.

        Returns the candidate stop = ``swing_low - structure_trail_buffer_pts``
        or ``None`` when the feature is disabled, the position is short, or
        no swing low can be identified yet.

        Callers are responsible for the ratchet-up check — this only proposes
        a stop, never lowers an existing one.
        """
        if not self.params.structure_trail_enabled:
            return None
        if position.direction != "long":
            # Structure trail for shorts would mirror this with swing-highs;
            # out of scope for the current Mancini quote (longs only).
            return None
        swing_low = self._find_recent_swing_low(position)
        if swing_low is None:
            return None
        return swing_low - self.params.structure_trail_buffer_pts
