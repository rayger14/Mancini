"""QuantConnect adapter for Mancini MES long-only day trading strategy.

Thin wrapper: all 3,100 lines of strategy logic remain unchanged.
This file only handles QC data ingestion, order routing, and session lifecycle.

Usage (LEAN CLI):
    lean cloud backtest "ManciniMES"
    lean cloud live "ManciniMES" --brokerage "Paper Trading"
"""

from AlgorithmImports import *

import pandas as pd
import numpy as np
from datetime import timedelta, time as dt_time

# --- Strategy imports (unchanged modules) ---
from config.settings import (
    MES_CONTRACT,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
    SessionTimes,
)
from strategy.mancini_long import ManciniLongStrategy, BarResult


# Production session times: chop zone 13-15 (locked config)
PROD_SESSION = SessionTimes(
    chop_zone_start=dt_time(13, 0),
    chop_zone_end=dt_time(15, 0),
)

# Production min R:R ratio
PROD_MIN_RR = 1.0


class ManciniAlgorithm(QCAlgorithm):
    """Mancini Method MES long-only strategy on QuantConnect."""

    def initialize(self) -> None:
        # --- Backtest date range ---
        self.set_start_date(2024, 2, 5)
        self.set_end_date(2026, 2, 5)
        self.set_cash(100_000)

        # --- Add MES futures (1-min resolution) ---
        self.mes = self.add_future(
            Futures.Indices.MICRO_SP500_E_MINI,
            Resolution.MINUTE,
            data_normalization_mode=DataNormalizationMode.RAW,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            contract_depth_offset=0,
            extended_market_hours=False,
        )
        self.mes.set_filter(timedelta(0), timedelta(90))
        self._contract_symbol = None

        # --- Create strategy (unchanged production params) ---
        self.strategy = ManciniLongStrategy(
            strategy_params=DEFAULT_STRATEGY,
            elevator_params=DEFAULT_ELEVATOR,
            exit_params=DEFAULT_EXIT,
            risk_params=DEFAULT_RISK,
            session_times=PROD_SESSION,
            contract=MES_CONTRACT,
            min_rr_ratio=PROD_MIN_RR,
        )

        # --- Warmup: 1 day for prior-day level initialization ---
        self.set_warm_up(timedelta(days=2))

        # --- Schedule EOD flatten at 15:55 ET ---
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.at(15, 55),
            self._flatten_eod,
        )

        # --- Session state ---
        self._current_day = None
        self._day_df = self._empty_df()
        self._prior_day_df = None
        self._bar_idx = 0
        self._session_initialized = False

        # --- Order tracking (manual OCO) ---
        self._entry_ticket = None
        self._stop_ticket = None
        self._target_ticket = None
        self._pending_signal = None  # Signal waiting for entry fill
        self._pending_entry = None   # EntryDecision waiting for fill

    # ------------------------------------------------------------------
    # Data event: called on every 1-min bar
    # ------------------------------------------------------------------

    def on_data(self, slice: Slice) -> None:
        if self.is_warming_up:
            return

        # --- Get active contract ---
        contract_symbol = self._get_contract_symbol()
        if contract_symbol is None:
            return

        # --- Extract bar data ---
        if contract_symbol not in slice.bars:
            return
        bar = slice.bars[contract_symbol]
        ts = bar.end_time
        o, h, l, c, v = (
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume),
        )

        # --- Skip Mondays (21.4% WR without filter) ---
        if ts.weekday() == 0:
            return

        # --- RTH only: 9:30 - 15:59 ET ---
        t = ts.time()
        if t < dt_time(9, 30) or t >= dt_time(16, 0):
            return

        # --- New day detection ---
        today = ts.date()
        if self._current_day != today:
            self._start_new_session(today, contract_symbol)

        if not self._session_initialized:
            return

        # --- Append bar to growing DataFrame ---
        new_row = pd.DataFrame(
            {"open": [o], "high": [h], "low": [l], "close": [c], "volume": [v]},
            index=[pd.Timestamp(ts)],
        )
        self._day_df = pd.concat([self._day_df, new_row])

        # --- Compute velocity (5-bar rolling) ---
        velocity = 0.0
        n = len(self._day_df)
        if n >= 6:
            velocity = (c - float(self._day_df["close"].iloc[-6])) / 5.0

        # --- Process bar through strategy pipeline ---
        result = self.strategy._process_bar(
            bar_idx=self._bar_idx,
            timestamp=ts,
            open_=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            velocity=velocity,
            df=self._day_df,
        )
        self._bar_idx += 1

        # --- Handle exit action ---
        if result.exit_action is not None:
            self._handle_exit_action(result.exit_action)

        # --- Handle entry decision ---
        if (
            result.entry_decision is not None
            and result.entry_decision.should_enter
            and self._entry_ticket is None
            and not self.portfolio[contract_symbol].invested
        ):
            self._place_entry(result, contract_symbol)

    # ------------------------------------------------------------------
    # Order events: manual OCO bracket management
    # ------------------------------------------------------------------

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.status != OrderStatus.FILLED:
            return

        order_id = order_event.order_id

        # --- Entry fill: place stop + target ---
        if self._entry_ticket is not None and order_id == self._entry_ticket.order_id:
            fill_price = float(order_event.fill_price)
            qty = int(order_event.fill_quantity)
            symbol = order_event.symbol
            self.log(
                f"ENTRY FILL: {qty} MES @ {fill_price:.2f}"
            )

            # Place bracket orders
            if self._pending_signal is not None:
                stop_price = self._pending_signal.stop_price
                target_price = self._pending_signal.target_1

                self._stop_ticket = self.stop_market_order(
                    symbol, -qty, stop_price, tag="ManciniStop"
                )
                self._target_ticket = self.limit_order(
                    symbol, -qty, target_price, tag="ManciniTarget"
                )
                self.log(
                    f"  Bracket: stop={stop_price:.2f} target={target_price:.2f}"
                )
            self._entry_ticket = None
            return

        # --- Stop fill: cancel target ---
        if self._stop_ticket is not None and order_id == self._stop_ticket.order_id:
            fill_price = float(order_event.fill_price)
            self.log(f"STOP HIT @ {fill_price:.2f}")
            if self._target_ticket is not None:
                self._target_ticket.cancel()
            self._clear_order_state()
            self._sync_position_close(fill_price, "Stop loss hit")
            return

        # --- Target fill: cancel stop ---
        if self._target_ticket is not None and order_id == self._target_ticket.order_id:
            fill_price = float(order_event.fill_price)
            self.log(f"TARGET HIT @ {fill_price:.2f}")
            if self._stop_ticket is not None:
                self._stop_ticket.cancel()
            self._clear_order_state()
            self._sync_position_close(fill_price, "Target 1 hit")
            return

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _start_new_session(self, today, contract_symbol) -> None:
        """Initialize strategy for a new trading day."""
        self._current_day = today
        self._bar_idx = 0
        self._day_df = self._empty_df()
        self._session_initialized = False

        # Fetch prior day 1-min bars for level initialization
        try:
            history = self.history(
                contract_symbol,
                timedelta(days=3),
                Resolution.MINUTE,
            )
            if history is not None and not history.empty:
                # Filter to prior RTH day
                if isinstance(history.index, pd.MultiIndex):
                    history = history.droplevel(0)

                history.index = pd.to_datetime(history.index)
                prior_mask = history.index.date < today
                prior_bars = history[prior_mask]

                if not prior_bars.empty:
                    # Get last trading day
                    last_day = prior_bars.index.date[-1]
                    last_day_mask = prior_bars.index.date == last_day
                    prior_day_bars = prior_bars[last_day_mask]

                    # Filter RTH hours
                    rth_mask = (
                        (prior_day_bars.index.time >= dt_time(9, 30))
                        & (prior_day_bars.index.time < dt_time(16, 0))
                    )
                    self._prior_day_df = prior_day_bars[rth_mask][
                        ["open", "high", "low", "close", "volume"]
                    ].copy()
                else:
                    self._prior_day_df = None
            else:
                self._prior_day_df = None
        except Exception as e:
            self.log(f"Warning: could not fetch prior day data: {e}")
            self._prior_day_df = None

        # Reset strategy and initialize session
        self.strategy.reset()
        self.strategy.position_manager.start_session(
            pd.Timestamp(today).to_pydatetime()
        )
        self.strategy.signal_aggregator.initialize_levels(
            self._day_df, self._prior_day_df
        )
        self._session_initialized = True
        self.log(
            f"Session started: {today} | "
            f"Prior day bars: {len(self._prior_day_df) if self._prior_day_df is not None else 0}"
        )

    def _flatten_eod(self) -> None:
        """End-of-day: cancel all orders and liquidate."""
        if self.portfolio.invested:
            self.log("EOD FLATTEN at 15:55 ET")
            self.transactions.cancel_open_orders()
            self.liquidate()
            self._clear_order_state()

            # Sync with strategy's position manager
            if self.strategy._current_position is not None:
                last_close = float(
                    self._day_df["close"].iloc[-1]
                ) if len(self._day_df) > 0 else 0.0
                self._sync_position_close(last_close, "EOD flatten")

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _place_entry(self, result: BarResult, symbol) -> None:
        """Place market entry order; bracket placed on fill."""
        entry = result.entry_decision
        signal = result.signal
        qty = entry.contracts

        self._pending_signal = signal
        self._pending_entry = entry
        self._entry_ticket = self.market_order(
            symbol, qty, tag="ManciniEntry"
        )
        self.log(
            f"ENTRY ORDER: {qty} MES @ ~{entry.entry_price:.2f} "
            f"stop={signal.stop_price:.2f} T1={signal.target_1:.2f}"
        )

    def _handle_exit_action(self, exit_action) -> None:
        """Process an exit action from the strategy's ExitManager."""
        # If ExitManager says to update stop (e.g., breakeven after T1)
        if (
            self._stop_ticket is not None
            and exit_action.new_stop != 0
            and exit_action.contracts_to_close == 0
        ):
            # Cancel old stop, place new one
            old_stop = self._stop_ticket
            symbol = old_stop.symbol
            qty = abs(old_stop.quantity)
            old_stop.cancel()
            self._stop_ticket = self.stop_market_order(
                symbol, -qty, exit_action.new_stop, tag="ManciniStopUpdate"
            )
            self.log(f"STOP UPDATED to {exit_action.new_stop:.2f}")

    def _sync_position_close(self, exit_price: float, reason: str) -> None:
        """Sync strategy's PositionManager after a fill."""
        if self.strategy._current_position is not None:
            ts = self.time
            self.strategy.position_manager.close_position(
                exit_price=exit_price,
                timestamp=ts,
                exit_reason=reason,
                pattern_type=self.strategy._current_pattern_type,
                signal=self.strategy._current_signal,
                entry_bar_idx=self.strategy._entry_bar_idx,
                exit_bar_idx=self._bar_idx,
            )
            self.strategy._current_position = None
            self.strategy._current_signal = None

    def _clear_order_state(self) -> None:
        """Reset order tracking."""
        self._entry_ticket = None
        self._stop_ticket = None
        self._target_ticket = None
        self._pending_signal = None
        self._pending_entry = None

    def _get_contract_symbol(self):
        """Get the active front-month MES contract symbol."""
        chain = self.futures_chains.get(self.mes.symbol)
        if chain is None:
            return None

        contracts = sorted(
            [c for c in chain],
            key=lambda c: c.expiry,
        )
        if not contracts:
            return None

        # Use front month
        self._contract_symbol = contracts[0].symbol
        return self._contract_symbol

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Create an empty OHLCV DataFrame with DatetimeIndex."""
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype=float,
        )

    def on_end_of_algorithm(self) -> None:
        """Log final session summary."""
        records = self.strategy.trade_records
        if records:
            total_pnl = sum(r.pnl_pts for r in records)
            wins = sum(1 for r in records if r.pnl_pts > 0)
            wr = wins / len(records) * 100 if records else 0
            self.log(
                f"SESSION SUMMARY: {len(records)} trades, "
                f"{wr:.0f}% WR, {total_pnl:+.1f} pts"
            )
