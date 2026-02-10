"""Live trading session loop wiring data feed, strategy, and order execution."""

from __future__ import annotations

import signal as os_signal
import sys
from datetime import datetime, time
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    SessionTimes,
    ESContractSpec,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
    DEFAULT_SESSION,
    DEFAULT_CONTRACT,
)
from core.indicators import compute_velocity
from core.signals import Signal
from live.data_feed import DataFeed, DataFeedConfig, BarData
from live.order_executor import (
    OrderExecutor,
    PaperOrderExecutor,
    Order,
    OrderSide,
    OrderType,
    Fill,
)
from strategy.entry_manager import EntryManager
from strategy.exit_manager import ExitManager, ExitAction, TradePosition
from strategy.mancini_long import ManciniLongStrategy
from strategy.position_manager import PositionManager
from strategy.risk_manager import RiskManager


class LiveSessionManager:
    """Manages a live (or paper) trading session.

    Wires the data feed to the strategy and routes signals to the order executor.
    """

    def __init__(
        self,
        executor: Optional[OrderExecutor] = None,
        data_feed_config: DataFeedConfig = DataFeedConfig(),
        strategy_params: StrategyParams = DEFAULT_STRATEGY,
        elevator_params: ElevatorParams = DEFAULT_ELEVATOR,
        exit_params: ExitParams = DEFAULT_EXIT,
        risk_params: RiskParams = DEFAULT_RISK,
        session_times: SessionTimes = DEFAULT_SESSION,
        contract: ESContractSpec = DEFAULT_CONTRACT,
        min_rr_ratio: float = 1.5,
    ):
        self.executor = executor or PaperOrderExecutor(contract)
        self.data_feed = DataFeed(data_feed_config)
        self.contract = contract

        self.strategy = ManciniLongStrategy(
            strategy_params=strategy_params,
            elevator_params=elevator_params,
            exit_params=exit_params,
            risk_params=risk_params,
            session_times=session_times,
            contract=contract,
            min_rr_ratio=min_rr_ratio,
        )
        self.entry_manager = EntryManager(
            session=session_times,
            exit_params=exit_params,
            risk_params=risk_params,
        )
        self.exit_manager = ExitManager(params=exit_params, contract=contract)
        self.position_manager = PositionManager(risk_params=risk_params)
        self.risk_manager = RiskManager(
            risk_params=risk_params,
            session=session_times,
            contract=contract,
        )

        self._position: Optional[TradePosition] = None
        self._pattern_type: str = ""
        self._bar_count: int = 0
        self._running: bool = False

    def start(self) -> None:
        """Start the live trading session."""
        logger.info("=" * 60)
        logger.info("MANCINI LIVE SESSION STARTING")
        logger.info("=" * 60)

        # Initialize session
        self.position_manager.start_session(datetime.now())
        self.strategy.reset()
        self._running = True

        # Register signal handler for graceful shutdown
        os_signal.signal(os_signal.SIGINT, self._handle_shutdown)
        os_signal.signal(os_signal.SIGTERM, self._handle_shutdown)

        # Register bar callback and start feed
        self.data_feed.add_callback(self._on_bar)

        try:
            self.data_feed.start()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Stop the session gracefully."""
        self._running = False
        self.data_feed.stop()

        # Flatten any open position
        if self._position is not None and self._position.is_open:
            logger.warning("Flattening open position on shutdown")
            self._flatten_position("Session shutdown")

        self._log_session_summary()
        logger.info("Session stopped")

    def _on_bar(self, bar: BarData) -> None:
        """Callback invoked on each new 1-minute bar."""
        self._bar_count += 1
        current_time = bar.timestamp.time() if isinstance(bar.timestamp, datetime) else time(0, 0)

        # Update executor price (for stop/limit orders)
        if isinstance(self.executor, PaperOrderExecutor):
            triggered = self.executor.set_price(bar.close)
            for fill in triggered:
                self._handle_fill(fill, bar)

        # Build DataFrame from accumulated bars
        df = self.data_feed.get_history_df()
        if len(df) < 6:
            return  # need at least 6 bars for velocity

        velocity = compute_velocity(df, window=5)
        vel = float(velocity.iat[-1]) if not np.isnan(velocity.iat[-1]) else 0.0

        # Step 1: Check exits on existing position
        if self._position is not None and self._position.is_open:
            exit_action = self.exit_manager.update(
                self._position, bar.high, bar.low, bar.close
            )
            if exit_action is not None:
                self._execute_exit(exit_action, bar)
                return

        # Step 2: Check for new signals
        if self._position is None or not self._position.is_open:
            if self.position_manager.is_done_for_day:
                return

            signal = self.strategy.signal_aggregator.update(
                bar_idx=self._bar_count - 1,
                timestamp=bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now(),
                open_=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=float(bar.volume),
                velocity=vel,
                df=df,
            )

            if signal is not None:
                self._evaluate_and_enter(signal, current_time, bar)

    def _evaluate_and_enter(
        self, signal: Signal, current_time: time, bar: BarData
    ) -> None:
        """Evaluate signal and enter if approved."""
        # Risk check
        risk_check = self.risk_manager.validate_entry(
            signal, current_time, self.position_manager
        )
        if not risk_check.passed:
            logger.info(f"Signal rejected by risk: {risk_check.reason}")
            return

        # Entry decision
        entry = self.entry_manager.evaluate(
            signal=signal,
            current_time=current_time,
            trades_today=self.position_manager.trades_today,
            is_in_profit_protection=self.position_manager.is_profit_protection,
            daily_pnl_pts=self.position_manager.daily_pnl_pts,
        )

        if not entry.should_enter:
            logger.info(f"Entry declined: {entry.reason}")
            return

        # Create position
        self._position = self.exit_manager.create_position(
            entry_price=entry.entry_price,
            stop_price=entry.stop_price,
            target_1=signal.target_1,
            target_2=signal.target_2,
            contracts=entry.contracts,
        )
        self._pattern_type = signal.pattern.pattern_type

        ts = bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now()
        self.position_manager.open_position(self._position, ts, self._pattern_type)

        # Submit entry order
        order = Order(
            side=OrderSide.BUY,
            contracts=entry.contracts,
            order_type=OrderType.MARKET,
            tag=f"entry_{signal.signal_type.name}",
        )
        self.executor.submit_order(order)

        logger.info(
            f"LIVE ENTRY: {entry.contracts} contracts @ {entry.entry_price:.2f} "
            f"stop={entry.stop_price:.2f} T1={signal.target_1:.2f} T2={signal.target_2:.2f}"
        )

    def _execute_exit(self, action: ExitAction, bar: BarData) -> None:
        """Execute an exit action via the order executor."""
        order = Order(
            side=OrderSide.SELL,
            contracts=action.contracts_to_close,
            order_type=OrderType.MARKET,
            tag=action.reason,
        )
        self.executor.submit_order(order)

        logger.info(
            f"LIVE EXIT: {action.contracts_to_close} contracts - {action.reason}"
        )

        # If fully closed, record trade
        if self._position is not None and not self._position.is_open:
            ts = bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now()
            self.position_manager.close_position(
                exit_price=action.exit_price,
                timestamp=ts,
                exit_reason=action.reason,
                pattern_type=self._pattern_type,
            )
            self._position = None

    def _flatten_position(self, reason: str) -> None:
        """Emergency flatten: close all contracts at market."""
        if self._position is None:
            return

        remaining = self._position.remaining_contracts
        if remaining <= 0:
            return

        order = Order(
            side=OrderSide.SELL,
            contracts=remaining,
            order_type=OrderType.MARKET,
            tag=f"flatten: {reason}",
        )
        self.executor.submit_order(order)
        self._position.remaining_contracts = 0
        logger.warning(f"FLATTEN: {remaining} contracts - {reason}")

    def _handle_fill(self, fill: Fill, bar: BarData) -> None:
        """Handle a fill from a triggered pending order."""
        logger.info(
            f"Triggered order filled: {fill.side.name} {fill.contracts} "
            f"@ {fill.fill_price:.2f}"
        )

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def _log_session_summary(self) -> None:
        """Log end-of-session summary."""
        if self.position_manager.session is None:
            return

        s = self.position_manager.session
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Trades: {s.trade_count}")
        logger.info(f"  Wins:   {s.wins}")
        logger.info(f"  Losses: {s.losses}")
        logger.info(f"  PnL:    {s.daily_pnl_pts:+.1f} pts (${s.daily_pnl_dollars:+,.0f})")
        logger.info(f"  State:  {s.state.name}")
        logger.info("=" * 60)
