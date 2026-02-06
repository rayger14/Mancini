"""NautilusTrader backtest runner for the Mancini strategy.

Sets up BacktestEngine with realistic ES futures execution:
venue, instrument, fill model (slippage), commissions, and data wrangling.
Produces BacktestResult compatible with compute_metrics().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    AssetClass,
    BarAggregation,
    PriceType,
    AggregationSource,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity, Currency
from nautilus_trader.persistence.wranglers import BarDataWrangler

from backtest.nautilus_strategy import ManciniNautilusStrategy, ManciniNautilusConfig
from backtest.runner import DayResult, BacktestResult
from config.settings import (
    StrategyParams,
    ElevatorParams,
    ExitParams,
    RiskParams,
    DEFAULT_STRATEGY,
    DEFAULT_ELEVATOR,
    DEFAULT_EXIT,
    DEFAULT_RISK,
    DEFAULT_CONTRACT,
)
from strategy.position_manager import TradeRecord


@dataclass
class NautilusBacktestConfig:
    """Configuration for the NautilusTrader backtest runner."""

    commission_per_side: float = 1.25       # USD per contract per side
    prob_slippage: float = 0.5              # 50% chance of 1-tick adverse fill
    starting_balance: float = 100_000.0     # USD

    strategy_params: StrategyParams = field(default_factory=lambda: DEFAULT_STRATEGY)
    elevator_params: ElevatorParams = field(default_factory=lambda: DEFAULT_ELEVATOR)
    exit_params: ExitParams = field(default_factory=lambda: DEFAULT_EXIT)
    risk_params: RiskParams = field(default_factory=lambda: DEFAULT_RISK)

    min_rr_ratio: float = 1.5


def _serialize_params(obj) -> dict:
    """Convert a frozen dataclass to a plain dict, handling nested types."""
    d = asdict(obj)
    return d


class NautilusBacktestRunner:
    """Runs the Mancini strategy through NautilusTrader's backtest engine."""

    def __init__(self, config: NautilusBacktestConfig | None = None):
        self.config = config or NautilusBacktestConfig()

    def _create_es_instrument(self) -> FuturesContract:
        """Create an ES futures contract instrument."""
        from nautilus_trader.test_kit.providers import TestInstrumentProvider

        instrument_id = InstrumentId(Symbol("ES"), Venue("GLBX"))

        return FuturesContract(
            instrument_id=instrument_id,
            raw_symbol=Symbol("ES"),
            asset_class=AssetClass.INDEX,
            currency=USD,
            price_precision=2,
            price_increment=Price.from_str("0.25"),
            multiplier=Quantity.from_int(50),
            lot_size=Quantity.from_int(1),
            underlying="ES",
            activation_ns=0,
            expiration_ns=0,
            ts_event=0,
            ts_init=0,
        )

    def _create_engine(self, instrument: FuturesContract) -> BacktestEngine:
        """Create and configure BacktestEngine with venue and fill model."""
        engine_config = BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
        )
        engine = BacktestEngine(config=engine_config)

        # Add venue with fill model
        fill_model = FillModel(
            prob_fill_on_limit=1.0,
            prob_fill_on_stop=1.0,
            prob_slippage=self.config.prob_slippage,
        )

        engine.add_venue(
            venue=Venue("GLBX"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self.config.starting_balance, USD)],
            fill_model=fill_model,
        )

        engine.add_instrument(instrument)

        return engine

    def _wrangle_bars(
        self, df: pd.DataFrame, instrument: FuturesContract
    ) -> list:
        """Convert OHLCV DataFrame to NautilusTrader Bar objects."""
        bar_type = BarType(
            instrument_id=instrument.id,
            bar_spec=BarType.from_str(
                f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"
            ).bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )
        wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)

        # Ensure DataFrame has the expected columns and DatetimeIndex
        wrangle_df = df[["open", "high", "low", "close", "volume"]].copy()
        if not isinstance(wrangle_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        return wrangler.process(wrangle_df)

    def _build_strategy_config(
        self, prior_day_data: dict | None = None
    ) -> ManciniNautilusConfig:
        """Build ManciniNautilusConfig from runner config."""
        return ManciniNautilusConfig(
            instrument_id="ES.GLBX",
            bar_type="ES.GLBX-1-MINUTE-LAST-EXTERNAL",
            strategy_params=_serialize_params(self.config.strategy_params),
            elevator_params=_serialize_params(self.config.elevator_params),
            exit_params=_serialize_params(self.config.exit_params),
            risk_params=_serialize_params(self.config.risk_params),
            min_rr_ratio=self.config.min_rr_ratio,
            prior_day_data=prior_day_data,
        )

    def run_single_day(
        self,
        df: pd.DataFrame,
        prior_day_df: Optional[pd.DataFrame] = None,
        day: Optional[date] = None,
    ) -> DayResult:
        """Run the strategy on a single day of data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV bars for the day with DatetimeIndex.
        prior_day_df : pd.DataFrame, optional
            Previous day's data for level initialization.
        day : date, optional
            Session date.

        Returns
        -------
        DayResult
        """
        if day is None:
            day = df.index[0].date()

        instrument = self._create_es_instrument()
        engine = self._create_engine(instrument)

        # Wrangle bar data
        bars = self._wrangle_bars(df, instrument)
        engine.add_data(bars)

        # Prepare prior day data as serialisable dict
        prior_data = None
        if prior_day_df is not None:
            prior_bars = []
            for i in range(len(prior_day_df)):
                prior_bars.append({
                    "open": float(prior_day_df["open"].iat[i]),
                    "high": float(prior_day_df["high"].iat[i]),
                    "low": float(prior_day_df["low"].iat[i]),
                    "close": float(prior_day_df["close"].iat[i]),
                    "volume": float(prior_day_df["volume"].iat[i]),
                    "timestamp": str(prior_day_df.index[i]),
                })
            prior_data = {"bars": prior_bars}

        # Create and register strategy
        strat_config = self._build_strategy_config(prior_day_data=prior_data)
        strategy = ManciniNautilusStrategy(config=strat_config)
        engine.add_strategy(strategy)

        # Run
        engine.run()

        # Extract results
        trade_records = strategy.completed_trades
        pnl_pts = sum(t.pnl_pts for t in trade_records)
        pnl_dollars = sum(t.pnl_dollars for t in trade_records)
        wins = sum(1 for t in trade_records if t.pnl_pts > 0)
        wr = wins / len(trade_records) if trade_records else 0.0

        engine.dispose()

        return DayResult(
            date=day,
            bar_results=[],  # NautilusTrader doesn't produce BarResults
            trade_records=trade_records,
            pnl_pts=pnl_pts,
            pnl_dollars=pnl_dollars,
            num_trades=len(trade_records),
            win_rate=wr,
        )

    def run_multi_day(
        self,
        daily_dfs: Optional[dict[date, pd.DataFrame]] = None,
        symbol: str = "ES",
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> BacktestResult:
        """Run the strategy across multiple days.

        Parameters
        ----------
        daily_dfs : dict, optional
            Pre-loaded DataFrames keyed by date.
        symbol : str
            Futures symbol (for logging).
        start, end : date, optional
            Date range.

        Returns
        -------
        BacktestResult
        """
        result = BacktestResult()

        if daily_dfs is not None:
            dates = sorted(daily_dfs.keys())
        elif start is not None and end is not None:
            dates = []
            d = start
            while d <= end:
                if d.weekday() < 5:
                    dates.append(d)
                d += timedelta(days=1)
        else:
            raise ValueError("Provide either daily_dfs or start+end dates")

        prior_day_df: Optional[pd.DataFrame] = None

        for day in dates:
            try:
                if daily_dfs is not None:
                    df = daily_dfs[day]
                else:
                    raise ValueError("Data fetching not supported in NautilusTrader runner; provide daily_dfs")

                if len(df) < 10:
                    logger.warning(f"Skipping {day}: only {len(df)} bars")
                    continue

                day_result = self.run_single_day(df, prior_day_df, day)
                result.days.append(day_result)
                result.all_trades.extend(day_result.trade_records)

                logger.info(
                    f"{day}: {day_result.num_trades} trades, "
                    f"PnL={day_result.pnl_pts:+.1f} pts, "
                    f"WR={day_result.win_rate:.0%}"
                )

                prior_day_df = df

            except Exception as e:
                logger.error(f"Error on {day}: {e}")
                continue

        logger.info(
            f"\nNautilus backtest complete: {len(result.days)} days, "
            f"{result.total_trades} trades, "
            f"PnL={result.total_pnl_pts:+.1f} pts "
            f"(${result.total_pnl_dollars:+,.0f}), "
            f"WR={result.win_rate:.0%}"
        )

        return result
