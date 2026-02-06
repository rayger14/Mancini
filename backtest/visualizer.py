"""Trade charts with entries/exits/levels overlay using mplfinance + Plotly."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtest.runner import DayResult
from config.levels import LevelStore
from strategy.position_manager import TradeRecord


def plot_trades(
    day_result: DayResult,
    df: pd.DataFrame,
    level_store: Optional[LevelStore] = None,
    backend: str = "mplfinance",
    save_path: Optional[str] = None,
) -> None:
    """Plot candlestick chart with trade entries, exits, and levels.

    Parameters
    ----------
    day_result : DayResult
        Results for the day.
    df : pd.DataFrame
        OHLCV data.
    level_store : LevelStore, optional
        Price levels to overlay.
    backend : str
        "mplfinance" or "plotly".
    save_path : str, optional
        If provided, save the chart to this path.
    """
    if backend == "plotly":
        return _plot_plotly(day_result, df, level_store, save_path)
    else:
        return _plot_mplfinance(day_result, df, level_store, save_path)


def _plot_mplfinance(
    day_result: DayResult,
    df: pd.DataFrame,
    level_store: Optional[LevelStore],
    save_path: Optional[str],
) -> None:
    """Matplotlib/mplfinance candlestick chart."""
    try:
        import mplfinance as mpf
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("mplfinance required: pip install mplfinance")

    # Prepare additional plots
    addplots = []

    # Entry markers
    entries = _get_entry_markers(day_result, df)
    if entries is not None:
        addplots.append(
            mpf.make_addplot(
                entries, type="scatter", markersize=100, marker="^", color="green"
            )
        )

    # Exit markers
    exits = _get_exit_markers(day_result, df)
    if exits is not None:
        addplots.append(
            mpf.make_addplot(
                exits, type="scatter", markersize=100, marker="v", color="red"
            )
        )

    # Level lines
    hlines = {}
    if level_store is not None:
        levels = level_store.get_active()
        if levels:
            hlines = {
                "hlines": [l.price for l in levels[:10]],  # max 10 lines
                "colors": ["blue"] * min(len(levels), 10),
                "linestyle": "--",
                "linewidths": 0.8,
            }

    kwargs = {
        "type": "candle",
        "style": "charles",
        "title": f"Mancini Strategy - {day_result.date} | "
        f"PnL: {day_result.pnl_pts:+.1f} pts | "
        f"Trades: {day_result.num_trades}",
        "volume": True,
        "figsize": (16, 9),
    }

    if addplots:
        kwargs["addplot"] = addplots
    if hlines:
        kwargs.update(hlines)
    if save_path:
        kwargs["savefig"] = save_path

    mpf.plot(df, **kwargs)


def _plot_plotly(
    day_result: DayResult,
    df: pd.DataFrame,
    level_store: Optional[LevelStore],
    save_path: Optional[str],
) -> None:
    """Interactive Plotly candlestick chart."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="ES",
        ),
        row=1,
        col=1,
    )

    # Volume
    colors = [
        "green" if c >= o else "red"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume"),
        row=2,
        col=1,
    )

    # Level lines
    if level_store is not None:
        for level in level_store.get_active()[:10]:
            fig.add_hline(
                y=level.price,
                line_dash="dash",
                line_color="blue",
                annotation_text=level.label,
                row=1,
                col=1,
            )

    # Entry/exit markers
    for record in day_result.trade_records:
        # Entry
        fig.add_trace(
            go.Scatter(
                x=[record.entry_time],
                y=[record.entry_price],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="green"),
                name="Entry",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Exit
        fig.add_trace(
            go.Scatter(
                x=[record.exit_time],
                y=[record.avg_exit_price],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red"),
                name="Exit",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"Mancini Strategy - {day_result.date} | "
        f"PnL: {day_result.pnl_pts:+.1f} pts | "
        f"Trades: {day_result.num_trades}",
        xaxis_rangeslider_visible=False,
        height=800,
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_entry_markers(
    day_result: DayResult, df: pd.DataFrame
) -> Optional[pd.Series]:
    """Create a Series with entry prices at entry bars (NaN elsewhere)."""
    markers = pd.Series(np.nan, index=df.index)
    for r in day_result.bar_results:
        if r.entry_decision is not None and r.entry_decision.should_enter:
            markers.iloc[r.bar_idx] = r.entry_decision.entry_price
    if markers.notna().any():
        return markers
    return None


def _get_exit_markers(
    day_result: DayResult, df: pd.DataFrame
) -> Optional[pd.Series]:
    """Create a Series with exit prices at exit bars (NaN elsewhere)."""
    markers = pd.Series(np.nan, index=df.index)
    for r in day_result.bar_results:
        if r.exit_action is not None:
            markers.iloc[r.bar_idx] = r.exit_action.exit_price
    if markers.notna().any():
        return markers
    return None


def plot_equity_curve(
    trades: list[TradeRecord],
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative P&L equity curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required")

    if not trades:
        return

    pnl = [t.pnl_pts for t in trades]
    equity = np.cumsum(pnl)
    dates = [t.exit_time for t in trades]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, equity, "b-", linewidth=1.5)
    ax.fill_between(dates, equity, alpha=0.1, color="blue")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Mancini Strategy Equity Curve")
    ax.set_ylabel("Cumulative P&L (pts)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
