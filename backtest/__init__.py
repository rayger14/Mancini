from backtest.runner import BacktestRunner
from backtest.metrics import compute_metrics, StrategyMetrics
from backtest.visualizer import plot_trades

__all__ = [
    "BacktestRunner",
    "compute_metrics",
    "StrategyMetrics",
    "plot_trades",
]

try:
    from backtest.nautilus_strategy import ManciniNautilusStrategy, ManciniNautilusConfig
    from backtest.nautilus_runner import NautilusBacktestRunner, NautilusBacktestConfig

    __all__ += [
        "ManciniNautilusStrategy",
        "ManciniNautilusConfig",
        "NautilusBacktestRunner",
        "NautilusBacktestConfig",
    ]
except ImportError:
    pass  # nautilus_trader not installed
