from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class RunConfiguration:
    symbols: list[str]
    benchmark: str
    lookback_years: int
    risk_free_rate: float
    output_path: Path


@dataclass(slots=True)
class AssetSnapshot:
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float | None
    last_close: float
    total_return: float
    cagr: float
    annualized_volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    beta: float | None
    alpha: float | None
    r_squared: float | None
    tracking_error: float | None
    information_ratio: float | None
    value_at_risk_95: float
    conditional_var_95: float
    skewness: float
    kurtosis: float
    hit_rate: float
    tail_ratio: float | None
    momentum_21d: float | None
    momentum_63d: float | None
    momentum_126d: float | None
    rsi_14: float | None
    macd_signal_gap: float | None
    valuation_pe: float | None
    dividend_yield: float | None
    analyst_target_price: float | None
    narrative: str = ""


@dataclass(slots=True)
class PairInsight:
    symbols: tuple[str, str]
    correlation: float
    cointegration_pvalue: float | None
    spread_zscore: float | None
    relative_strength_gap: float | None
    narrative: str = ""


@dataclass(slots=True)
class PortfolioSolution:
    name: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


@dataclass(slots=True)
class AnalysisBundle:
    config: RunConfiguration
    prices: pd.DataFrame
    returns: pd.DataFrame
    log_returns: pd.DataFrame
    normalized_prices: pd.DataFrame
    drawdowns: pd.DataFrame
    rolling_volatility: pd.DataFrame
    rolling_beta: pd.DataFrame
    correlation_matrix: pd.DataFrame
    cointegration_matrix: pd.DataFrame
    feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    pair_zscore_series: pd.Series | None = None
    snapshots: list[AssetSnapshot] = field(default_factory=list)
    max_sharpe_portfolio: PortfolioSolution | None = None
    min_vol_portfolio: PortfolioSolution | None = None
    efficient_frontier: pd.DataFrame = field(default_factory=pd.DataFrame)
    pair_insight: PairInsight | None = None
