from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import coint

from .models import (
    AnalysisBundle,
    AssetSnapshot,
    PairInsight,
    PortfolioSolution,
    RunConfiguration,
)


TRADING_DAYS = 252


class QuantAnalyzer:
    """Builds quant features, risk metrics, pair analytics, and portfolio optimizations."""

    def __init__(
        self,
        config: RunConfiguration,
        histories: dict[str, pd.DataFrame],
        overviews: dict[str, dict[str, Any]],
    ) -> None:
        self.config = config
        self.histories = histories
        self.overviews = overviews

    def analyze(self) -> AnalysisBundle:
        feature_frames = {
            symbol: self._build_feature_frame(history)
            for symbol, history in self.histories.items()
        }
        prices = self._column_stack(feature_frames, "adjusted_close")
        returns = self._column_stack(feature_frames, "return")
        log_returns = self._column_stack(feature_frames, "log_return")
        normalized_prices = self._column_stack(feature_frames, "normalized_price")
        drawdowns = self._column_stack(feature_frames, "drawdown")
        rolling_volatility = self._column_stack(feature_frames, "rolling_vol_21")
        correlation_matrix = returns[self.config.symbols].dropna().corr()
        cointegration_matrix = self._cointegration_matrix(prices[self.config.symbols].dropna())
        rolling_beta = self._rolling_beta(returns)

        snapshots = self._build_snapshots(feature_frames, returns)
        pair_insight, pair_zscore = self._pair_insight(prices, returns, cointegration_matrix)
        frontier, max_sharpe_portfolio, min_vol_portfolio = self._portfolio_study(returns[self.config.symbols].dropna())

        return AnalysisBundle(
            config=self.config,
            prices=prices,
            returns=returns,
            log_returns=log_returns,
            normalized_prices=normalized_prices,
            drawdowns=drawdowns,
            rolling_volatility=rolling_volatility,
            rolling_beta=rolling_beta,
            correlation_matrix=correlation_matrix,
            cointegration_matrix=cointegration_matrix,
            feature_frames=feature_frames,
            pair_zscore_series=pair_zscore,
            snapshots=snapshots,
            max_sharpe_portfolio=max_sharpe_portfolio,
            min_vol_portfolio=min_vol_portfolio,
            efficient_frontier=frontier,
            pair_insight=pair_insight,
        )

    def _build_feature_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = history.copy()
        frame["return"] = frame["adjusted_close"].pct_change()
        frame["log_return"] = np.log(frame["adjusted_close"]).diff()
        frame["normalized_price"] = frame["adjusted_close"] / frame["adjusted_close"].iloc[0]
        frame["drawdown"] = frame["normalized_price"] / frame["normalized_price"].cummax() - 1.0
        frame["rolling_vol_21"] = frame["return"].rolling(21).std() * np.sqrt(TRADING_DAYS)
        frame["rolling_vol_63"] = frame["return"].rolling(63).std() * np.sqrt(TRADING_DAYS)
        frame["sma_21"] = frame["adjusted_close"].rolling(21).mean()
        frame["sma_63"] = frame["adjusted_close"].rolling(63).mean()
        frame["ema_21"] = frame["adjusted_close"].ewm(span=21, adjust=False).mean()
        frame["momentum_21"] = frame["adjusted_close"].pct_change(21)
        frame["momentum_63"] = frame["adjusted_close"].pct_change(63)
        frame["momentum_126"] = frame["adjusted_close"].pct_change(126)
        frame["rsi_14"] = self._relative_strength_index(frame["adjusted_close"], 14)
        macd_line, signal_line, histogram = self._macd(frame["adjusted_close"])
        frame["macd"] = macd_line
        frame["macd_signal"] = signal_line
        frame["macd_hist"] = histogram
        frame["atr_14"] = self._average_true_range(frame, 14)
        frame["range_pct"] = (frame["high"] - frame["low"]) / frame["adjusted_close"]
        volume_mean = frame["volume"].rolling(21).mean()
        volume_std = frame["volume"].rolling(21).std()
        frame["volume_zscore"] = (frame["volume"] - volume_mean) / volume_std
        return frame

    def _column_stack(self, feature_frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
        data = {
            symbol: frame[column]
            for symbol, frame in feature_frames.items()
            if column in frame.columns
        }
        return pd.DataFrame(data).sort_index()

    def _build_snapshots(
        self,
        feature_frames: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> list[AssetSnapshot]:
        snapshots: list[AssetSnapshot] = []
        benchmark_symbol = self.config.benchmark
        benchmark_returns = returns.get(benchmark_symbol)
        risk_free_daily = self.config.risk_free_rate / TRADING_DAYS

        for symbol in self.config.symbols:
            frame = feature_frames[symbol].dropna(subset=["return"]).copy()
            asset_returns = frame["return"].dropna()
            aligned_returns = pd.DataFrame({symbol: asset_returns})
            if benchmark_returns is not None:
                aligned_returns[benchmark_symbol] = benchmark_returns
                aligned_returns = aligned_returns.dropna()

            annual_return = self._cagr(frame["adjusted_close"])
            annual_volatility = asset_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
            downside_volatility = self._downside_volatility(asset_returns, risk_free_daily)
            max_drawdown = frame["drawdown"].min()
            value_at_risk = asset_returns.quantile(0.05)
            conditional_var = asset_returns[asset_returns <= value_at_risk].mean()
            hit_rate = float((asset_returns > 0).mean())
            tail_ratio = self._tail_ratio(asset_returns)

            beta = None
            alpha = None
            r_squared = None
            tracking_error = None
            information_ratio = None
            if benchmark_symbol in aligned_returns and len(aligned_returns) > 25:
                active_returns = aligned_returns[symbol] - aligned_returns[benchmark_symbol]
                tracking_error = active_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
                if tracking_error == 0.0 or pd.isna(tracking_error):
                    information_ratio = None
                else:
                    information_ratio = (active_returns.mean() * TRADING_DAYS) / tracking_error

                regression = sm.OLS(
                    aligned_returns[symbol] - risk_free_daily,
                    sm.add_constant(aligned_returns[benchmark_symbol] - risk_free_daily),
                ).fit()
                beta = float(regression.params.iloc[1])
                alpha = float(regression.params.iloc[0] * TRADING_DAYS)
                r_squared = float(regression.rsquared)

            sharpe_ratio = self._safe_divide(asset_returns.mean() * TRADING_DAYS - self.config.risk_free_rate, annual_volatility)
            sortino_ratio = self._safe_divide(asset_returns.mean() * TRADING_DAYS - self.config.risk_free_rate, downside_volatility)
            calmar_ratio = self._safe_divide(annual_return, abs(max_drawdown))

            overview = self.overviews.get(symbol, {})
            analyst_target_price = self._to_float(
                overview.get("analyst_target_mean")
                or overview.get("targetMeanPrice")
                or overview.get("targetMedianPrice")
            )

            snapshot = AssetSnapshot(
                symbol=symbol,
                company_name=str(overview.get("longName") or overview.get("shortName") or symbol),
                sector=str(overview.get("sector") or "Unknown"),
                industry=str(overview.get("industry") or "Unknown"),
                market_cap=self._to_float(overview.get("marketCap")),
                last_close=float(frame["adjusted_close"].iloc[-1]),
                total_return=float(frame["normalized_price"].iloc[-1] - 1.0),
                cagr=annual_return,
                annualized_volatility=annual_volatility,
                downside_volatility=downside_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=float(max_drawdown),
                beta=beta,
                alpha=alpha,
                r_squared=r_squared,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                value_at_risk_95=float(value_at_risk),
                conditional_var_95=float(conditional_var),
                skewness=float(skew(asset_returns, bias=False, nan_policy="omit")),
                kurtosis=float(kurtosis(asset_returns, fisher=True, bias=False, nan_policy="omit")),
                hit_rate=hit_rate,
                tail_ratio=tail_ratio,
                momentum_21d=self._last_value(frame["momentum_21"]),
                momentum_63d=self._last_value(frame["momentum_63"]),
                momentum_126d=self._last_value(frame["momentum_126"]),
                rsi_14=self._last_value(frame["rsi_14"]),
                macd_signal_gap=self._last_value(frame["macd_hist"]),
                valuation_pe=self._to_float(overview.get("trailingPE")),
                dividend_yield=self._to_float(overview.get("dividendYield")),
                analyst_target_price=analyst_target_price,
            )
            snapshot.narrative = self._build_asset_narrative(snapshot)
            snapshots.append(snapshot)

        return snapshots

    def _pair_insight(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        cointegration_matrix: pd.DataFrame,
    ) -> tuple[PairInsight | None, pd.Series | None]:
        symbols = self.config.symbols
        if len(symbols) < 2:
            return None, None

        best_pair: tuple[str, str] | None = None
        best_pvalue = np.inf
        for left, right in combinations(symbols, 2):
            pvalue = cointegration_matrix.loc[left, right]
            if pd.isna(pvalue):
                continue
            if pvalue < best_pvalue:
                best_pvalue = float(pvalue)
                best_pair = (left, right)

        if best_pair is None:
            best_pair = (symbols[0], symbols[1])
            best_pvalue = float("nan")

        pair_prices = prices[list(best_pair)].dropna()
        if pair_prices.empty:
            return None, None

        spread = np.log(pair_prices[best_pair[0]]) - np.log(pair_prices[best_pair[1]])
        zscore = (spread - spread.rolling(63).mean()) / spread.rolling(63).std()
        current_zscore = self._last_value(zscore)
        correlation = float(returns[best_pair[0]].corr(returns[best_pair[1]]))
        left_strength = pair_prices[best_pair[0]].iloc[-1] / pair_prices[best_pair[0]].iloc[0] - 1.0
        right_strength = pair_prices[best_pair[1]].iloc[-1] / pair_prices[best_pair[1]].iloc[0] - 1.0
        relative_strength_gap = float(left_strength - right_strength)

        insight = PairInsight(
            symbols=best_pair,
            correlation=correlation,
            cointegration_pvalue=best_pvalue if np.isfinite(best_pvalue) else None,
            spread_zscore=current_zscore,
            relative_strength_gap=relative_strength_gap,
        )
        insight.narrative = self._build_pair_narrative(insight)
        return insight, zscore

    def _portfolio_study(
        self,
        returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, PortfolioSolution | None, PortfolioSolution | None]:
        if returns.empty:
            return pd.DataFrame(), None, None

        if returns.shape[1] == 1:
            symbol = returns.columns[0]
            expected_return = float(returns[symbol].mean() * TRADING_DAYS)
            volatility = float(returns[symbol].std(ddof=0) * np.sqrt(TRADING_DAYS))
            sharpe = self._safe_divide(expected_return - self.config.risk_free_rate, volatility)
            solution = PortfolioSolution(
                name="Single Asset",
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                weights={symbol: 1.0},
            )
            frontier = pd.DataFrame(
                [{"target_return": expected_return, "volatility": volatility, "sharpe_ratio": sharpe}]
            )
            return frontier, solution, solution

        expected_returns = returns.mean() * TRADING_DAYS
        covariance = returns.cov() * TRADING_DAYS
        frontier = self._efficient_frontier(expected_returns, covariance)
        max_sharpe = self._solve_portfolio("Max Sharpe", expected_returns, covariance, "max_sharpe")
        min_vol = self._solve_portfolio("Min Vol", expected_returns, covariance, "min_volatility")
        return frontier, max_sharpe, min_vol

    def _efficient_frontier(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
        n_assets = len(expected_returns)
        weights0 = np.repeat(1.0 / n_assets, n_assets)
        bounds = [(0.0, 1.0)] * n_assets
        min_return = float(expected_returns.min())
        max_return = float(expected_returns.max())
        targets = np.linspace(min_return, max_return, 20)

        frontier_points: list[dict[str, float]] = []
        for target in targets:
            constraints = (
                {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
                {"type": "eq", "fun": lambda weights, target=target: self._portfolio_return(weights, expected_returns) - target},
            )
            result = minimize(
                lambda weights: self._portfolio_volatility(weights, covariance),
                weights0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if not result.success:
                continue

            volatility = self._portfolio_volatility(result.x, covariance)
            frontier_points.append(
                {
                    "target_return": target,
                    "volatility": volatility,
                    "sharpe_ratio": self._safe_divide(target - self.config.risk_free_rate, volatility),
                }
            )

        return pd.DataFrame(frontier_points)

    def _solve_portfolio(
        self,
        name: str,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        objective: str,
    ) -> PortfolioSolution:
        n_assets = len(expected_returns)
        weights0 = np.repeat(1.0 / n_assets, n_assets)
        bounds = [(0.0, 1.0)] * n_assets
        constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)

        if objective == "max_sharpe":
            objective_fn = lambda weights: -self._portfolio_sharpe(weights, expected_returns, covariance)
        else:
            objective_fn = lambda weights: self._portfolio_volatility(weights, covariance)

        result = minimize(
            objective_fn,
            weights0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        weights = result.x if result.success else weights0
        expected_return = self._portfolio_return(weights, expected_returns)
        volatility = self._portfolio_volatility(weights, covariance)
        sharpe_ratio = self._safe_divide(expected_return - self.config.risk_free_rate, volatility)
        return PortfolioSolution(
            name=name,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            weights=dict(zip(expected_returns.index.tolist(), weights, strict=True)),
        )

    def _cointegration_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.DataFrame(np.nan, index=prices.columns, columns=prices.columns, dtype=float)
        for left, right in combinations(prices.columns, 2):
            pair = prices[[left, right]].dropna()
            if len(pair) < 60:
                continue
            try:
                _, pvalue, _ = coint(np.log(pair[left]), np.log(pair[right]))
            except Exception:
                pvalue = np.nan
            matrix.loc[left, right] = pvalue
            matrix.loc[right, left] = pvalue

        for symbol in matrix.index:
            matrix.loc[symbol, symbol] = 0.0
        return matrix

    def _rolling_beta(self, returns: pd.DataFrame) -> pd.DataFrame:
        benchmark = self.config.benchmark
        if benchmark not in returns.columns:
            return pd.DataFrame(index=returns.index)

        benchmark_returns = returns[benchmark]
        rolling_beta = pd.DataFrame(index=returns.index)
        benchmark_variance = benchmark_returns.rolling(63).var()
        for symbol in self.config.symbols:
            if symbol == benchmark:
                continue
            covariance = returns[symbol].rolling(63).cov(benchmark_returns)
            rolling_beta[symbol] = covariance / benchmark_variance
        return rolling_beta

    def _relative_strength_index(self, series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        return 100 - (100 / (1 + relative_strength))

    def _macd(self, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast = series.ewm(span=12, adjust=False).mean()
        slow = series.ewm(span=26, adjust=False).mean()
        macd_line = fast - slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _average_true_range(self, frame: pd.DataFrame, window: int) -> pd.Series:
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift()).abs(),
                (frame["low"] - frame["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(window).mean()

    def _cagr(self, series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) < 2:
            return float("nan")
        total_return = clean.iloc[-1] / clean.iloc[0]
        years = len(clean) / TRADING_DAYS
        return float(total_return ** (1 / years) - 1.0)

    def _downside_volatility(self, returns: pd.Series, threshold: float) -> float:
        downside = np.minimum(returns - threshold, 0.0)
        return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS))

    def _tail_ratio(self, returns: pd.Series) -> float | None:
        lower = returns.quantile(0.05)
        upper = returns.quantile(0.95)
        if lower == 0 or pd.isna(lower) or pd.isna(upper):
            return None
        return float(abs(upper / lower))

    def _portfolio_return(self, weights: np.ndarray, expected_returns: pd.Series) -> float:
        return float(np.dot(weights, expected_returns.to_numpy()))

    def _portfolio_volatility(self, weights: np.ndarray, covariance: pd.DataFrame) -> float:
        return float(np.sqrt(weights @ covariance.to_numpy() @ weights))

    def _portfolio_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> float:
        volatility = self._portfolio_volatility(weights, covariance)
        expected_return = self._portfolio_return(weights, expected_returns)
        return self._safe_divide(expected_return - self.config.risk_free_rate, volatility)

    def _build_asset_narrative(self, snapshot: AssetSnapshot) -> str:
        tone = "efficiently paying you for risk" if snapshot.sharpe_ratio >= 1.0 else "asking for a lot of risk for a modest payoff"
        beta_tone = "aggressive versus the benchmark" if (snapshot.beta or 0.0) > 1.1 else "relatively defensive versus the benchmark"
        rsi_tone = "extended on the upside" if (snapshot.rsi_14 or 50.0) > 65 else "not overheated"
        return (
            f"{snapshot.symbol} is {tone}: CAGR is {snapshot.cagr:.1%} with annualized volatility at "
            f"{snapshot.annualized_volatility:.1%}. The current profile is {beta_tone}, max drawdown reached "
            f"{snapshot.max_drawdown:.1%}, and the momentum stack is {rsi_tone}."
        )

    def _build_pair_narrative(self, insight: PairInsight) -> str:
        if insight.cointegration_pvalue is not None and insight.cointegration_pvalue < 0.05:
            relationship = "statistically sticky enough to watch as a relative-value pair"
        else:
            relationship = "moving together directionally, but not strongly enough to call it a stable mean-reversion pair"

        stretch = "stretched" if abs(insight.spread_zscore or 0.0) >= 2.0 else "close to its recent equilibrium"
        return (
            f"{insight.symbols[0]} versus {insight.symbols[1]} looks {relationship}. Their daily return correlation is "
            f"{insight.correlation:.2f}, the spread is currently {stretch}, and the relative performance gap over the study "
            f"window is {insight.relative_strength_gap:.1%}."
        )

    def _last_value(self, series: pd.Series) -> float | None:
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])

    def _safe_divide(self, numerator: float, denominator: float | None) -> float:
        if denominator in (0.0, None) or pd.isna(denominator):
            return float("nan")
        return float(numerator / denominator)

    def _to_float(self, value: Any) -> float | None:
        try:
            if value in (None, "", "None"):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
