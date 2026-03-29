from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import AnalysisBundle, AssetSnapshot, PortfolioSolution


class ConsoleReporter:
    """Prints a trader-friendly summary with narrative explanations."""

    def __init__(self) -> None:
        self.console = Console()

    def render(self, bundle: AnalysisBundle) -> None:
        self.console.rule("[bold #16324F]StockVisor Quant Deck")
        self.console.print(
            Panel.fit(
                self._market_narrative(bundle),
                title="Market Read",
                border_style="#16324F",
            )
        )
        self.console.print(self._asset_table(bundle.snapshots))

        for snapshot in bundle.snapshots:
            self.console.print(
                Panel(
                    snapshot.narrative,
                    title=f"{snapshot.symbol} Interpretation",
                    border_style="#C46B2E",
                )
            )

        if bundle.pair_insight is not None:
            self.console.print(
                Panel(
                    bundle.pair_insight.narrative,
                    title="Relative-Value Read",
                    border_style="#2A6F97",
                )
            )

        portfolio_table = self._portfolio_table(bundle.max_sharpe_portfolio, bundle.min_vol_portfolio)
        if portfolio_table is not None:
            self.console.print(portfolio_table)

        self.console.print(
            Panel.fit(
                self._graph_guide_text(bundle),
                title="How To Read The Graphs",
                border_style="#5C8001",
            )
        )
        self.console.print(
            Panel(
                self._stat_guide_text(),
                title="What The Quant Stats Mean",
                border_style="#2A6F97",
            )
        )

    def _market_narrative(self, bundle: AnalysisBundle) -> str:
        snapshots = bundle.snapshots
        if not snapshots:
            return "No assets were analyzed."

        strongest = max(snapshots, key=lambda snapshot: snapshot.sharpe_ratio)
        weakest = min(snapshots, key=lambda snapshot: snapshot.max_drawdown)
        corr_matrix = bundle.correlation_matrix.copy()
        if len(corr_matrix) > 1:
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            avg_corr = corr_matrix.where(mask).stack().mean()
            diversification_line = (
                f"Average cross-correlation is {avg_corr:.2f}, which tells you how much true diversification "
                f"you are really getting when the tape becomes volatile."
            )
        else:
            diversification_line = (
                "This is a single-stock study, so diversification math is intentionally limited and the emphasis is "
                "on trend, volatility, drawdown, and benchmark behavior."
            )

        return (
            f"The basket is centered on {strongest.symbol} as the best risk-adjusted leader, while {weakest.symbol} carries "
            f"the deepest peak-to-trough pain. {diversification_line}"
        )

    def _asset_table(self, snapshots: list[AssetSnapshot]) -> Table:
        table = Table(title="Risk / Return / Signal Grid", show_lines=False)
        table.add_column("Symbol", style="bold")
        table.add_column("Company")
        table.add_column("CAGR", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Sortino", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Beta", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("VaR 95", justify="right")

        for snapshot in snapshots:
            table.add_row(
                snapshot.symbol,
                snapshot.company_name,
                self._fmt_pct(snapshot.cagr),
                self._fmt_pct(snapshot.annualized_volatility),
                self._fmt_num(snapshot.sharpe_ratio),
                self._fmt_num(snapshot.sortino_ratio),
                self._fmt_pct(snapshot.max_drawdown),
                self._fmt_opt(snapshot.beta),
                self._fmt_opt(snapshot.rsi_14),
                self._fmt_pct(snapshot.value_at_risk_95),
            )

        return table

    def _portfolio_table(
        self,
        max_sharpe: PortfolioSolution | None,
        min_vol: PortfolioSolution | None,
    ) -> Table | None:
        if max_sharpe is None or min_vol is None:
            return None

        symbols = list(max_sharpe.weights)
        table = Table(title="Optimized Portfolio Views")
        table.add_column("Portfolio", style="bold")
        table.add_column("Exp. Return", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Sharpe", justify="right")
        for symbol in symbols:
            table.add_column(f"{symbol} Wt", justify="right")

        solutions = [max_sharpe]
        if not self._same_solution(max_sharpe, min_vol):
            solutions.append(min_vol)

        for solution in solutions:
            table.add_row(
                solution.name,
                self._fmt_pct(solution.expected_return),
                self._fmt_pct(solution.volatility),
                self._fmt_num(solution.sharpe_ratio),
                *[self._fmt_pct(solution.weights[symbol]) for symbol in symbols],
            )

        return table

    def _fmt_pct(self, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.1%}"

    def _fmt_num(self, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.2f}"

    def _fmt_opt(self, value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:.2f}"

    def _same_solution(
        self,
        left: PortfolioSolution | None,
        right: PortfolioSolution | None,
    ) -> bool:
        if left is None or right is None:
            return False
        if left.name != right.name:
            return False
        return (
            np.isclose(left.expected_return, right.expected_return)
            and np.isclose(left.volatility, right.volatility)
            and np.isclose(left.sharpe_ratio, right.sharpe_ratio)
            and left.weights == right.weights
        )

    def _graph_guide_text(self, bundle: AnalysisBundle) -> str:
        lines = [
            "- Normalized Price Paths: starts every stock at the same base so you can compare compounding fairly.",
            "- Daily Return Correlation: shows which names tend to move together day to day.",
            "- Rolling 21-Day Volatility: shows when each stock became calmer or more explosive.",
            "- Drawdown Curve: shows how deep each selloff got from its prior peak.",
            "- Efficient Frontier: shows the best historical return available for each level of volatility.",
        ]
        if bundle.pair_insight is not None:
            lines.append(
                "- Spread Z-Score: shows whether the strongest pair is stretched or close to its recent relationship."
            )
        else:
            lines.append(
                "- Rolling Beta vs Benchmark: shows how aggressively each stock has been moving versus the benchmark over time."
            )
        return "\n".join(lines)

    def _stat_guide_text(self) -> str:
        return "\n".join(
            [
                "- Symbol: the ticker being analyzed.",
                "- CAGR: compounded annual growth rate, which tells you how fast the stock grew per year over the full window.",
                "- Vol: annualized volatility, which tells you how rough or explosive the ride was.",
                "- Sharpe: return earned per unit of total risk after comparing against the risk-free rate. Higher is better.",
                "- Sortino: like Sharpe, but it only penalizes downside volatility. Higher is better.",
                "- Max DD: maximum drawdown, the worst peak-to-trough loss in the period. Closer to zero is better.",
                "- Beta: sensitivity to the benchmark. Around 1 means market-like, above 1 is more aggressive, below 1 is calmer.",
                "- RSI: short-term momentum gauge from 0 to 100. Above 70 can mean extended, below 30 can mean washed out.",
                "- Risk-free rate: the low-risk baseline return used in Sharpe and Sortino, often thought of like cash or short-term Treasury yield.",
            ]
        )


class PlotlyReporter:
    """Builds an interactive HTML dashboard."""

    PALETTE = ["#16324F", "#C46B2E", "#2A6F97", "#5C8001", "#A23E48", "#7A6C5D"]

    def write_dashboard(self, bundle: AnalysisBundle) -> Path:
        output_path = bundle.config.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = make_subplots(
            rows=5,
            cols=2,
            specs=[
                [{"type": "xy"}, {"type": "heatmap"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "table"}, {"type": "table"}],
                [{"type": "table", "colspan": 2}, None],
            ],
            subplot_titles=[
                "Normalized Price Paths",
                "Daily Return Correlation",
                "Rolling 21-Day Volatility",
                "Drawdown Curve",
                "Efficient Frontier",
                self._dynamic_subplot_title(bundle),
                "Quant Summary Table",
                "How To Read These Charts",
                "What The Quant Stats Mean",
            ],
            row_heights=[0.2, 0.18, 0.18, 0.18, 0.26],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )

        self._add_normalized_prices(fig, bundle)
        self._add_correlation_heatmap(fig, bundle)
        self._add_rolling_volatility(fig, bundle)
        self._add_drawdowns(fig, bundle)
        self._add_frontier(fig, bundle)
        self._add_pair_or_beta_panel(fig, bundle)
        self._add_summary_table(fig, bundle)
        self._add_graph_guide_table(fig, bundle)
        self._add_stat_guide_table(fig)

        fig.update_layout(
            template="plotly_white",
            height=2280,
            title={
                "text": (
                    "StockVisor Quant Dashboard"
                    f"<br><sup>{', '.join(bundle.config.symbols)} vs {bundle.config.benchmark}</sup>"
                ),
                "x": 0.03,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 26},
            },
            font={"family": "IBM Plex Sans, Arial, sans-serif", "color": "#132A3E"},
            paper_bgcolor="#F7F3EB",
            plot_bgcolor="#FFFDFC",
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.06,
                "x": 0.0,
                "font": {"size": 12},
            },
            margin={"l": 50, "r": 30, "t": 140, "b": 110},
        )
        fig.update_annotations(font={"size": 16})
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        return output_path

    def _add_normalized_prices(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        for index, symbol in enumerate(bundle.config.symbols):
            fig.add_trace(
                go.Scatter(
                    x=bundle.normalized_prices.index,
                    y=bundle.normalized_prices[symbol],
                    mode="lines",
                    name=symbol,
                    line={"width": 2.4, "color": self.PALETTE[index % len(self.PALETTE)]},
                    hovertemplate="%{x|%Y-%m-%d}<br>Normalized %{y:.2f}<extra>" + symbol + "</extra>",
                ),
                row=1,
                col=1,
            )

    def _add_correlation_heatmap(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        fig.add_trace(
            go.Heatmap(
                z=bundle.correlation_matrix.values,
                x=bundle.correlation_matrix.columns,
                y=bundle.correlation_matrix.index,
                colorscale=[[0.0, "#A23E48"], [0.5, "#FFF6E5"], [1.0, "#16324F"]],
                zmin=-1,
                zmax=1,
                text=bundle.correlation_matrix.round(2).astype(str).values,
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                showscale=True,
            ),
            row=1,
            col=2,
        )

    def _add_rolling_volatility(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        for index, symbol in enumerate(bundle.config.symbols):
            fig.add_trace(
                go.Scatter(
                    x=bundle.rolling_volatility.index,
                    y=bundle.rolling_volatility[symbol],
                    mode="lines",
                    name=f"{symbol} 21d vol",
                    line={"width": 1.8, "dash": "solid", "color": self.PALETTE[index % len(self.PALETTE)]},
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>Vol %{y:.1%}<extra>" + symbol + "</extra>",
                ),
                row=2,
                col=1,
            )

    def _add_drawdowns(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        for index, symbol in enumerate(bundle.config.symbols):
            fig.add_trace(
                go.Scatter(
                    x=bundle.drawdowns.index,
                    y=bundle.drawdowns[symbol],
                    mode="lines",
                    name=f"{symbol} drawdown",
                    line={"width": 2.0, "color": self.PALETTE[index % len(self.PALETTE)]},
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>Drawdown %{y:.1%}<extra>" + symbol + "</extra>",
                ),
                row=2,
                col=2,
            )

    def _add_frontier(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        frontier = bundle.efficient_frontier
        if not frontier.empty:
            fig.add_trace(
                go.Scatter(
                    x=frontier["volatility"],
                    y=frontier["target_return"],
                    mode="lines",
                    name="Efficient Frontier",
                    line={"color": "#8C5E34", "width": 3},
                    showlegend=False,
                    hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra>Frontier</extra>",
                ),
                row=3,
                col=1,
            )

        for index, snapshot in enumerate(bundle.snapshots):
            fig.add_trace(
                go.Scatter(
                    x=[snapshot.annualized_volatility],
                    y=[snapshot.cagr],
                    mode="markers+text",
                    text=[snapshot.symbol],
                    textposition="top center",
                    marker={
                        "size": 12,
                        "color": self.PALETTE[index % len(self.PALETTE)],
                        "line": {"color": "#FFFFFF", "width": 1},
                    },
                    name=f"{snapshot.symbol} profile",
                    showlegend=False,
                    hovertemplate="Vol %{x:.1%}<br>CAGR %{y:.1%}<extra>" + snapshot.symbol + "</extra>",
                ),
                row=3,
                col=1,
            )

        for solution, marker_symbol, marker_color in (
            (bundle.max_sharpe_portfolio, "star", "#2A6F97"),
            (bundle.min_vol_portfolio, "diamond", "#C46B2E"),
        ):
            if solution is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[solution.volatility],
                    y=[solution.expected_return],
                    mode="markers",
                    marker={"size": 16, "symbol": marker_symbol, "color": marker_color},
                    name=solution.name,
                    showlegend=False,
                    hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra>" + solution.name + "</extra>",
                ),
                row=3,
                col=1,
            )

    def _add_pair_or_beta_panel(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        if bundle.pair_zscore_series is not None and bundle.pair_insight is not None:
            fig.add_trace(
                go.Scatter(
                    x=bundle.pair_zscore_series.index,
                    y=bundle.pair_zscore_series,
                    mode="lines",
                    name="Spread Z-Score",
                    line={"color": "#16324F", "width": 2.2},
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>Z-score %{y:.2f}<extra></extra>",
                ),
                row=3,
                col=2,
            )
            for level, color in ((0.0, "#7A6C5D"), (2.0, "#A23E48"), (-2.0, "#A23E48")):
                fig.add_hline(y=level, line_dash="dot", line_color=color, row=3, col=2)
        else:
            for index, column in enumerate(bundle.rolling_beta.columns):
                fig.add_trace(
                    go.Scatter(
                        x=bundle.rolling_beta.index,
                        y=bundle.rolling_beta[column],
                        mode="lines",
                        line={"color": self.PALETTE[index % len(self.PALETTE)], "width": 2.0},
                        name=f"{column} beta",
                        showlegend=False,
                        hovertemplate="%{x|%Y-%m-%d}<br>Beta %{y:.2f}<extra>" + column + "</extra>",
                    ),
                    row=3,
                    col=2,
                )

    def _add_summary_table(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        summary = pd.DataFrame(
            [
                {
                    "Symbol": snapshot.symbol,
                    "CAGR": f"{snapshot.cagr:.1%}",
                    "Vol": f"{snapshot.annualized_volatility:.1%}",
                    "Sharpe": f"{snapshot.sharpe_ratio:.2f}",
                    "Sortino": f"{snapshot.sortino_ratio:.2f}",
                    "Max DD": f"{snapshot.max_drawdown:.1%}",
                    "Beta": "n/a" if snapshot.beta is None else f"{snapshot.beta:.2f}",
                    "RSI": "n/a" if snapshot.rsi_14 is None else f"{snapshot.rsi_14:.1f}",
                }
                for snapshot in bundle.snapshots
            ]
        )

        fig.add_trace(
            go.Table(
                header={
                    "values": list(summary.columns),
                    "fill_color": "#16324F",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [summary[column] for column in summary.columns],
                    "fill_color": "#FFFDFC",
                    "align": "left",
                    "font": {"color": "#132A3E", "size": 12},
                },
            ),
            row=4,
            col=1,
        )

    def _add_graph_guide_table(self, fig: go.Figure, bundle: AnalysisBundle) -> None:
        chart_names = [
            "Normalized Price Paths",
            "Daily Return Correlation",
            "Rolling 21-Day Volatility",
            "Drawdown Curve",
            "Efficient Frontier",
        ]
        chart_meanings = [
            "Rebases every stock to the same starting point so relative compounding is easy to compare.",
            "Measures how tightly the stocks move together from one day to the next.",
            "Shows when the ride became calmer or more violent over time.",
            "Shows how far each stock fell from its own prior peak during selloffs.",
            "Shows the best historical return available for each level of volatility under long-only weights.",
        ]

        if bundle.pair_insight is not None:
            chart_names.append("Spread Z-Score")
            chart_meanings.append(
                "Shows whether the strongest pair relationship is stretched or close to its recent equilibrium."
            )
        else:
            chart_names.append("Rolling Beta vs Benchmark")
            chart_meanings.append(
                "Shows how aggressively each stock has been moving versus the chosen benchmark over time."
            )

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Chart", "What It Represents"],
                    "fill_color": "#2A6F97",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [chart_names, chart_meanings],
                    "fill_color": "#FFFDFC",
                    "align": "left",
                    "font": {"color": "#132A3E", "size": 12},
                    "height": 34,
                },
            ),
            row=4,
            col=2,
        )

    def _add_stat_guide_table(self, fig: go.Figure) -> None:
        stats = [
            "Symbol",
            "CAGR",
            "Vol",
            "Sharpe",
            "Sortino",
            "Max DD",
            "Beta",
            "RSI",
            "Risk-free rate",
        ]
        meanings = [
            "The ticker being analyzed.",
            "Compounded annual growth rate across the full study window.",
            "Annualized volatility, which measures how violently returns swing.",
            "Return per unit of total risk after comparing against the risk-free rate.",
            "Return per unit of downside risk, penalizing bad volatility more than good volatility.",
            "Worst peak-to-trough decline during the period.",
            "Sensitivity to the benchmark such as SPY or QQQ.",
            "Short-term momentum oscillator on a 0 to 100 scale.",
            "Low-risk baseline return used inside Sharpe and Sortino calculations.",
        ]
        how_to_read = [
            "Just identifies the asset row.",
            "Higher usually means stronger long-run compounding.",
            "Higher means a rougher ride and larger swings.",
            "Higher is better for risk-adjusted performance.",
            "Higher is better, especially when downside control matters.",
            "Closer to zero is better. More negative means deeper pain.",
            "Around 1 is market-like, above 1 is more aggressive, below 1 is calmer.",
            "Above 70 can mean stretched, below 30 can mean washed out.",
            "A higher baseline makes risky assets work harder to look attractive.",
        ]

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Stat", "Meaning", "How To Read It"],
                    "fill_color": "#5C8001",
                    "font": {"color": "white", "size": 13},
                    "align": "left",
                },
                cells={
                    "values": [stats, meanings, how_to_read],
                    "fill_color": "#FFFDFC",
                    "align": "left",
                    "font": {"color": "#132A3E", "size": 12},
                    "height": 34,
                },
            ),
            row=5,
            col=1,
        )

    def _dynamic_subplot_title(self, bundle: AnalysisBundle) -> str:
        if bundle.pair_insight is not None:
            left, right = bundle.pair_insight.symbols
            return f"Spread Z-Score: {left} vs {right}"
        return "Rolling Beta vs Benchmark"
