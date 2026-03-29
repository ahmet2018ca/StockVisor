from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
import webbrowser

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from .analytics import QuantAnalyzer
from .market_data import MarketDataError, YahooFinanceClient
from .models import RunConfiguration
from .reporting import ConsoleReporter, PlotlyReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive quant-grade stock analysis and comparison engine.",
    )
    parser.add_argument("--symbols", nargs="*", help="Ticker symbols to analyze, e.g. AAPL MSFT NVDA")
    parser.add_argument("--benchmark", help="Benchmark ticker used for beta and alpha calculations")
    parser.add_argument("--lookback-years", type=int, help="Historical lookback window in years")
    parser.add_argument("--risk-free-rate", type=float, help="Annual risk-free rate as a decimal, e.g. 0.04")
    parser.add_argument(
        "--output",
        default="stockvisor_quant_report.html",
        help="HTML dashboard output path",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated HTML report in your default browser",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()
    fully_interactive = (
        args.symbols is None
        and args.benchmark is None
        and args.lookback_years is None
        and args.risk_free_rate is None
    )

    console.print(
        Panel.fit(
            "Pick one stock for a focused deep dive or a basket of stocks for a comparison study. "
            "StockVisor will pull live Yahoo Finance data, explain what each setting changes, and then turn "
            "that into a quant-style read on growth, risk, correlation, benchmark behavior, and portfolio fit.",
            title="StockVisor Quant Launcher",
            border_style="#16324F",
        )
    )

    config = _resolve_configuration(args, console, fully_interactive)
    _print_analysis_plan(console, config)
    end = datetime.utcnow() + timedelta(days=1)
    start = end - timedelta(days=365 * config.lookback_years)

    unique_symbols = _unique_preserve_order(config.symbols + [config.benchmark])
    data_client = YahooFinanceClient()

    try:
        console.print(f"[bold #16324F]Loading remote market data for:[/] {', '.join(unique_symbols)}")
        histories = data_client.fetch_histories(unique_symbols, start=start, end=end)
        overviews = {symbol: data_client.fetch_overview(symbol) for symbol in config.symbols}
    except MarketDataError as error:
        console.print(f"[bold red]Market data error:[/] {error}")
        return 1
    except Exception as error:  # pragma: no cover - defensive runtime handling
        console.print(f"[bold red]Unexpected failure while downloading market data:[/] {error}")
        return 1

    analyzer = QuantAnalyzer(config=config, histories=histories, overviews=overviews)
    bundle = analyzer.analyze()

    ConsoleReporter().render(bundle)
    dashboard_path = PlotlyReporter().write_dashboard(bundle)

    console.print(
        Panel.fit(
            f"Interactive dashboard written to:\n{dashboard_path}",
            title="Report Ready",
            border_style="#2A6F97",
        )
    )

    if args.open_report or (_is_interactive() and Confirm.ask("Open the HTML dashboard now?", default=False)):
        webbrowser.open(dashboard_path.as_uri())

    return 0


def _resolve_configuration(
    args: argparse.Namespace,
    console: Console,
    fully_interactive: bool,
) -> RunConfiguration:
    output_path = Path(args.output)
    if not output_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / output_path

    if not fully_interactive:
        symbols = [symbol.upper() for symbol in args.symbols] if args.symbols else _prompt_symbols(console)
        benchmark = args.benchmark.upper() if args.benchmark else _prompt_benchmark(console, symbols)
        lookback_years = args.lookback_years if args.lookback_years else _prompt_lookback(console, symbols)
        risk_free_rate = args.risk_free_rate if args.risk_free_rate is not None else _prompt_risk_free_rate(console)
        return RunConfiguration(
            symbols=symbols,
            benchmark=benchmark,
            lookback_years=lookback_years,
            risk_free_rate=risk_free_rate,
            output_path=output_path,
        )

    while True:
        _print_guided_overview(console)
        symbols = _prompt_symbols(console)
        benchmark = _prompt_benchmark(console, symbols)
        lookback_years = _prompt_lookback(console, symbols)
        risk_free_rate = _prompt_risk_free_rate(console)

        config = RunConfiguration(
            symbols=symbols,
            benchmark=benchmark,
            lookback_years=lookback_years,
            risk_free_rate=risk_free_rate,
            output_path=output_path,
        )
        _print_selection_summary(console, config)
        if Confirm.ask("Run this analysis?", default=True):
            return config
        console.print(
            Panel.fit(
                "No problem. Let's adjust the setup and shape the study a different way.",
                title="Rebuild The Study",
                border_style="#C46B2E",
            )
        )


def _prompt_symbols(console: Console) -> list[str]:
    console.print(
        Panel.fit(
            "Step 1: choose what you want to study.\n\n"
            "Pick a single stock like AAPL or NVDA for a deep dive, or enter a basket like "
            "AAPL,MSFT,NVDA to compare leadership, volatility, correlation, and portfolio fit.\n\n"
            "If you just want a quick demo run, pressing Enter will use AAPL, MSFT, and NVDA.\n\n"
            "Ideas:\n"
            "- Big tech: AAPL,MSFT,NVDA\n"
            "- Semiconductors: NVDA,AMD,AVGO,TSM\n"
            "- Banks: JPM,BAC,C\n"
            "- Single-name deep dive: TSLA",
            title="Choose Stocks",
            border_style="#16324F",
        )
    )
    while True:
        raw = Prompt.ask(
            "Pick a stock or stock basket",
            default="AAPL,MSFT,NVDA",
            show_default=False,
        )
        symbols = _parse_symbols(raw)
        if 1 <= len(symbols) <= 8:
            return symbols
        if len(symbols) > 8:
            console.print(
                "[bold yellow]Try to keep it to 1-8 tickers so the comparison stays readable and the dashboard stays useful.[/]"
            )
        else:
            console.print("[bold red]Please enter at least one valid ticker symbol.[/]")


def _prompt_benchmark(console: Console, symbols: list[str]) -> str:
    console.print(
        Panel.fit(
            "Step 2: pick a benchmark.\n\n"
            "This gives your study a reference point. It helps StockVisor estimate beta, alpha, and whether your picks "
            "are simply following the market or actually behaving differently.\n\n"
            "Pressing Enter will use a suggested benchmark based on your basket.\n\n"
            "Common choices:\n"
            "- SPY for the broad U.S. market\n"
            "- QQQ for tech and growth\n"
            "- IWM for smaller-cap exposure\n"
            "- XLK for a tech-sector lens",
            title="Choose Benchmark",
            border_style="#2A6F97",
        )
    )
    default = _default_benchmark(symbols)
    return Prompt.ask(
        "Benchmark ticker",
        default=default,
        show_default=False,
    ).upper().strip()


def _prompt_lookback(console: Console, symbols: list[str]) -> int:
    study_type = "single-stock trend study" if len(symbols) == 1 else "comparison study"
    console.print(
        Panel.fit(
            "Step 3: choose how much history to include.\n\n"
            f"For a {study_type}, the lookback window changes what you learn:\n"
            "- 1 year highlights current regime, recent momentum, and near-term drawdowns\n"
            "- 3 years balances recent behavior with a broader market cycle\n"
            "- 5 to 10 years shows durability, long drawdowns, and how stable the relationship has been over time\n\n"
            "Pressing Enter will use 5 years.",
            title="Choose Lookback Window",
            border_style="#5C8001",
        )
    )
    while True:
        lookback = IntPrompt.ask("Lookback window in years", default=5, show_default=False)
        if 1 <= lookback <= 15:
            return lookback
        console.print("[bold yellow]Choose a window between 1 and 15 years for a cleaner and faster study.[/]")


def _prompt_risk_free_rate(console: Console) -> float:
    console.print(
        Panel.fit(
            "Step 4: choose the risk-free rate.\n\n"
            "The risk-free rate is your baseline annual return from something very low-risk, usually short-term U.S. "
            "Treasury-style returns or 'cash-like' yield.\n\n"
            "StockVisor uses it in Sharpe, Sortino, and alpha calculations to ask a harder question: "
            "'Was taking stock risk actually worth it compared with the low-risk alternative?'\n\n"
            "If you raise it, risky stocks have to earn more to look impressive on a risk-adjusted basis. "
            "If you are just exploring, 4.0 is a perfectly fine default.",
            title="Choose Risk-Free Rate",
            border_style="#A23E48",
        )
    )
    percent = FloatPrompt.ask("Annual risk-free rate (%)", default=4.0, show_default=False)
    return percent / 100.0


def _parse_symbols(raw: str) -> list[str]:
    cleaned = raw.replace(";", ",").strip()
    return _unique_preserve_order(
        [
            symbol.strip().upper()
            for symbol in re.split(r"[\s,]+", cleaned)
            if symbol.strip()
        ]
    )


def _unique_preserve_order(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.add(symbol)
            ordered.append(symbol)
    return ordered


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_guided_overview(console: Console) -> None:
    console.print(
        Panel.fit(
            "You are not locked into one style of analysis here.\n\n"
            "- Pick one stock if you want a deep read on trend, volatility, drawdowns, and benchmark behavior.\n"
            "- Pick several stocks if you want to compare leaders, laggards, correlation, pair behavior, and possible portfolio mixes.\n"
            "- Use a broad benchmark like SPY if you want a market-relative read, or a sector benchmark like XLK if you want a tighter lens.",
            title="Build Your Study",
            border_style="#16324F",
        )
    )


def _print_selection_summary(console: Console, config: RunConfiguration) -> None:
    symbols_text = ", ".join(config.symbols)
    study_mode = _study_mode_text(config.symbols)
    console.print(
        Panel.fit(
            f"We're about to run a {study_mode} on {symbols_text} against {config.benchmark} over {config.lookback_years} "
            f"year{'s' if config.lookback_years != 1 else ''} with a {config.risk_free_rate:.2%} risk-free rate.\n\n"
            f"{_selection_takeaway(config)}",
            title="Study Summary",
            border_style="#C46B2E",
        )
    )


def _print_analysis_plan(console: Console, config: RunConfiguration) -> None:
    lines = [
        f"Stocks: {', '.join(config.symbols)}",
        f"Benchmark: {config.benchmark}",
        f"Lookback: {config.lookback_years} year{'s' if config.lookback_years != 1 else ''}",
        f"Risk-free rate: {config.risk_free_rate:.2%}",
        "",
        "What StockVisor will study for you:",
        "- price path and CAGR to show how strongly each name compounded",
        "- volatility and drawdown to show how rough the ride was",
        "- beta and alpha to show how much behavior came from the benchmark",
    ]
    if len(config.symbols) > 1:
        lines.extend(
            [
                "- correlation and cointegration to show who tends to move together",
                "- efficient-frontier math to show how the basket mixes under long-only weights",
            ]
        )
    else:
        lines.append("- momentum, RSI, MACD, and tail-risk measures to deepen the single-name read")

    console.print(
        Panel.fit(
            "\n".join(lines),
            title="Analysis Plan",
            border_style="#2A6F97",
        )
    )


def _default_benchmark(symbols: list[str]) -> str:
    tech_tickers = {"AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSM", "META", "GOOGL", "AMZN", "NFLX", "TSLA"}
    if symbols and all(symbol in tech_tickers for symbol in symbols):
        return "QQQ"
    return "SPY"


def _study_mode_text(symbols: list[str]) -> str:
    return "single-stock deep dive" if len(symbols) == 1 else "multi-stock comparison"


def _selection_takeaway(config: RunConfiguration) -> str:
    if len(config.symbols) == 1:
        return (
            "This setup is best when you want to understand one company in context: trend quality, drawdowns, "
            "risk-adjusted performance, and how aggressively it moves versus the benchmark."
        )
    return (
        "This setup is best when you want to compare leadership, diversification, pair behavior, and whether a mix "
        "of these names can produce a cleaner risk/return profile than any one stock on its own."
    )
