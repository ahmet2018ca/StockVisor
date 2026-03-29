# StockVisor Quant Edition

StockVisor is a live-data stock analysis tool that blends guided user prompts, quantitative risk analysis, and interactive visualization into one workflow. You choose one stock for a deep dive or a group of stocks for comparison, and StockVisor pulls fresh market data, computes quant metrics, explains what they mean, and generates an HTML dashboard.

At a high level, this project answers questions like:

- Which stock compounded the fastest?
- Which stock had the roughest ride?
- How tightly do these stocks move together?
- How aggressive is a stock relative to the market?
- Was the return strong enough to justify the risk?
- If I combine these names, is there a cleaner risk/return mix?

This README is written in two layers:

- Non-technical: what the tool does and how to use it
- Technical: how the code is structured and how the math works

## What This Project Does

StockVisor turns raw market history into a research-style report.

You pick:

- the stock or stocks to study
- a benchmark like `SPY` or `QQQ`
- a lookback window such as `1`, `3`, or `5` years
- a risk-free rate

Then StockVisor:

1. downloads live market data from Yahoo Finance via `yfinance`
2. engineers return, volatility, drawdown, and momentum features
3. computes quant metrics such as CAGR, Sharpe, Sortino, beta, and max drawdown
4. compares multiple assets using correlation, cointegration, and long-only portfolio optimization
5. prints plain-English explanations in the CLI
6. writes an interactive HTML dashboard

## Who This Is For

This project works for several audiences:

- Non-technical users who want a guided stock research tool
- Students learning investing, risk, and portfolio math
- Python developers interested in financial analytics workflows
- Data-oriented users who want a stronger explanation of what common quant metrics actually mean

## Repo Structure

```text
StockVisor/
├── StockVisor.py
├── pyproject.toml
├── README.md
├── .gitignore
└── stockvisor/
    ├── __init__.py
    ├── analytics.py
    ├── cli.py
    ├── market_data.py
    ├── models.py
    └── reporting.py
```

## Architecture

### Entry Point

[StockVisor.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\StockVisor.py)

This is the thin launcher. It simply calls the package CLI entrypoint.

### CLI Layer

[cli.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\cli.py)

Responsibilities:

- collects user input
- explains the meaning of each configuration step
- resolves defaults
- runs the analysis pipeline
- writes the final report

### Market Data Layer

[market_data.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\market_data.py)

Responsibilities:

- downloads historical OHLCV data from Yahoo Finance through `yfinance`
- normalizes column names
- handles missing values and fallback downloads
- fetches company metadata used in summaries

### Analytics Layer

[analytics.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\analytics.py)

Responsibilities:

- computes returns and log returns
- builds rolling volatility and drawdown series
- computes risk-adjusted performance metrics
- runs CAPM-style regression for alpha and beta
- tests pair relationships using cointegration
- solves long-only portfolio optimization problems

### Reporting Layer

[reporting.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\reporting.py)

Responsibilities:

- renders the CLI tables and plain-English analysis
- explains what each graph means
- explains what each quant stat means
- writes the interactive Plotly dashboard

### Models

[models.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\models.py)

Responsibilities:

- stores strongly typed run configuration
- stores per-asset metric snapshots
- stores pair-analysis results
- stores optimized portfolio outputs

## Technology Stack

### Core Libraries

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `plotly`
- `rich`
- `yfinance`

### Why These Libraries

`pandas`

- aligns market time series by date
- makes rolling calculations straightforward
- simplifies multi-asset tabular analytics

`numpy`

- supports vectorized mathematical computation
- provides fast numerical operations and matrix logic

`scipy`

- supplies optimization routines for portfolio construction
- provides distribution statistics like skewness and kurtosis

`statsmodels`

- supports regression-based market-relative analysis
- provides the cointegration test used in pair analysis

`plotly`

- generates the HTML dashboard
- enables interactive charts without building a web app

`rich`

- improves terminal readability
- makes guided CLI prompts and summary tables easier to understand

`yfinance`

- provides live remote market data
- avoids the old local-CSV workflow

## Installation

From the project folder:

```bash
pip install -e .
```

If your default `python` does not have `pip`, use the Python install that does:

```powershell
& 'C:\Users\Ahmet\AppData\Local\Python\PythonCore-3.14-64\python.exe' -m pip install -e .
```

## Running The Project

### Guided Interactive Mode

```bash
python StockVisor.py
```

This is the best mode if you want the CLI to explain:

- what stocks to choose
- what benchmark means
- what the risk-free rate means
- what the lookback window changes

### Direct CLI Mode

```bash
python StockVisor.py --symbols AAPL MSFT NVDA --benchmark SPY --lookback-years 5 --risk-free-rate 0.04
```

### Example Use Cases

Single-stock deep dive:

```bash
python StockVisor.py --symbols NVDA --benchmark SPY --lookback-years 3 --risk-free-rate 0.04
```

Big-tech comparison:

```bash
python StockVisor.py --symbols AAPL MSFT NVDA AMZN META --benchmark QQQ --lookback-years 5 --risk-free-rate 0.04
```

Sector comparison:

```bash
python StockVisor.py --symbols JPM BAC C GS --benchmark SPY --lookback-years 5 --risk-free-rate 0.04
```

## What The Dashboard Shows

The HTML report includes:

- Normalized Price Paths
- Daily Return Correlation
- Rolling 21-Day Volatility
- Drawdown Curve
- Efficient Frontier
- Spread Z-Score or Rolling Beta vs Benchmark
- Quant Summary Table
- Chart explanations
- Quant-stat explanations

## Quant Concepts Explained

This section is intentionally both practical and technical.

### Risk-Free Rate

Non-technical meaning:

This is the low-risk baseline return you compare stocks against. Think of it as the return you might earn from something much safer than stocks, such as short-term Treasury-style yield or cash-like yield.

Technical meaning:

It is the baseline rate used in risk-adjusted metrics such as Sharpe, Sortino, and alpha. It helps answer whether the return came from real excess performance or simply from taking risk above a safer alternative.

### Return

Daily simple return:

```text
r_t = (P_t / P_{t-1}) - 1
```

Where:

- `P_t` is today’s adjusted close
- `P_{t-1}` is the previous day’s adjusted close

Interpretation:

- positive means the asset rose
- negative means the asset fell

### Log Return

Formula:

```text
log_return_t = ln(P_t / P_{t-1})
```

Why it matters:

- often preferred in quantitative finance because log returns are additive across time
- useful for certain statistical models

### CAGR

Compound Annual Growth Rate.

Formula:

```text
CAGR = (P_end / P_start)^(1 / years) - 1
```

What it means:

- the annualized growth rate over the full period
- smooths the path into a single yearly growth number

How to read it:

- higher usually means stronger long-run compounding
- it says nothing by itself about how volatile the path was

### Volatility

In this project, volatility is annualized standard deviation of returns.

Formula:

```text
Vol = std(daily_returns) * sqrt(252)
```

Why `252`:

- there are roughly 252 trading days in a year

What it means:

- a measure of dispersion in returns
- higher volatility means returns swing more widely

How to read it:

- low volatility usually means a calmer ride
- high volatility usually means larger upside and downside swings

### Downside Volatility

This focuses only on harmful volatility, not all volatility.

Conceptually:

- only returns below a threshold matter
- upside variation is not penalized

Why it matters:

- useful when you care more about bad outcomes than total movement

### Sharpe Ratio

Formula:

```text
Sharpe = (E[R] - R_f) / sigma
```

Where:

- `E[R]` is expected or average return
- `R_f` is the risk-free rate
- `sigma` is total volatility

What it means:

- how much excess return you earned for each unit of total risk

How to read it:

- higher is better
- below `0` is poor
- around `1` is often considered solid

### Sortino Ratio

Formula:

```text
Sortino = (E[R] - R_f) / downside_volatility
```

What it means:

- like Sharpe, but it only punishes downside moves

Why it matters:

- often more intuitive for investors who care more about bad volatility than upside volatility

### Max Drawdown

This measures the worst loss from a previous peak.

Formula concept:

```text
drawdown_t = current_normalized_price / running_peak - 1
max_drawdown = min(drawdown_t)
```

What it means:

- the deepest pain point during the study window

How to read it:

- `-10%` is much gentler than `-60%`
- one of the most emotionally realistic risk metrics

### Beta

Beta measures sensitivity to the benchmark.

Regression idea:

```text
asset_excess_return = alpha + beta * benchmark_excess_return + error
```

What it means:

- `beta = 1`: moves roughly with the benchmark
- `beta > 1`: more aggressive than the benchmark
- `beta < 1`: less aggressive than the benchmark

### Alpha

Alpha is the return not explained by benchmark exposure.

What it means:

- positive alpha suggests outperformance beyond simple market exposure
- negative alpha suggests underperformance after accounting for benchmark behavior

### R-Squared

This measures how much of the asset’s return variation is explained by the benchmark regression.

What it means:

- high `R²` means benchmark behavior explains a lot of the stock’s movement
- low `R²` means the stock has more idiosyncratic behavior

### Tracking Error

Formula concept:

```text
tracking_error = std(asset_return - benchmark_return) * sqrt(252)
```

What it means:

- how far the asset tends to wander from the benchmark

### Information Ratio

Formula concept:

```text
IR = active_return / tracking_error
```

What it means:

- excess return versus the benchmark per unit of benchmark-relative risk

### Value at Risk (VaR)

In this project, VaR is estimated from the historical return distribution.

Concept:

- the 5th percentile of returns at the `95%` confidence level

What it means:

- a threshold for bad days
- it answers: “How bad do returns get in the left tail before only the worst 5% remain?”

### Conditional Value at Risk (CVaR)

Also called Expected Shortfall.

What it means:

- the average return once you are already in the bad tail beyond VaR

Why it matters:

- often more informative than VaR because it measures the severity of tail damage, not just the cutoff

### Skewness

Skewness measures asymmetry in the return distribution.

What it means:

- positive skew means more upside tail behavior
- negative skew means more downside tail behavior

### Kurtosis

Kurtosis measures tail heaviness and extremity.

What it means:

- higher kurtosis suggests more extreme outcomes than a normal distribution would imply

### RSI

Relative Strength Index.

Range:

- from `0` to `100`

Typical interpretation:

- above `70`: extended or overbought
- below `30`: weak or oversold

What it means:

- a short-term momentum oscillator, not a guarantee of reversal

### MACD

Moving Average Convergence Divergence.

Concept:

- compares a fast exponential moving average to a slow one
- often paired with a signal line

What it means:

- helps identify momentum direction and changes in momentum

### ATR

Average True Range.

What it measures:

- trading-range expansion and contraction
- a practical volatility measure based on price ranges, not just close-to-close returns

### Correlation

Formula concept:

```text
Corr(X, Y) = Cov(X, Y) / (std(X) * std(Y))
```

What it means:

- `1` means they move together perfectly
- `0` means no linear relationship
- `-1` means they move opposite each other perfectly

Why it matters:

- high correlation usually means less diversification benefit

### Cointegration

Cointegration is stricter than correlation.

What it means:

- two assets may drift together in a statistically stable long-run relationship

Why it matters:

- useful in relative-value and pair-trading style analysis
- correlation alone does not guarantee mean reversion

### Efficient Frontier

The efficient frontier is the set of long-only portfolios with the best expected return for each volatility level.

Optimization idea:

- choose weights that minimize volatility for a target return
- or maximize Sharpe subject to long-only weights summing to 1

In this project:

- weights are constrained between `0` and `1`
- total weights sum to `1`
- no short selling is used

## Technical Workflow

The internal workflow is:

1. collect user configuration
2. download remote historical data
3. normalize columns and align dates
4. engineer return and indicator features
5. compute point-in-time summary metrics
6. compute multi-asset relationship metrics
7. solve portfolio optimization problems
8. render CLI explanations
9. render the Plotly HTML dashboard

## Quant Math Used In Code

The main quantitative computations live in [analytics.py](C:\CODES\RESUME__PROJECTS\STOCK__VISOR\StockVisor\stockvisor\analytics.py).

Implemented families of computation include:

- simple returns
- log returns
- annualized volatility
- downside volatility
- CAGR
- drawdown series
- RSI
- MACD
- ATR
- CAPM-style beta and alpha regression
- tracking error and information ratio
- historical VaR
- historical CVaR
- skewness
- kurtosis
- correlation matrix
- cointegration matrix
- long-only efficient frontier
- max-Sharpe portfolio
- minimum-volatility portfolio

## Design Decisions

### Why Live Data Instead of Local CSV Files

The original repo depended on static local files. The rewrite intentionally moved to live remote data so each run reflects current market history.

Benefits:

- no manual CSV preparation
- fresher analysis
- easier onboarding

Tradeoff:

- the tool now depends on external data availability

### Why Long-Only Optimization

Long-only portfolios are:

- easier to explain
- more stable for casual users
- closer to how many real users think about allocation

This keeps the math useful without overcomplicating the interface.

### Why Explain The Metrics In The CLI And HTML

Quant tools often fail by assuming the user already knows the vocabulary. StockVisor deliberately explains:

- the setup choices
- the graphs
- the stats

So the project is not just computationally stronger, but more interpretable.

## Limitations

- Yahoo Finance data via `yfinance` is useful for research, not institutional execution
- historical optimization is sensitive to the chosen lookback window
- risk metrics are backward-looking, not predictive guarantees
- correlation can shift quickly during market stress
- cointegration is suggestive, not a full trading system
- this is a research and educational tool, not financial advice

## Future Directions

Good technical extensions from here would include:

- factor-model exposure using Fama-French data
- Black-Litterman portfolio construction
- Monte Carlo scenario analysis
- rolling regime detection
- earnings-event annotations
- exportable PDF reports
- a web interface

## Bottom Line

StockVisor is no longer a toy stock script. It is now a structured quantitative analysis project with:

- live remote data
- guided CLI onboarding
- technical metrics with explanations
- mathematically grounded risk analysis
- portfolio optimization
- interactive HTML reporting

If you want a project that is understandable to a beginner but still grounded in real financial math and software structure, that is exactly what this rewrite is designed to be.
