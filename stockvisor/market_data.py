from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Raised when remote market data cannot be loaded or parsed."""


@dataclass(slots=True)
class SymbolBundle:
    history: pd.DataFrame
    overview: dict[str, Any]


class YahooFinanceClient:
    """Fetches remote equity data from Yahoo Finance through yfinance."""

    COLUMN_MAP = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }

    def fetch_histories(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )

        if raw.empty:
            raise MarketDataError("Yahoo Finance returned an empty dataset.")

        histories: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._extract_symbol_frame(raw, symbol, len(symbols) == 1)
            if frame.empty:
                frame = self._download_single_symbol(symbol, start, end)

            normalized = self._normalize_history(frame)
            if normalized.empty:
                raise MarketDataError(f"No usable adjusted-close history was returned for ticker '{symbol}'.")
            histories[symbol] = normalized

        return histories

    def fetch_overview(self, symbol: str) -> dict[str, Any]:
        ticker = yf.Ticker(symbol)
        info: dict[str, Any] = {}

        for loader in (ticker.get_info, lambda: ticker.info):
            try:
                info = loader() or {}
            except Exception:
                continue
            if info:
                break

        try:
            analyst_targets = ticker.get_analyst_price_targets() or {}
        except Exception:
            analyst_targets = {}

        if analyst_targets:
            info["analyst_target_mean"] = analyst_targets.get("mean")
            info["analyst_target_median"] = analyst_targets.get("median")
            info["analyst_target_high"] = analyst_targets.get("high")
            info["analyst_target_low"] = analyst_targets.get("low")

        return info

    def fetch_symbol_bundle(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> SymbolBundle:
        return SymbolBundle(history=history, overview=self.fetch_overview(symbol))

    def _extract_symbol_frame(
        self,
        raw: pd.DataFrame,
        symbol: str,
        single_symbol: bool,
    ) -> pd.DataFrame:
        if single_symbol and not isinstance(raw.columns, pd.MultiIndex):
            return raw.copy()

        if not isinstance(raw.columns, pd.MultiIndex):
            return raw.copy()

        try:
            return raw.xs(symbol, axis=1, level=0).copy()
        except KeyError as error:
            return pd.DataFrame()

    def _download_single_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        try:
            return yf.Ticker(symbol).history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                raise_errors=True,
            )
        except Exception as error:
            raise MarketDataError(f"Unable to download price history for ticker '{symbol}'.") from error

    def _normalize_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.rename(columns=self.COLUMN_MAP).copy()
        normalized.index = pd.to_datetime(normalized.index)
        if getattr(normalized.index, "tz", None) is not None:
            normalized.index = normalized.index.tz_localize(None)
        normalized.index.name = "date"

        numeric_columns = [column for column in normalized.columns if column in self.COLUMN_MAP.values()]
        normalized[numeric_columns] = normalized[numeric_columns].apply(pd.to_numeric, errors="coerce")
        normalized = normalized.sort_index()

        required = ["open", "high", "low", "close", "adjusted_close", "volume"]
        missing = [column for column in required if column not in normalized.columns]
        if missing:
            raise MarketDataError(f"Downloaded history is missing expected columns: {', '.join(missing)}.")

        normalized = normalized.dropna(subset=["adjusted_close"])
        if "dividends" not in normalized.columns:
            normalized["dividends"] = 0.0
        else:
            normalized["dividends"] = normalized["dividends"].fillna(0.0)

        if "stock_splits" not in normalized.columns:
            normalized["stock_splits"] = 0.0
        else:
            normalized["stock_splits"] = normalized["stock_splits"].fillna(0.0)
        return normalized
