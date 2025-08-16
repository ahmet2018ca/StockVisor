"""
StockVisor - Financial Quantitative Analytics Platform

A modular platform for computing key performance metrics including correlation coefficients,
volatility measures, and trend indicators across multiple equities.

Features:
- Data processing pipelines with NumPy vectorized operations
- Interactive visual analytics with Matplotlib
- Flexible correlation analysis between ANY stock metrics
- Comprehensive volatility measures and risk metrics
- High-efficiency computation on large historical datasets
"""

import csv
import glob
import logging
import mplcursors
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StockMetrics:
    """Data class for stock performance metrics."""
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    adjusted_close: float
    previous_close: Optional[float]
    volume: int
    change_percentage: float
    change_absolute: float
    daily_return: Optional[float] = None
    price_range: Optional[float] = None
    volume_ma: Optional[float] = None
    price_ma: Optional[float] = None
    volatility: Optional[float] = None
    rolling_volatility: Optional[float] = None


class DataProcessor:
    """Handles data processing and transformation operations."""
    
    @staticmethod
    def parse_float(value: str) -> float:
        """Parse string to float with precision."""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.error(f"Invalid float format: {value}")
            return 0.0
    
    @staticmethod
    def parse_int(value: str) -> int:
        """Parse string to integer."""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.error(f"Invalid int format: {value}")
            return 0
    
    @staticmethod
    def calculate_change_percentage(open_val: float, close_val: float) -> float:
        """Calculate percentage change from open to close."""
        if open_val == 0:
            return 0.0
        return ((close_val - open_val) / open_val) * 100
    
    @staticmethod
    def calculate_change_absolute(open_val: float, close_val: float) -> float:
        """Calculate absolute change from open to close."""
        return close_val - open_val
    
    @staticmethod
    def calculate_daily_return(current_price: float, previous_price: float) -> float:
        """Calculate daily return."""
        if previous_price == 0:
            return 0.0
        return (current_price - previous_price) / previous_price
    
    @staticmethod
    def calculate_price_range(high: float, low: float) -> float:
        """Calculate daily price range."""
        return high - low


class VolatilityAnalyzer:
    """Comprehensive volatility analysis and risk metrics."""
    
    @staticmethod
    def calculate_historical_volatility(daily_returns: List[float], periods: int = 252) -> float:
        """Calculate annualized historical volatility."""
        if len(daily_returns) < 2:
            return 0.0
        
        # Remove None values
        returns = [r for r in daily_returns if r is not None]
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation and annualize
        volatility = np.std(returns) * np.sqrt(periods)
        return volatility
    
    @staticmethod
    def calculate_rolling_volatility(daily_returns: List[float], window: int = 30) -> List[float]:
        """Calculate rolling volatility over a specified window."""
        rolling_vols = []
        
        for i in range(len(daily_returns)):
            if i < window - 1:
                rolling_vols.append(None)
            else:
                window_returns = daily_returns[max(0, i - window + 1):i + 1]
                window_returns = [r for r in window_returns if r is not None]
                
                if len(window_returns) >= 2:
                    vol = np.std(window_returns) * np.sqrt(252)
                    rolling_vols.append(vol)
                else:
                    rolling_vols.append(None)
        
        return rolling_vols
    
    @staticmethod
    def calculate_value_at_risk(daily_returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR) at specified confidence level."""
        returns = [r for r in daily_returns if r is not None]
        if len(returns) < 10:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        returns = [r for r in daily_returns if r is not None]
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        avg_excess_return = np.mean(excess_returns)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        return (avg_excess_return / volatility) * np.sqrt(252)
    
    def analyze_volatility_metrics(self, stock_data: Dict[str, Dict[str, StockMetrics]]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive volatility metrics for all stocks."""
        volatility_metrics = {}
        
        for stock_symbol, dates_data in stock_data.items():
            daily_returns = []
            
            # Extract daily returns
            for date, metrics in sorted(dates_data.items()):
                if metrics.daily_return is not None:
                    daily_returns.append(metrics.daily_return)
            
            if len(daily_returns) < 10:
                continue
            
            # Calculate various volatility measures
            hist_vol = self.calculate_historical_volatility(daily_returns)
            var_5 = self.calculate_value_at_risk(daily_returns, 0.05)
            var_1 = self.calculate_value_at_risk(daily_returns, 0.01)
            sharpe = self.calculate_sharpe_ratio(daily_returns)
            
            volatility_metrics[stock_symbol] = {
                'historical_volatility': round(hist_vol, 4),
                'var_5_percent': round(var_5, 4),
                'var_1_percent': round(var_1, 4),
                'sharpe_ratio': round(sharpe, 4),
                'max_daily_return': round(max(daily_returns), 4),
                'min_daily_return': round(min(daily_returns), 4),
                'avg_daily_return': round(np.mean(daily_returns), 4)
            }
        
        return volatility_metrics


class FileManager:
    """Handles file operations and data loading."""
    
    @staticmethod
    def find_stock_files() -> List[str]:
        """Find all NASDAQ stock CSV files in current directory."""
        pattern = "NASDAQ_STOCK*.csv"
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} stock files: {files}")
        return files
    
    @staticmethod
    def extract_stock_symbol(filename: str) -> str:
        """Extract stock symbol from filename."""
        try:
            return Path(filename).stem.split("_")[2]
        except IndexError:
            logger.error(f"Invalid filename format: {filename}")
            return "UNKNOWN"


class FlexibleCorrelationAnalyzer:
    """Analyzes correlations between ANY stock metrics."""
    
    AVAILABLE_METRICS = [
        'open_price', 'close_price', 'high_price', 'low_price', 
        'adjusted_close', 'volume', 'change_percentage', 'change_absolute',
        'daily_return', 'price_range', 'volume_ma', 'price_ma', 'volatility', 'rolling_volatility'
    ]
    
    @staticmethod
    def get_metric_values(stock_data: Dict[str, Dict[str, StockMetrics]], 
                         stock_symbol: str, metric_name: str) -> List[float]:
        """Extract values for a specific metric from stock data."""
        if stock_symbol not in stock_data:
            return []
        
        values = []
        for date, metrics in stock_data[stock_symbol].items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                if value is not None:
                    values.append(float(value))
        
        return values
    
    @staticmethod
    def calculate_correlation(values1: List[float], values2: List[float]) -> float:
        """Calculate correlation coefficient between two sets of values."""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        try:
            correlation_matrix = np.corrcoef(values1, values2)
            return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def analyze_metric_correlation(self, stock_data: Dict[str, Dict[str, StockMetrics]], 
                                 metric1: str, metric2: str) -> Dict[str, float]:
        """Calculate correlation between two metrics for all stocks."""
        correlations = {}
        
        for stock_symbol in stock_data.keys():
            values1 = self.get_metric_values(stock_data, stock_symbol, metric1)
            values2 = self.get_metric_values(stock_data, stock_symbol, metric2)
            
            min_length = min(len(values1), len(values2))
            if min_length > 1:
                correlation = self.calculate_correlation(values1[:min_length], values2[:min_length])
                correlations[stock_symbol] = round(correlation, 4)
            else:
                correlations[stock_symbol] = 0.0
        
        return correlations
    
    def create_correlation_matrix(self, stock_data: Dict[str, Dict[str, StockMetrics]], 
                                stock_symbol: str, metrics_list: List[str] = None) -> pd.DataFrame:
        """Create correlation matrix for multiple metrics of a single stock."""
        if metrics_list is None:
            metrics_list = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 'change_absolute', 'volatility']
        
        data_dict = {}
        for metric in metrics_list:
            values = self.get_metric_values(stock_data, stock_symbol, metric)
            if values:
                data_dict[metric] = values
        
        if not data_dict:
            return pd.DataFrame()
        
        # Make all arrays the same length
        min_length = min(len(values) for values in data_dict.values())
        for metric in data_dict:
            data_dict[metric] = data_dict[metric][:min_length]
        
        df = pd.DataFrame(data_dict)
        return df.corr()


class EnhancedVisualizer:
    """Creates enhanced visualizations for flexible correlation analysis."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> None:
        """Create a correlation heatmap from a correlation matrix."""
        if correlation_matrix.empty:
            logger.warning("Cannot create heatmap: empty correlation matrix")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        ax.set_title(f'StockVisor - {title}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_comparison(self, volatility_metrics: Dict[str, Dict[str, float]]) -> None:
        """Create volatility comparison charts."""
        if not volatility_metrics:
            logger.warning("No volatility data to plot")
            return
        
        stocks = list(volatility_metrics.keys())
        hist_vols = [volatility_metrics[stock]['historical_volatility'] for stock in stocks]
        sharpe_ratios = [volatility_metrics[stock]['sharpe_ratio'] for stock in stocks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Historical Volatility Chart
        colors1 = ['red' if x > 0.3 else 'orange' if x > 0.2 else 'green' for x in hist_vols]
        bars1 = ax1.bar(stocks, hist_vols, color=colors1, alpha=0.7)
        ax1.set_title('Historical Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Volatility', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, hist_vols):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Sharpe Ratio Chart
        colors2 = ['green' if x > 1 else 'orange' if x > 0 else 'red' for x in sharpe_ratios]
        bars2 = ax2.bar(stocks, sharpe_ratios, color=colors2, alpha=0.7)
        ax2.set_title('Sharpe Ratio', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars2, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metric_comparison(self, correlations: Dict[str, float], 
                             metric1: str, metric2: str) -> None:
        """Create bar chart comparing correlations across stocks."""
        if not correlations:
            logger.warning("No correlation data to plot")
            return
        
        stocks = list(correlations.keys())
        corr_values = list(correlations.values())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['darkred' if x < -0.5 else 'red' if x < 0 else 'lightgreen' if x < 0.5 else 'darkgreen' for x in corr_values]
        bars = ax.bar(stocks, corr_values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_title(f'Correlation: {metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_xlabel('Stock Symbol', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong Positive')
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong Negative')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, corr_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                   f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_analysis(self, stock_data: Dict[str, Dict[str, StockMetrics]], 
                            stock_symbol: str, metric1: str, metric2: str) -> None:
        """Create scatter plot to visualize correlation between two metrics."""
        analyzer = FlexibleCorrelationAnalyzer()
        values1 = analyzer.get_metric_values(stock_data, stock_symbol, metric1)
        values2 = analyzer.get_metric_values(stock_data, stock_symbol, metric2)
        
        if len(values1) < 2 or len(values2) < 2:
            logger.warning(f"Insufficient data for scatter plot: {stock_symbol}")
            return
        
        min_length = min(len(values1), len(values2))
        values1 = values1[:min_length]
        values2 = values2[:min_length]
        
        correlation = analyzer.calculate_correlation(values1, values2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(values1, values2, alpha=0.6, c=range(len(values1)), cmap='viridis')
        
        # Add trend line
        z = np.polyfit(values1, values2, 1)
        p = np.poly1d(z)
        ax.plot(values1, p(values1), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel(f'{metric1.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel(f'{metric2.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'{stock_symbol} - {metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}\n'
                    f'Correlation: {correlation:.4f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for time progression
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Progression', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.show()


class StockVisor:
    """Enhanced StockVisor with flexible correlation analysis and volatility measures."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.file_manager = FileManager()
        self.correlation_analyzer = FlexibleCorrelationAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.visualizer = EnhancedVisualizer()
        self.stock_data: Dict[str, Dict[str, StockMetrics]] = {}
        self.volatility_metrics: Dict[str, Dict[str, float]] = {}
    
    def load_stock_data(self) -> None:
        """Load and process stock data from CSV files."""
        stock_files = self.file_manager.find_stock_files()
        
        if not stock_files:
            logger.error("No stock files found!")
            return
        
        logger.info("Loading stock data...")
        
        for file_path in stock_files:
            stock_symbol = self.file_manager.extract_stock_symbol(file_path)
            self._process_stock_file(file_path, stock_symbol)
        
        # Calculate additional metrics including volatility
        self._calculate_additional_metrics()
        
        # Calculate volatility metrics
        self.volatility_metrics = self.volatility_analyzer.analyze_volatility_metrics(self.stock_data)
        
        logger.info(f"Loaded data for {len(self.stock_data)} stocks")
    
    def _process_stock_file(self, file_path: str, stock_symbol: str) -> None:
        """Process individual stock CSV file."""
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                self.stock_data[stock_symbol] = {}
                previous_close = None
                
                for row in reader:
                    date = row["Date"]
                    
                    # Process all available metrics
                    open_price = self.data_processor.parse_float(row.get("Open", "0"))
                    close_price = self.data_processor.parse_float(row.get("Close", "0"))
                    high_price = self.data_processor.parse_float(row.get("High", "0"))
                    low_price = self.data_processor.parse_float(row.get("Low", "0"))
                    adjusted_close = self.data_processor.parse_float(row.get("Adj Close", str(close_price)))
                    volume = self.data_processor.parse_int(row.get("Volume", "0"))
                    
                    # Calculate derived metrics
                    change_pct = self.data_processor.calculate_change_percentage(open_price, close_price)
                    change_abs = self.data_processor.calculate_change_absolute(open_price, close_price)
                    daily_return = self.data_processor.calculate_daily_return(close_price, previous_close) if previous_close else None
                    price_range = self.data_processor.calculate_price_range(high_price, low_price)
                    
                    # Create metrics object
                    metrics = StockMetrics(
                        open_price=open_price,
                        close_price=close_price,
                        high_price=high_price,
                        low_price=low_price,
                        adjusted_close=adjusted_close,
                        previous_close=previous_close,
                        volume=volume,
                        change_percentage=change_pct,
                        change_absolute=change_abs,
                        daily_return=daily_return,
                        price_range=price_range
                    )
                    
                    self.stock_data[stock_symbol][date] = metrics
                    previous_close = close_price
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def _calculate_additional_metrics(self) -> None:
        """Calculate moving averages, volatility, and other derived metrics."""
        for stock_symbol in self.stock_data:
            dates = sorted(self.stock_data[stock_symbol].keys())
            daily_returns = []
            
            # Collect daily returns for volatility calculation
            for date in dates:
                if self.stock_data[stock_symbol][date].daily_return is not None:
                    daily_returns.append(self.stock_data[stock_symbol][date].daily_return)
            
            # Calculate rolling volatility
            rolling_vols = self.volatility_analyzer.calculate_rolling_volatility(daily_returns)
            vol_index = 0
            
            for i, date in enumerate(dates):
                # Calculate 10-day moving averages
                start_idx = max(0, i - 9)
                recent_dates = dates[start_idx:i+1]
                
                volume_values = [self.stock_data[stock_symbol][d].volume for d in recent_dates]
                price_values = [self.stock_data[stock_symbol][d].close_price for d in recent_dates]
                
                self.stock_data[stock_symbol][date].volume_ma = np.mean(volume_values)
                self.stock_data[stock_symbol][date].price_ma = np.mean(price_values)
                
                # Add rolling volatility
                if self.stock_data[stock_symbol][date].daily_return is not None:
                    if vol_index < len(rolling_vols):
                        self.stock_data[stock_symbol][date].rolling_volatility = rolling_vols[vol_index]
                    vol_index += 1
    
    def analyze_volatility(self) -> Dict[str, Dict[str, float]]:
        """Analyze and display volatility metrics."""
        logger.info("\nVolatility Analysis:")
        logger.info("=" * 60)
        
        for stock, metrics in self.volatility_metrics.items():
            logger.info(f"\n{stock}:")
            logger.info(f"  Historical Volatility (Annual): {metrics['historical_volatility']:.4f}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"  VaR (5%): {metrics['var_5_percent']:.4f}")
            logger.info(f"  VaR (1%): {metrics['var_1_percent']:.4f}")
            logger.info(f"  Max Daily Return: {metrics['max_daily_return']:.4f}")
            logger.info(f"  Min Daily Return: {metrics['min_daily_return']:.4f}")
            logger.info(f"  Average Daily Return: {metrics['avg_daily_return']:.4f}")
        
        return self.volatility_metrics
    
    def visualize_volatility(self) -> None:
        """Create volatility visualizations."""
        if self.volatility_metrics:
            self.visualizer.plot_volatility_comparison(self.volatility_metrics)
    
    def analyze_custom_correlation(self, metric1: str, metric2: str) -> Dict[str, float]:
        """Analyze correlation between any two metrics."""
        if metric1 not in self.correlation_analyzer.AVAILABLE_METRICS:
            logger.error(f"Metric '{metric1}' not available. Available: {self.correlation_analyzer.AVAILABLE_METRICS}")
            return {}
        
        if metric2 not in self.correlation_analyzer.AVAILABLE_METRICS:
            logger.error(f"Metric '{metric2}' not available. Available: {self.correlation_analyzer.AVAILABLE_METRICS}")
            return {}
        
        correlations = self.correlation_analyzer.analyze_metric_correlation(self.stock_data, metric1, metric2)
        
        logger.info(f"\nCorrelation Analysis: {metric1} vs {metric2}")
        logger.info("=" * 50)
        for stock, corr in correlations.items():
            interpretation = self._interpret_correlation(corr)
            logger.info(f"{stock}: {corr:>8.4f} ({interpretation})")
        
        return correlations
    
    def create_full_correlation_matrix(self, stock_symbol: str) -> None:
        """Create correlation matrix for all metrics of a specific stock."""
        if stock_symbol not in self.stock_data:
            logger.error(f"Stock {stock_symbol} not found")
            return
        
        correlation_matrix = self.correlation_analyzer.create_correlation_matrix(
            self.stock_data, stock_symbol, self.correlation_analyzer.AVAILABLE_METRICS
        )
        
        if not correlation_matrix.empty:
            self.visualizer.plot_correlation_heatmap(
                correlation_matrix, f"{stock_symbol} - All Metrics Correlation Matrix"
            )
    
    def visualize_metric_correlation(self, metric1: str, metric2: str) -> None:
        """Visualize correlation between two metrics."""
        correlations = self.analyze_custom_correlation(metric1, metric2)
        if correlations:
            self.visualizer.plot_metric_comparison(correlations, metric1, metric2)
    
    def create_scatter_analysis(self, stock_symbol: str, metric1: str, metric2: str) -> None:
        """Create scatter plot analysis for specific stock and metrics."""
        self.visualizer.plot_scatter_analysis(self.stock_data, stock_symbol, metric1, metric2)
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        direction = "Positive" if correlation >= 0 else "Negative"
        
        if abs_corr >= 0.8:
            strength = "Very Strong"
        elif abs_corr >= 0.6:
            strength = "Strong"
        elif abs_corr >= 0.4:
            strength = "Moderate"
        elif abs_corr >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        return f"{strength} {direction}"
    
    def interactive_analysis(self) -> None:
        """Interactive correlation analysis."""
        print("\n" + "="*60)
        print("         STOCKVISOR - CORRELATION ANALYSIS TOOL")
        print("="*60)
        print(f"Available metrics: {', '.join(self.correlation_analyzer.AVAILABLE_METRICS)}")
        print(f"Available stocks: {', '.join(self.stock_data.keys())}")
        print("\nCommands:")
        print("1. 'corr <metric1> <metric2>' - Analyze correlation between two metrics")
        print("2. 'matrix <stock>' - Show full correlation matrix for a stock")
        print("3. 'scatter <stock> <metric1> <metric2>' - Create scatter plot")
        print("4. 'volatility' - Show volatility analysis and charts")
        print("5. 'vol <stock>' - Show specific stock volatility details")
        print("6. 'quit' - Exit")
        print("="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command.startswith('corr '):
                    parts = command.split()
                    if len(parts) == 3:
                        self.visualize_metric_correlation(parts[1], parts[2])
                    else:
                        print("Usage: corr <metric1> <metric2>")
                
                elif command.startswith('matrix '):
                    parts = command.split()
                    if len(parts) == 2:
                        self.create_full_correlation_matrix(parts[1].upper())
                    else:
                        print("Usage: matrix <stock_symbol>")
                
                elif command.startswith('scatter '):
                    parts = command.split()
                    if len(parts) == 4:
                        self.create_scatter_analysis(parts[1].upper(), parts[2], parts[3])
                    else:
                        print("Usage: scatter <stock> <metric1> <metric2>")
                
                elif command == 'volatility':
                    self.analyze_volatility()
                    self.visualize_volatility()
                
                elif command.startswith('vol '):
                    parts = command.split()
                    if len(parts) == 2:
                        stock = parts[1].upper()
                        if stock in self.volatility_metrics:
                            print(f"\n{stock} Volatility Metrics:")
                            for metric, value in self.volatility_metrics[stock].items():
                                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
                        else:
                            print(f"No volatility data for {stock}")
                    else:
                        print("Usage: vol <stock_symbol>")
                
                else:
                    print("Invalid command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    def run_analysis(self) -> None:
        """Run the enhanced analysis pipeline."""
        logger.info("Starting Enhanced StockVisor Analysis...")
        
        # Load data
        self.load_stock_data()
        
        if not self.stock_data:
            logger.error("No data loaded. Exiting.")
            return
        
        # Example analyses
        print("\nRunning sample analyses...")
        
        # 1. Volatility analysis
        self.analyze_volatility()
        self.visualize_volatility()
        
        # 2. Classic volume-price analysis
        self.visualize_metric_correlation('volume', 'change_absolute')
        
        # 3. Volatility correlation analysis
        self.visualize_metric_correlation('rolling_volatility', 'volume')
        
        # 4. Full correlation matrix for first stock
        first_stock = list(self.stock_data.keys())[0]
        self.create_full_correlation_matrix(first_stock)
        
        # 5. Interactive mode
        self.interactive_analysis()


def main():
    """Main execution function."""
    try:
        stockvisor = StockVisor()
        stockvisor.run_analysis()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()







