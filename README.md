# 🚀 StockVisor - Financial Quantitative Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn"/>
</p>

## 💡 **The Ultimate Financial Analytics Powerhouse!**

**StockVisor** is a blazing-fast, modular financial quantitative analytics platform that transforms raw stock data into actionable insights! Built with cutting-edge Python technologies, it's designed to compute complex financial metrics, correlations, and volatility measures with **unprecedented flexibility** and **professional-grade visualizations**.

---

## 🎯 **What Makes StockVisor Revolutionary?**

### 🔥 **Flexible Correlation Analysis Engine**
Unlike traditional tools that only analyze volume-price relationships, StockVisor lets you explore correlations between **ANY** stock metrics:
- **Open vs Close prices** 📈
- **Volume vs Volatility** 📊
- **High vs Low ranges** 📉
- **Daily returns vs Price ranges** 💹
- **Rolling volatility vs Moving averages** 🌊
- **And literally ANY combination you can imagine!** 🧠

### ⚡ **Blazing Performance with NumPy Vectorization**
- **Vectorized operations** for lightning-fast computations on massive datasets
- **Memory-efficient** data processing pipelines
- **Optimized algorithms** for real-time analysis of thousands of data points

### 📊 **Professional-Grade Interactive Visualizations**
- **Correlation heatmaps** with stunning color gradients
- **Interactive scatter plots** with trend lines and time progression
- **Risk-categorized volatility charts** with intelligent color coding
- **Hover annotations** revealing detailed metrics on-demand

---

## 🛠️ **Technologies & Advanced Techniques**

### **Core Technologies Stack**
```python
🐍 Python 3.8+          # Modern, type-hinted code
📊 NumPy                 # Vectorized mathematical operations
🐼 Pandas                # High-performance data manipulation
📈 Matplotlib            # Publication-quality plots
🌈 Seaborn               # Statistical data visualization
📅 DateTime/DateUtil     # Advanced time series handling
🎯 Type Hints            # Professional code quality
```

### **Advanced Programming Patterns**
- **🏗️ Modular Architecture**: Clean separation of concerns with specialized classes
- **📦 Dataclasses**: Type-safe data structures for financial metrics
- **🔧 Static Methods**: Optimized utility functions for maximum performance
- **🎨 Factory Pattern**: Flexible metric calculation and analysis
- **📝 Comprehensive Logging**: Professional debugging and monitoring
- **🔄 Pipeline Architecture**: Streamlined data processing workflows

### **Financial Engineering Techniques**
- **📈 Historical Volatility**: Annualized standard deviation calculations
- **🎲 Value at Risk (VaR)**: 1% and 5% confidence level risk metrics
- **⚖️ Sharpe Ratio**: Risk-adjusted return analysis
- **🌊 Rolling Volatility**: Dynamic 30-day volatility windows
- **📊 Moving Averages**: 10-day volume and price trend indicators
- **🔗 Correlation Matrices**: Multi-dimensional relationship analysis

---

## 🚀 **Quick Start Guide**

### **Installation**
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn python-dateutil mplcursors

# Clone and run
git clone <your-repo>
cd stockvisor
python xd.py
```

### **File Format**
Place your CSV files with the naming pattern: `NASDAQ_STOCK_SYMBOL.csv`
```
NASDAQ_STOCK_NVDA.csv
NASDAQ_STOCK_AAPL.csv
NASDAQ_STOCK_TSLA.csv
```

### **Required CSV Columns**
```csv
Date,Open,High,Low,Close,Adj Close,Volume
2023-01-01,100.50,102.75,99.25,101.20,101.20,1500000
```

---

## 🎮 **Interactive Commands**

Launch the interactive analysis mode and unleash the power:

```bash
# Correlation Analysis
corr open_price close_price        # Analyze open vs close correlation
corr volume rolling_volatility     # Volume vs volatility relationship
corr high_price change_absolute    # High prices vs daily changes

# Advanced Visualizations
matrix NVDA                        # Complete correlation heatmap
scatter NVDA volume price_range    # Interactive scatter analysis
volatility                         # Comprehensive risk analysis

# Individual Stock Deep Dive
vol NVDA                          # Detailed volatility metrics
```

---

## 📊 **Available Metrics Universe**

StockVisor analyzes **14 comprehensive metrics**:

| **Price Metrics** | **Volume Metrics** | **Derived Metrics** | **Risk Metrics** |
|-------------------|--------------------|--------------------|------------------|
| `open_price` | `volume` | `change_percentage` | `volatility` |
| `close_price` | `volume_ma` | `change_absolute` | `rolling_volatility` |
| `high_price` | | `daily_return` | |
| `low_price` | | `price_range` | |
| `adjusted_close` | | `price_ma` | |

---

## 🧮 **Mathematical Powerhouse**

### **Correlation Coefficient Calculation**
```python
correlation = np.corrcoef(metric1_values, metric2_values)[0, 1]
```

### **Annualized Volatility Formula**
```python
volatility = np.std(daily_returns) * np.sqrt(252)  # 252 trading days
```

### **Sharpe Ratio Computation**
```python
sharpe_ratio = (avg_excess_return / volatility) * np.sqrt(252)
```

### **Value at Risk (VaR)**
```python
var_5_percent = np.percentile(returns, 5.0)  # 5% worst-case scenario
```

---

## 🎨 **Visualization Showcase**

### **1. Correlation Heatmap**
- **Upper triangular masking** for clean presentation
- **Color-coded correlation strength** (Red = negative, Blue = positive)
- **Precise 3-decimal annotation** for exact values

### **2. Interactive Scatter Plots**
- **Time-progression color mapping** showing data evolution
- **Polynomial trend lines** for relationship visualization
- **Real-time correlation calculation** and display

### **3. Risk-Categorized Charts**
- **Green**: Low risk (Volatility < 20%, Sharpe > 1)
- **Orange**: Medium risk (Moderate metrics)
- **Red**: High risk (Volatility > 30%, Sharpe < 0)

---

## 🎯 **Professional Features**

### **🔍 Smart Error Handling**
```python
# Robust data validation
try:
    return float(value)
except (ValueError, TypeError):
    logger.error(f"Invalid format: {value}")
    return 0.0
```

### **⚡ Memory Optimization**
```python
# Efficient data alignment
min_length = min(len(values1), len(values2))
correlation = np.corrcoef(values1[:min_length], values2[:min_length])
```

### **📊 Type Safety**
```python
@dataclass
class StockMetrics:
    open_price: float
    volume: int
    daily_return: Optional[float] = None
```

---

## 🚀 **Example Analysis Results**

```
NVDA Volatility Metrics:
  Historical Volatility (Annual): 0.3245
  Sharpe Ratio: 1.2850
  VaR (5%): -0.0245
  Max Daily Return: 0.1250
  Average Daily Return: 0.0012

Correlation Analysis: volume vs change_absolute
NVDA:   0.2340 (Weak Positive)
```

---

## 🎓 **Perfect For**

- **📚 Financial Students**: Learn quantitative analysis techniques
- **💼 Investment Professionals**: Advanced risk assessment tools
- **🔬 Researchers**: Flexible correlation analysis platform
- **💻 Developers**: Clean, extensible financial analytics codebase
- **📊 Data Scientists**: Professional visualization and statistical tools

---

## 🌟 **Why StockVisor Rocks**

✅ **Unlimited Flexibility**: Analyze ANY metric combination  
✅ **Lightning Fast**: NumPy-powered vectorized operations  
✅ **Publication Ready**: Professional-grade visualizations  
✅ **Interactive**: Real-time analysis and exploration  
✅ **Extensible**: Modular design for easy customization  
✅ **Production Ready**: Comprehensive error handling and logging  

---

## 🔮 **Future Enhancements**

- **🌐 Real-time data feeds** integration
- **🤖 Machine learning** predictive models
- **📱 Web dashboard** interface
- **🔄 Automated reporting** generation
- **📊 Advanced technical indicators**

---

## 📞 **Get Started Today!**

Dive into the world of professional financial analytics with StockVisor! Your journey to mastering quantitative finance starts here! 🚀

```bash
python xd.py
# Enter command: corr open_price close_price
# Watch the magic happen! ✨
```

---

*Built with ❤️ and lots of ☕ by passionate developers for the financial community*
