# Masters-Project
# Scientific Portfolio Optimization: A Risk-Adjusted Approach

This repository contains the research artefact for my MSc Data Science dissertation:  
**“Scientific Portfolio Optimization: A Risk-Adjusted Approach to Asset Allocation.”**

The project evaluates classical and machine-learning–enhanced portfolio optimization strategies under rolling, out-of-sample backtesting. The focus is not on profitability but on **scientific evaluation of robustness and risk-adjusted returns**, aligning with academic research standards.

---

## 📂 Repository Structure

- **`mean_variance_optimization.py`** – Classical Mean-Variance Optimization (Markowitz, 1952) with constraints.  
- **`machine_learning_strategies.py`** – Feature engineering + ML models (RF, GBM, XGBoost) with walk-forward validation.
- **`hrp.py`** – Hierarchical Risk Parity allocation (López de Prado, 2016/2020).  
- **`rp.py`** – Risk Parity (inverse volatility weights).  
- **`rebalancing.py`** – Rolling-window backtests (monthly and deviation-based rebalancing).
- **`portfolio_statistics.py`** – Performance metrics (Sharpe, Sortino, Max Drawdown, Volatility, Turnover).  


---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/naifawad/Masters-Project.git
cd Masters-Project
pip install -r requirements.txt
