# ⚛️ Spring-Mass Portfolio: AI-Physics Financial Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> "What if stocks were masses and correlations were springs?"

This repository implements a compact research/demo pipeline that treats a basket of assets as a coupled spring-mass system. A small PyTorch model (PhysicsDiscoveryNet) learns symmetric, non-negative pairwise "spring constants" between assets. Those learned couplings are then used as a penalty in portfolio optimization so we minimize both variance and "network tension."

Contents:
- A single executable Python script that fetches data, trains the physics-informed neural network, runs constrained portfolio optimization, backtests, and produces visualizations.
- Auto-install helper to install missing Python packages (convenient for demos — see caveats).

---

## 🚀 Key Features

- Physics-Informed AI: A PyTorch network that learns a market topology (interpretable as spring constants).
- Tension-Regularized Optimization: Penalizes allocations that concentrate risk across strongly coupled assets.
- Synthetic Fallback: If yfinance download fails, a correlated random-walk dataset is generated automatically.
- Backtest & Visualization: Compares cumulative wealth of Standard MPT vs AI-Physics portfolios and saves plots.

---

## 🛠️ Prerequisites

- Python 3.8+
- (Recommended) Virtual environment: venv or conda

The script will attempt to auto-install the following packages if they are not present:
- yfinance, numpy, pandas, torch, matplotlib, seaborn, scipy

For reproducibility it's recommended to install dependencies manually within a controlled environment.

---

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spring-mass-portfolio.git
cd spring-mass-portfolio
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows
```

3. Install dependencies (or allow the script to auto-install):
```bash
pip install yfinance numpy pandas torch matplotlib seaborn scipy
```

---

## 💻 Usage

Run the end-to-end pipeline (data -> training -> optimization -> backtest -> plots):

```bash
python spring_portfolio.py
```

Behavior:
- The script will attempt to pip-install missing packages.
- It will download market data for configured tickers via `yfinance`. If that fails, it generates synthetic correlated data.
- Training and backtesting run automatically; outputs are saved and printed.

---

## ⚙️ Configuration (edit at top of script)

- TICKERS — list of tickers (default: `['NVDA','AMD','TSM','ASML','MU']`)
- START_DATE / END_DATE — date range for market data
- LEARNING_RATE — optimizer learning rate
- EPOCHS — training epochs (default: 1500)
- TRAIN_RATIO — fraction of data used for training (default: 0.8)
- INITIAL_CAPITAL — starting capital for backtest

Example:
```python
TICKERS = ['AAPL','MSFT','GOOG']
START_DATE = '2022-01-01'
END_DATE = '2025-01-01'
```

---

## ⚙️ Execution Flow (high level)

1. Data fetching: downloads adjusted close prices using yfinance (handles MultiIndex outputs). If download fails or returns empty, a synthetic correlated random walk is used.
2. Preprocessing: log-prices are detrended by subtracting a 60-day moving average; targets are next-day velocities.
3. Model training: PhysicsDiscoveryNet learns a symmetric non-negative coupling matrix K via an L1-regularized MSE objective.
4. Extract topology: Convert learned K to Laplacian L = D - K.
5. Portfolio optimization: compute standard MPT weights (minimize variance) and physics-regularized weights (minimize variance + 5 * w^T L w) with constraints sum(w)=1 and 0 <= w_i <= 0.45.
6. Backtest: compute cumulative returns and plot comparisons.
7. Save outputs (image + stdout results).

---

## 🔬 The Physics (concise)

We approximate dynamics with overdamped spring-like coupling:
dx/dt ≈ -L x(t) + external + noise

- x: log-price deviations from moving average
- K: learned symmetric coupling ("spring constants")
- L: graph Laplacian built from K (L = D - K)
- The model predicts next-day velocity as F = -L x(t)

Sparsity (L1) on K encourages simpler topologies (Occam's razor). The optimization penalty w^T L w penalizes allocations that put weight on tightly-coupled groups.

---

## 📊 Results & Outputs

- ai_vs_mpt_final.png — combined visualization including:
  - Portfolio allocations (bar chart)
  - Learned coupling heatmap (K)
  - Cumulative returns (AI vs MPT) with shaded areas
  - Training loss curve
- STDOUT prints final MPT and AI portfolio values and percent improvement.

Interpreting the learned K:
- Larger K_ij implies stronger learned coupling (assets move together in the learned dynamics).
- The Laplacian-based penalty discourages placing high weight on clusters that are structurally coupled.

---

## ✅ Reproducibility & Randomness

- The script sets seeds for NumPy and PyTorch (42) but exact reproducibility can still vary depending on platform and library versions.
- For stricter reproducibility:
  - Pin package versions in requirements.txt
  - Use deterministic torch flags and single-thread BLAS, or run in Docker
  - Save and reload trained model parameters

---

## 🐛 Troubleshooting

- yfinance returns empty or raises errors:
  - Check network connectivity and ticker names. The script will fall back to synthetic data automatically.
- Long training times:
  - Reduce `EPOCHS` or use fewer tickers / shorter date ranges.
- Missing packages after auto-install:
  - Activate your environment and install dependencies manually:
    ```bash
    pip install yfinance numpy pandas torch matplotlib seaborn scipy
    ```
- Optimization or NaNs:
  - Ensure `returns` has no NaNs; drop or forward-fill missing values first.
  - Check that covariance matrix is positive semidefinite; adding a small diagonal jitter can help.

---

## 💡 Suggested Improvements

- Add CLI using argparse to override in-file configuration.
- Save model weights and learned K/L matrices to disk for inspection and reproducibility.
- Include transaction costs and rebalancing in the backtest for realism.
- Cross-validate the physics penalty coefficient and L1 regularization strength.
- Try time-series models (RNN, TCN) or ensemble approaches for improved predictive accuracy.
- Add a Jupyter notebook walkthrough to visualize intermediate steps interactively.

---

## 📂 Project Structure

```
spring-mass-portfolio/
│
├── spring_portfolio.py     # Main executable script (Data -> AI -> Opt -> Plot)
├── requirements.txt        # Optional: pinned dependencies
├── README.md               # This file
├── LICENSE                 # e.g., MIT
└── outputs/                # Generated plots and logs (ai_vs_mpt_final.png, etc.)
```

---

## 🤝 Contributing

Contributions are welcome! Suggested flow:
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/Awesome`
3. Commit and push your changes
4. Open a pull request describing your changes and rationale

Please include tests or notebooks demonstrating behavior when adding features or altering the model.

---

## 📜 License

This project is provided under the MIT License. See LICENSE for details.

---

## Acknowledgements

- yfinance for data access
- PyTorch for model training
- scipy.optimize for constrained optimization
- matplotlib / seaborn for plotting

---

If you'd like, I can:
- produce a requirements.txt with pinned versions,
- add an argparse-based CLI wrapper,
- create a short Jupyter notebook demonstrating step-by-step outputs,
- or save and attach the learned K and model weights after training.
