# ==========================================
# 0. Environment Setup (Auto-Install)
# ==========================================
import subprocess
import sys


def install_packages():
    packages = ['yfinance', 'numpy', 'pandas', 'torch', 'matplotlib', 'seaborn', 'scipy']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install_packages()

# ==========================================
# 1. Main Simulation Code
# ==========================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yfinance as yf
from scipy.optimize import minimize

# --- Configuration ---
TICKERS = ['NVDA', 'AMD', 'TSM', 'ASML', 'MU']
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
LEARNING_RATE = 0.01
EPOCHS = 1500
TRAIN_RATIO = 0.8
INITIAL_CAPITAL = 10000

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid")


# --- Helper: Data Loading ---
def get_market_data():
    logging.info(f"Downloading real market data for: {TICKERS}...")
    try:
        df = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)

        # Handle yfinance MultiIndex columns
        if 'Adj Close' in df:
            df = df['Adj Close']
        elif 'Close' in df:
            df = df['Close']

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[TICKERS].dropna()
        if df.empty: raise ValueError("Empty DataFrame")
        logging.info(f"Loaded {len(df)} days of real data.")
        return df
    except Exception as e:
        logging.warning(f"Download failed ({e}). Generating synthetic fallback data.")
        return generate_fallback_data()


def generate_fallback_data(n_days=750):
    # Synthetic correlated random walk
    dates = pd.date_range(start=START_DATE, periods=n_days, freq='B')
    cov = np.eye(len(TICKERS)) * 0.5 + 0.5
    L = np.linalg.cholesky(cov)
    returns = np.random.normal(0, 1, (n_days, len(TICKERS))) @ L.T
    prices = 100 * np.exp(np.cumsum(returns * 0.02, axis=0))
    return pd.DataFrame(prices, columns=TICKERS, index=dates)


# --- AI Model: Physics Discovery Net ---
class PhysicsDiscoveryNet(nn.Module):
    def __init__(self, num_assets):
        super().__init__()
        self.n = num_assets
        # Learnable "Spring Constants" matrix (K)
        # We init with small random values, the AI will learn the real structure
        self.raw_k = nn.Parameter(torch.randn(self.n, self.n) * 0.05)

    def get_topology(self):
        # Enforce symmetry (K_ij = K_ji) and non-negativity
        K = (self.raw_k + self.raw_k.t()) / 2
        K = torch.nn.functional.softplus(K)
        # Remove self-loops (diagonal = 0)
        mask = 1 - torch.eye(self.n).to(K.device)
        return K * mask

    def forward(self, x_t):
        K = self.get_topology()
        # Build Laplacian L = D - A
        D = torch.diag(torch.sum(K, dim=1))
        L = D - K
        # Predict Force: F = -L * x
        return -torch.mm(x_t, L)


# --- Training Loop ---
def train_physics_model(df):
    # Prepare Data: Log prices detrended (deviation from moving average)
    log_prices = np.log(df)
    ma = log_prices.rolling(window=60).mean()
    x = (log_prices - ma).dropna()

    # Target: Next day's velocity (approx by diff)
    v = x.diff().shift(-1).dropna()
    x = x.iloc[:-1]

    # Convert to Tensors
    X_tensor = torch.tensor(x.values, dtype=torch.float32)
    V_target = torch.tensor(v.values, dtype=torch.float32)

    # Split
    split = int(len(X_tensor) * TRAIN_RATIO)
    X_train, V_train = X_tensor[:split], V_target[:split]

    # Init Model
    model = PhysicsDiscoveryNet(len(TICKERS))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    losses = []
    logging.info(f"Training AI on {len(X_train)} days of history...")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        v_pred = model(X_train)

        # Loss = Prediction Error + L1 Sparsity (Occam's Razor)
        mse = loss_fn(v_pred, V_train)
        l1_reg = 0.001 * torch.sum(torch.abs(model.get_topology()))
        loss = mse + l1_reg

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Return data needed for backtesting (The "Test" portion)
    test_start_idx = df.index.get_loc(x.index[split])
    test_df = df.iloc[test_start_idx:]
    return model, losses, test_df


# --- Pipeline ---
def run_pipeline():
    # 1. Load Data
    df = get_market_data()

    # 2. Train AI
    model, history, test_df = train_physics_model(df)

    # 3. Extract Learned Physics (The "Springs")
    learned_K = model.get_topology().detach().numpy()
    learned_L = np.diag(np.sum(learned_K, axis=1)) - learned_K

    # 4. Portfolio Optimization (Using Test Data Statistics)
    returns = test_df.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(TICKERS)

    def get_weights(use_physics=False):
        def obj(w):
            # Standard MPT Risk (Variance)
            risk = w.T @ sigma.values @ w
            if use_physics:
                # Add Physics Penalty: "Network Tension"
                # Penalizes holding assets that the AI learned are "tightly coupled"
                tension = w.T @ learned_L @ w
                return risk + 5.0 * tension
            return risk

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0.0, 0.45) for _ in range(n))  # Max 45% per stock
        res = minimize(obj, np.ones(n) / n, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x

    w_mpt = get_weights(use_physics=False)
    w_phy = get_weights(use_physics=True)

    # 5. Backtest Performance
    daily_rets_mpt = returns.dot(w_mpt)
    daily_rets_phy = returns.dot(w_phy)

    cum_mpt = INITIAL_CAPITAL * (1 + daily_rets_mpt).cumprod()
    cum_phy = INITIAL_CAPITAL * (1 + daily_rets_phy).cumprod()

    final_mpt = cum_mpt.iloc[-1]
    final_phy = cum_phy.iloc[-1]
    diff_pct = ((final_phy - final_mpt) / final_mpt) * 100

    # 6. Visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    # A. Weights
    ax1 = fig.add_subplot(gs[0, 0])
    pd.DataFrame({"Standard MPT": w_mpt, "AI-Physics": w_phy}, index=TICKERS).plot(kind='bar', ax=ax1,
                                                                                   color=['#95a5a6', '#2ecc71'])
    ax1.set_title("Portfolio Allocations")

    # B. Learned Springs
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(learned_K, annot=True, xticklabels=TICKERS, yticklabels=TICKERS, cmap="Greens", ax=ax2, fmt=".2f")
    ax2.set_title("Learned AI Market Structure")

    # C. Cumulative Returns (Comparison)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(cum_phy, label=f"AI-Physics Model (${final_phy:,.0f})", color='#2ecc71', linewidth=2.5)
    ax3.plot(cum_mpt, label=f"Standard MPT (${final_mpt:,.0f})", color='#95a5a6', linestyle='--', linewidth=2)
    ax3.fill_between(cum_phy.index, cum_mpt, cum_phy, where=(cum_phy > cum_mpt), color='#2ecc71', alpha=0.1)
    ax3.fill_between(cum_phy.index, cum_mpt, cum_phy, where=(cum_phy <= cum_mpt), color='red', alpha=0.1)
    ax3.set_title(f"Performance: AI vs MPT (Difference: {diff_pct:+.2f}%)", fontsize=14, fontweight='bold')
    ax3.legend()

    # D. Training Loss
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(history, color='#e74c3c')
    ax4.set_title("AI Training Loss")

    plt.tight_layout()
    plt.savefig("ai_vs_mpt_final.png")
    print(f"\n=== Final Results ===")
    print(f"MPT Final Value: ${final_mpt:,.2f}")
    print(f"AI  Final Value: ${final_phy:,.2f}")
    print(f"Improvement:     {diff_pct:+.2f}%")
    plt.show()


if __name__ == "__main__":
    run_pipeline()