import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# Directory Management
OUTPUT_DIR = "backtest_results_markowitz"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_MVO.csv")

# --- Strategy Parameters ---
TARGET_BUDGET = 1000000.0   
LOOKBACK_MOMENTUM = 252      # 1 An (Signal)
LOOKBACK_COV = 126           # 6 Mois (Risque)

# --- Transaction Costs (Spreads moyens + Slippage) ---
# Valeurs estimées en valeur absolue (0.0001 = 1 pip/bps pour une paire à 1.0000)
# Majors: ~1-1.5 bps | Crosses: ~2-4 bps | Exotics: ~10+ bps
DEFAULT_COST = 0.0005 # Fallback pour les paires inconnues (5 bps)

SPECIFIC_SPREADS = {
    # MAJORS
    'EURUSD': 0.00012, 'USDJPY': 0.00012, 'GBPUSD': 0.00015, 
    'AUDUSD': 0.00015, 'USDCAD': 0.00018, 'USDCHF': 0.00018, 'NZDUSD': 0.00020,
    
    # CROSSES & MINORS
    'EURGBP': 0.00020, 'EURJPY': 0.00022, 'GBPJPY': 0.00028,
    'AUDJPY': 0.00025, 'CADJPY': 0.00025, 'CHFJPY': 0.00025,
    'EURAUD': 0.00030, 'EURCAD': 0.00030, 'GBPAUD': 0.00035, 'GBPCAD': 0.00035,
    'AUDNZD': 0.00035, 'AUDCAD': 0.00030,
    
    # EXOTICS / VOLATILE
    'USDMXN': 0.00100, 'USDZAR': 0.00120, 'USDTRY': 0.00200, 
    'USDSGD': 0.00030, 'USDCNH': 0.00040, 'USDSEK': 0.00040, 'USDNOK': 0.00040,
    'EURTRY': 0.00250, 'EURZAR': 0.00150
}

# --- Markowitz Parameters ---
RISK_AVERSION_DELTA = 2.5    
MAX_WEIGHT = 0.30            # Max 30% par devise
GROSS_LEVERAGE = 1.2         # Levier total max 120%

# --- Risk Analysis Parameters ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252        
MC_SIMULATIONS = 10000    

print("="*80)
print(f" 🚀 STARTING: MARKOWITZ MEAN-VARIANCE FX STRATEGY (DATA: FRED + YAHOO LOCAL)")
print("="*80)

# ==============================================================================
# 2. DATA LOADING (LOCAL CSV)
# ==============================================================================
print("\n[1] Loading Local Data...")

try:
    prices = pd.read_csv('data/data_forex_prices.csv', index_col=0, parse_dates=True)
    rates_daily = pd.read_csv('data/data_fred_rates.csv', index_col=0, parse_dates=True)
    
    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx]
    rates_daily = rates_daily.loc[common_idx]
    
    prices = prices.loc["2015-01-01":]
    rates_daily = rates_daily.loc["2015-01-01":]
    
    tickers = prices.columns.tolist()
    print(f"   ✅ Data Loaded: {len(tickers)} pairs, {len(prices)} days.")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: Data files not found in 'data/' folder.")
    exit()

# ==============================================================================
# 3. SIGNAL GENERATION (MOMENTUM + CARRY)
# ==============================================================================
print("[2] Calculating Total Returns (Price + Carry) & Expected Returns...")

returns_total = pd.DataFrame(index=prices.index, columns=tickers)
expected_returns = pd.DataFrame(index=prices.index, columns=tickers)

def parse_pair(ticker):
    clean = ticker.replace('=X', '')
    return clean[:3], clean[3:]

for t in tickers:
    base, quote = parse_pair(t)
    
    # 1. Total Return Calculation
    r_price = prices[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = (rates_daily[base] - rates_daily[quote]) / 252.0
    
    returns_total[t] = r_price + r_carry
    
    # 2. Expected Return Signal (Markowitz Input)
    # On utilise le Momentum (12 mois) annualisé comme proxy du rendement futur espéré
    # Markowitz a besoin d'un vecteur "Mu" (Expected Returns)
    # On lisse sur 5 jours pour éviter des changements brutaux
    log_rets = np.log(1 + returns_total[t])
    momentum_annualized = log_rets.rolling(LOOKBACK_MOMENTUM).mean() * 252
    expected_returns[t] = momentum_annualized.shift(1) # SHIFT ANTI LOOK-AHEAD

returns_total.dropna(inplace=True)
expected_returns.dropna(inplace=True)

# Découpage Backtest
start_bt = "2018-01-01"
returns_total = returns_total.loc[start_bt:]
expected_returns = expected_returns.loc[start_bt:]
prices = prices.loc[start_bt:]

print(f"   ✅ Signal Ready. Backtest Period: {returns_total.index[0].date()} -> {returns_total.index[-1].date()}")

# ==============================================================================
# 4. MARKOWITZ OPTIMIZER ENGINE
# ==============================================================================
def get_markowitz_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    # 1. Matrice de Covariance (Risque)
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)
    sigma = history.cov() * 252 
    
    # 2. Vecteur de Rendements Espérés (Mu)
    # Si pas de donnée pour ce jour, on sort
    if curr_date not in expected_returns.index: return np.zeros(n_assets)
    
    # On récupère le signal brut (Momentum annualisé)
    mu_raw = expected_returns.loc[curr_date].values
    
    # FILTRE "SMART" : On ne garde que les convictions fortes
    # Si le rendement espéré est faible (< 2% annuel), on le force à 0 pour éviter le bruit
    # Cela remplace la logique "Top 4 / Bottom 4" du BL
    mu_clean = np.where(np.abs(mu_raw) > 0.02, mu_raw, 0.0)

    # Si tout est à zéro (marché plat), on reste cash
    if np.all(mu_clean == 0): return np.zeros(n_assets)

    # 3. Optimisation Mean-Variance
    # Fonction Objectif : Maximiser Sharpe (Mu - Risque)
    def negative_utility(w):
        port_ret = np.dot(w, mu_clean)
        port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        # Formule Utility: E[R] - (lambda/2) * Var
        return -(port_ret - (RISK_AVERSION_DELTA / 2) * (port_vol**2))

    # Contraintes
    # Levier Total <= GROSS_LEVERAGE (120%)
    cons = ({'type': 'ineq', 'fun': lambda x: GROSS_LEVERAGE - np.sum(np.abs(x))})
    
    # Bornes par actif (-30% à +30%)
    bounds = tuple((-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n_assets))
    
    # Initial Guess (Equipondéré directionnel)
    init_guess = np.sign(mu_clean) * (GROSS_LEVERAGE / n_assets)
    
    try:
        res = minimize(negative_utility, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-6)
        return res.x
    except:
        return np.zeros(n_assets)

# ==============================================================================
# 5. BACKTEST ENGINE (STRICT MODE)
# ==============================================================================
print("\n[3] Executing Markowitz Backtest (Strict Mode)...")

rebal_dates = returns_total.resample('W-FRI').last().index
daily_dates = returns_total.index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

last_log_month = 0

# --- PRE-CALCUL DES FRAIS PAR PAIRE ---
# On construit un vecteur numpy aligné avec 'tickers' contenant le coût de chaque paire
costs_list = []
for t in tickers:
    clean_ticker = t.replace('=X', '') # Nettoyage si format Yahoo
    # On cherche dans la map, sinon on prend le default
    cost_val = SPECIFIC_SPREADS.get(clean_ticker, DEFAULT_COST)
    costs_list.append(cost_val)
cost_vector = np.array(costs_list)
# -------------------------------------

print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY ($)':<12} | {'ALLOCATION (Markowitz Optimized)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    
    # 1. PnL Calculation (Trade-On-Close logic)
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector)
    capital *= (1 + port_ret)
    
    # 2. Rebalancing Check
    cost = 0.0
    if d in rebal_set:
        target_weights = get_markowitz_weights(d)
        target_weights[np.abs(target_weights) < 0.02] = 0 
        
        # Calcul du turnover par actif
        turnover_vector = np.abs(target_weights - current_weights_daily)
        
        # Calcul des frais spécifiques : Somme(Turnover_i * Cost_i)
        cost = np.sum(turnover_vector * cost_vector)
        
        capital -= (capital * cost)
        
        current_weights_daily = target_weights
        
        # Logging
        if d.month != last_log_month:
            pos_longs = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w > 0.05]
            pos_shorts = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w < -0.05]
            l_str = ", ".join(pos_longs) if pos_longs else "None"
            s_str = ", ".join(pos_shorts) if pos_shorts else "None"
            print(f"{d.date()} | {capital:,.0f} | L: [{l_str}] | S: [{s_str}]")
            last_log_month = d.month

    equity_curve.append(capital)
    equity_dates.append(d)
    net_ret = port_ret - (cost if d in rebal_set else 0.0)
    strategy_returns.append(net_ret)

# Export
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)

print("-" * 100)
print(f"Backtest Complete. Final Capital: {capital:,.0f} $")

# ==============================================================================
# 6. RISK ANALYSIS
# ==============================================================================
def run_full_risk_analysis():
    print("\n" + "="*80)
    print(" [4] FULL RISK AUDIT (MARKOWITZ)")
    print("="*80)
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = df_ret['STRATEGY']
    
    vol_ann = strat_ret.std() * np.sqrt(TRADING_DAYS)
    mean_ret_ann = strat_ret.mean() * TRADING_DAYS
    sharpe = mean_ret_ann / vol_ann if vol_ann > 0 else 0
    
    nav = (1 + strat_ret).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax()
    max_dd = dd.min()
    calmar = mean_ret_ann / abs(max_dd) if max_dd < 0 else 0
    var_95 = strat_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    print(f"\n--- PERFORMANCE ---")
    print(f"Annualized Return  : {mean_ret_ann:+.2%}")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    
    print(f"\n--- RISK ---")
    print(f"Max Drawdown       : {max_dd:.2%}")
    print(f"Annual Volatility  : {vol_ann:.2%}")
    
    # Save Report
    pd.DataFrame({
        'Metric': ['Return', 'Sharpe', 'DD', 'Vol'],
        'Value': [mean_ret_ann, sharpe, max_dd, vol_ann]
    }).to_csv(os.path.join(OUTPUT_DIR, "RISK_REPORT_MVO.csv"))

# ==============================================================================
# 7. CHARTS
# ==============================================================================
def generate_charts():
    print("\n[5] Generating Charts...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    
    # Equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(df_equity.index, df_equity['Equity'], label='Markowitz Strategy', color='darkorange', linewidth=1.5)
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Strategy Performance: Markowitz Mean-Variance", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    ax2.plot(dd.index, dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_PERF_MVO.png"), dpi=150)
    
    # Scatter
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 8))
    subset = summary.drop('STRATEGY', errors='ignore')
    sns.scatterplot(data=subset, x='Volatility', y='Returns', s=100, color='gray', alpha=0.5)
    
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=300, color='darkorange', edgecolors='black', label='STRATEGY')
        plt.text(strat['Volatility']+0.003, strat['Returns'], "  MARKOWITZ", fontsize=11, fontweight='bold', color='darkorange')
    
    plt.title('Risk/Return Profile (Markowitz)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_SCATTER_MVO.png"), dpi=150)
    
    print(f"-> Charts saved to {OUTPUT_DIR}")
    plt.show()

if __name__ == "__main__":
    run_full_risk_analysis()
    generate_charts()