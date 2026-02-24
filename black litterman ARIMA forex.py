import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from statsmodels.regression.linear_model import OLS
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
OUTPUT_DIR = "backtest_results_arima_strict"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_ARIMA_BL.csv")

# --- Strategy Parameters ---
TARGET_BUDGET = 1000000.0   
LOOKBACK_COV = 126           # 6 Mois (Structure de risque plus réactive)
ARIMA_WINDOW = 60            # Fenêtre glissante pour l'entraînement AR(1)

# --- Transaction Costs (Spreads moyens + Slippage) ---
DEFAULT_COST = 0.0005 # Fallback (5 bps)

# --- BROKER REALISM ---
# Marge que le broker prend sur le différentiel de taux (0.50% annuel standard)
BROKER_SWAP_MARKUP = 0.0050 

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

# --- Black-Litterman Parameters ---
RISK_AVERSION_DELTA = 2.5    
TAU = 0.05                   
MAX_WEIGHT = 1.0             # Max 100% par devise (Risk ON)
GROSS_LEVERAGE = 5.0         # Levier total max x5 (Risk ON)

# --- Risk Analysis Parameters ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252        
MC_SIMULATIONS = 10000    

print("="*80)
print(f" 🚀 STARTING: ULTIMATE BL FX STRATEGY (ARIMA + AGGRESSIVE MODE)")
print(f"    LEVERAGE: {GROSS_LEVERAGE}x | MAX POS: {MAX_WEIGHT*100}% | REALISTIC COSTS: ON")
print("="*80)

# ==============================================================================
# 2. DATA LOADING (LOCAL CSV)
# ==============================================================================
print("\n[1] Loading Local Data...")

try:
    # Chargement
    prices = pd.read_csv('data/data_forex_prices.csv', index_col=0, parse_dates=True)
    rates_daily = pd.read_csv('data/data_fred_rates.csv', index_col=0, parse_dates=True)
    
    # Alignement des dates (Intersection)
    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx]
    rates_daily = rates_daily.loc[common_idx]
    
    # Filtrage par date de début du backtest (plus tôt pour l'historique ARIMA)
    prices = prices.loc["2015-01-01":] 
    rates_daily = rates_daily.loc["2015-01-01":]
    
    tickers = prices.columns.tolist()
    print(f"   ✅ Data Loaded: {len(tickers)} pairs, {len(prices)} days.")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: Data files not found in 'data/' folder.")
    exit()

# ==============================================================================
# 3. SIGNAL GENERATION (ARIMA ROLLING FORECAST + REALISTIC CARRY)
# ==============================================================================
print("[2] Generating Total Returns & ARIMA Signals (Realistic Carry applied)...")

returns_total = pd.DataFrame(index=prices.index, columns=tickers)
arima_signals = pd.DataFrame(index=prices.index, columns=tickers)

def parse_pair(ticker):
    clean = ticker.replace('=X', '')
    return clean[:3], clean[3:]

# 1. Calcul des Rendements Totaux (Prix + Carry Réaliste)
for t in tickers:
    base, quote = parse_pair(t)
    r_price = prices[t].pct_change()
    
    # Carry Return Réaliste (Differentiel journalier - Broker Markup)
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        raw_diff = rates_daily[base] - rates_daily[quote]
        realistic_diff = raw_diff - BROKER_SWAP_MARKUP
        r_carry = realistic_diff / 252.0
        
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# 2. ARIMA Rolling Loop (Proxy via AR1 vectorisé pour la vitesse)
print("   -> Computing Rolling AR(1) Forecasts...")

for t in tickers:
    y = returns_total[t]
    x = y.shift(1)
    
    cov_xy = y.rolling(ARIMA_WINDOW).cov(x)
    var_x = x.rolling(ARIMA_WINDOW).var()
    mean_x = x.rolling(ARIMA_WINDOW).mean()
    mean_y = y.rolling(ARIMA_WINDOW).mean()
    
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    
    forecast = alpha + beta * y
    arima_signals[t] = forecast.shift(1)

# On restreint le backtest à la période demandée
start_bt = "2018-01-01"
returns_total = returns_total.loc[start_bt:]
arima_signals = arima_signals.loc[start_bt:]
prices = prices.loc[start_bt:]

print(f"   ✅ Signals Ready. Backtest Period: {returns_total.index[0].date()} -> {returns_total.index[-1].date()}")

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    # 1. Prior (Markowitz)
    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # 2. Views (ARIMA)
    if curr_date not in arima_signals.index: return np.zeros(n_assets)
    raw_signal = arima_signals.loc[curr_date]
    
    # On trie par force du signal (absolue ou relative)
    ranked = raw_signal.sort_values(ascending=False)
    
    # Top 3 Long et Bottom 3 Short (adaptable à 4 comme dans le code de référence)
    views_assets = ranked.head(3).index.tolist() + ranked.tail(3).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        val = raw_signal[t]
        
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # Amplification du signal ARIMA pour la Vue BL
        annualized_signal = val * 252
        view_ret = np.tanh(annualized_signal * 2.0) * 0.30 
        Q.append(view_ret)
        
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P); Q = np.array(Q); Omega = np.diag(Omega_list)
    
    # 3. Posterior
    try:
        tau_sigma = TAU * sigma
        inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n_assets)*1e-6)
        inv_omega = np.linalg.inv(Omega)
        M_inverse = inv_tau_sigma + np.dot(np.dot(P.T, inv_omega), P)
        M = np.linalg.inv(M_inverse + np.eye(n_assets)*1e-6)
        term2 = np.dot(inv_tau_sigma, pi) + np.dot(np.dot(P.T, inv_omega), Q)
        bl_returns = np.dot(M, term2)
    except:
        return np.zeros(n_assets)

    # 4. Optimization
    def negative_utility(w):
        port_ret = np.dot(w, bl_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        return -(port_ret - (RISK_AVERSION_DELTA / 2) * (port_vol**2))

    cons = ({'type': 'ineq', 'fun': lambda x: GROSS_LEVERAGE - np.sum(np.abs(x))})
    bounds = tuple((-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n_assets))
    init_guess = np.sign(bl_returns) * (1/n_assets)
    
    try:
        res = minimize(negative_utility, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-6)
        return res.x
    except:
        return np.zeros(n_assets)

# ==============================================================================
# 5. BACKTEST ENGINE (W-WED REBAL + REALISTIC COSTS)
# ==============================================================================
print("\n[3] Executing Backtest (Strict Mode)...")

# Application du rebalancement le mercredi (W-WED) comme dans le modèle de référence
rebal_dates = returns_total.resample('W-WED').last().index
daily_dates = returns_total.index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

last_log_month = 0

# --- PRE-CALCUL DES FRAIS PAR PAIRE ---
costs_list = []
for t in tickers:
    clean_ticker = t.replace('=X', '') 
    cost_val = SPECIFIC_SPREADS.get(clean_ticker, DEFAULT_COST)
    costs_list.append(cost_val)
cost_vector = np.array(costs_list)
# -------------------------------------

print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY ($)':<12} | {'ALLOCATION (Effective for NEXT Period)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    
    # 1. PnL Journalier
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector)
    
    capital *= (1 + port_ret)
    
    # 2. Rebalancement (Mercredi)
    cost = 0.0
    if d in rebal_set:
        target_weights = get_black_litterman_weights(d)
        target_weights[np.abs(target_weights) < 0.02] = 0 
        
        # Frais de transaction
        turnover_vector = np.abs(target_weights - current_weights_daily)
        cost = np.sum(turnover_vector * cost_vector)
        capital -= (capital * cost)
        
        current_weights_daily = target_weights
        
        # LOGGING
        if d.month != last_log_month:
            pos_longs = []
            pos_shorts = []
            for idx, w in enumerate(target_weights):
                t_name = tickers[idx].replace('=X', '')
                if w > 0.05: pos_longs.append(f"{t_name}:{w:.0%}")
                elif w < -0.05: pos_shorts.append(f"{t_name}:{w:.0%}")
            
            l_str = ", ".join(pos_longs) if pos_longs else "None"
            s_str = ", ".join(pos_shorts) if pos_shorts else "None"
            
            print(f"{d.date()} | {capital:,.0f} | L: [{l_str}] | S: [{s_str}]")
            last_log_month = d.month

    # Stockage
    equity_curve.append(capital)
    equity_dates.append(d)
    net_ret = port_ret - (cost if d in rebal_set else 0.0)
    strategy_returns.append(net_ret)

# Export Data
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
    print(" [4] FULL RISK AUDIT & ANALYTICS")
    print("="*80)
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = df_ret['STRATEGY']
    
    # Metrics
    vol_ann = strat_ret.std() * np.sqrt(TRADING_DAYS)
    mean_ret_ann = strat_ret.mean() * TRADING_DAYS
    sharpe = mean_ret_ann / vol_ann if vol_ann > 0 else 0
    
    nav = (1 + strat_ret).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax()
    max_dd = dd.min()
    calmar = mean_ret_ann / abs(max_dd) if max_dd < 0 else 0
    
    var_95 = strat_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    print(f"\n--- PERFORMANCE (2018-End) ---")
    print(f"Annualized Return  : {mean_ret_ann:+.2%}")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Calmar Ratio       : {calmar:.2f}")
    
    print(f"\n--- RISK METRICS ---")
    print(f"Max Drawdown       : {max_dd:.2%}")
    print(f"Annual Volatility  : {vol_ann:.2%}")
    print(f"Daily VaR (95%)    : {var_95:.2%}")

    # Report CSV
    pd.DataFrame({
        'Metric': ['Ann. Return', 'Sharpe', 'Max DD', 'Volatility'],
        'Value': [mean_ret_ann, sharpe, max_dd, vol_ann]
    }).to_csv(os.path.join(OUTPUT_DIR, "RISK_REPORT.csv"))

# ==============================================================================
# 7. CHARTS
# ==============================================================================
def generate_charts():
    print("\n[5] Generating Charts...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    
    # 1. Equity & DD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df_equity.index, df_equity['Equity'], label='BL Strategy (ARIMA + Real Costs)', color='#1f77b4', linewidth=1.5)
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--', alpha=0.5)
    ax1.set_title("Strategy Performance: Black-Litterman ARIMA (Strict Mode)", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    ax2.plot(dd.index, dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_PERF_ARIMA_STRICT.png"), dpi=150)
    
    # 2. Scatter Risk/Return
    summary = pd.DataFrame()
    summary['Returns'] = returns.mean() * 252
    summary['Volatility'] = returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 8))
    subset = summary.drop('STRATEGY', errors='ignore')
    sns.scatterplot(data=subset, x='Volatility', y='Returns', s=100, color='gray', alpha=0.5, label='Forex Pairs')
    
    if 'STRATEGY' in summary.index:
        strat = summary.loc['STRATEGY']
        plt.scatter(strat['Volatility'], strat['Returns'], s=300, color='green', edgecolors='black', label='STRATEGY')
        plt.text(strat['Volatility']+0.003, strat['Returns'], "  BL STRATEGY", fontsize=11, fontweight='bold', color='green')
        
    for i in range(len(subset)):
        plt.text(subset.iloc[i]['Volatility'], subset.iloc[i]['Returns'], subset.index[i], fontsize=8, alpha=0.7)
        
    plt.axhline(0, color='black')
    plt.title('Risk/Return Profile (ARIMA Signal + Real Carry)', fontsize=14)
    plt.xlabel('Volatility')
    plt.ylabel('Total Return (Price + Carry)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_SCATTER_ARIMA_STRICT.png"), dpi=150)
    
    print(f"-> Charts saved to {OUTPUT_DIR}")
    plt.show()

if __name__ == "__main__":
    run_full_risk_analysis()
    generate_charts()