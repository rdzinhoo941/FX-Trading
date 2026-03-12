import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
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
OUTPUT_DIR = "backtest_results_fred"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_BL.csv")

# --- Strategy Parameters ---
TARGET_BUDGET = 1000000.0   
LOOKBACK_MOMENTUM = 252      # 1 Year (Robust signal)
LOOKBACK_COV = 126           # 6 Months (More reactive risk structure)

# --- Transaction Costs (Average Spreads + Slippage) ---
DEFAULT_COST = 0.0005 # Fallback (5 bps)

# --- BROKER REALISM ---
# Broker margin taken on the interest rate differential (Standard 0.50% annual)
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

# --- Black-Litterman Base Parameters ---
RISK_AVERSION_DELTA = 2.5    
# Note : La variable TAU est maintenant définie dynamiquement dans le profil de risque.

# ==============================================================================
# 2. USER PROFILE SELECTION
# ==============================================================================
print("="*80)
print(" 🌍 SELECT YOUR RISK PROFILE")
print("="*80)
print("  [1] Conservative — Weight=5%,  Lev=1.0x, Tau=0.02")
print("  [2] Balanced     — Weight=12.5%, Lev=2.0x, Tau=0.05")
print("  [3] Aggressive   — Weight=20%,Lev=5.0x, Tau=0.20")
print("  [4] Custom       — Set Weight & Leverage manually, auto-scale Tau (0-10)")

choice = input("\nEnter your choice (1/2/3/4): ").strip()

if choice == '1':
    MAX_WEIGHT = 0.05
    GROSS_LEVERAGE = 1.0
    TAU = 0.02
    profile_name = "CONSERVATIVE"
elif choice == '2':
    MAX_WEIGHT = 0.25
    GROSS_LEVERAGE = 2.0
    TAU = 0.05
    profile_name = "BALANCED"
elif choice == '3':
    MAX_WEIGHT = 1.0
    GROSS_LEVERAGE = 5.0
    TAU = 0.20
    profile_name = "AGGRESSIVE"
elif choice == '4':
    profile_name = "CUSTOM"
    try:
        # 1. User defines the total portfolio leverage
        GROSS_LEVERAGE = float(input("Enter max leverage (e.g., 5.0): "))
        
        # 2. User defines the max weight of an asset relative to this total leverage
        user_weight_pct = float(input("Enter max weight as a % of TOTAL portfolio (e.g., 0.20 for 20%): "))
        
        # CALCULATION: Convert to a percentage of the base capital for the optimizer
        MAX_WEIGHT = user_weight_pct * GROSS_LEVERAGE
        
        # 3. User provides a score from 0 to 10 ONLY for the Tau conviction level
        tau_score = float(input("Enter your conviction level for TAU (0 to 10, 10=Max Risk): "))
        tau_score = max(0.0, min(10.0, tau_score)) # Safeguard to keep the score between 0 and 10
        
        # Automatic Tau calculation
        TAU = 0.01 + (tau_score / 10.0) * 0.19 # Scales from 0.01 to 0.20
        
        print(f"\n -> Mapped Settings: EFFECTIVE MAX_WEIGHT={MAX_WEIGHT:.0%} (of base capital), LEVERAGE={GROSS_LEVERAGE:.1f}x, TAU={TAU:.3f}")
    except ValueError:
        print("Invalid input. Defaulting to Balanced profile.")
        MAX_WEIGHT = 0.25
        GROSS_LEVERAGE = 2.0
        TAU = 0.05
        profile_name = "BALANCED (Fallback)"
else:
    print("Invalid choice. Defaulting to Balanced profile.")
    MAX_WEIGHT = 0.25
    GROSS_LEVERAGE = 2.0
    TAU = 0.05
    profile_name = "BALANCED (Fallback)"

# --- Risk Analysis Parameters ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252        

print("\n" + "="*80)
print(f" 🚀 STARTING: ULTIMATE BL FX STRATEGY ({profile_name} MODE)")
print(f"    LEVERAGE: {GROSS_LEVERAGE:.1f}x | MAX POS: {MAX_WEIGHT*100:.0f}% | TAU: {TAU:.3f} | REALISTIC COSTS: ON")
print("="*80)

# ==============================================================================
# 3. DATA LOADING (LOCAL CSV)
# ==============================================================================
print("\n[1] Loading Local Data...")

try:
    # Loading
    prices = pd.read_csv('data/data_forex_prices.csv', index_col=0, parse_dates=True)
    rates_daily = pd.read_csv('data/data_fred_rates.csv', index_col=0, parse_dates=True)
    
    # Date alignment (Intersection)
    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx]
    rates_daily = rates_daily.loc[common_idx]
    
    # Filtering by backtest start date
    prices = prices.loc["2015-01-01":] 
    rates_daily = rates_daily.loc["2015-01-01":]
    
    tickers = prices.columns.tolist()
    print(f"   ✅ Data Loaded: {len(tickers)} pairs, {len(prices)} days.")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: Data files not found in 'data/' folder.")
    exit()

# ==============================================================================
# 4. SIGNAL GENERATION (MOMENTUM + REALISTIC CARRY)
# ==============================================================================
print("[2] Calculating Total Returns (Price + Carry - Broker Markup)...")

returns_total = pd.DataFrame(index=prices.index, columns=tickers)
momentum_score = pd.DataFrame(index=prices.index, columns=tickers)

def parse_pair(ticker):
    clean = ticker.replace('=X', '')
    return clean[:3], clean[3:]

for t in tickers:
    base, quote = parse_pair(t)
    
    # 1. Price Return
    r_price = prices[t].pct_change()
    
    # 2. Realistic Carry Return (Daily Interest Differential - Broker Markup)
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        # Raw differential
        raw_diff = rates_daily[base] - rates_daily[quote]
        
        # Applying Broker Markup (Yield haircut)
        # Subtract broker margin from raw differential
        realistic_diff = raw_diff - BROKER_SWAP_MARKUP
        
        # Daily conversion
        r_carry = realistic_diff / 252.0
    
    # Total Return
    returns_total[t] = r_price + r_carry
    
    # 3. Momentum Signal (12-Month Total Return)
    log_rets = np.log(1 + returns_total[t])
    momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

returns_total.dropna(inplace=True)
momentum_score.dropna(inplace=True)

# Restrict backtest to requested period
start_bt = "2018-01-01"
returns_total = returns_total.loc[start_bt:]
momentum_score = momentum_score.loc[start_bt:]
prices = prices.loc[start_bt:]

print(f"   ✅ Signal Ready. Backtest Period: {returns_total.index[0].date()} -> {returns_total.index[-1].date()}")

# ==============================================================================
# 5. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    # Historical window for Covariance
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    # 1. Prior (Markowitz)
    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # 2. Views (Momentum)
    if curr_date not in momentum_score.index: return np.zeros(n_assets)
    curr_mom = momentum_score.loc[curr_date]
    
    # Select pairs with strongest/weakest momentum
    ranked = curr_mom.sort_values(ascending=False)
    # Top 4 (Long) and Bottom 4 (Short)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        mom_val = curr_mom[t]
        
        # Pick Matrix (P)
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # View Return Vector (Q)
        # If strong Momentum -> Expect +20% annualized
        view_ret = np.tanh(mom_val) * 0.20 
        Q.append(view_ret)
        
        # Uncertainty Matrix (Omega)
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P); Q = np.array(Q); Omega = np.diag(Omega_list)
    
    # 3. Posterior (Black-Litterman Master Formula)
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
# 6. BACKTEST ENGINE (QUARTERLY REBALANCING)
# ==============================================================================
print("\n[3] Executing Backtest (Quarterly Rebalancing)...")

# 1. DEFINE REBALANCING DATES (Each Quarter)
# Group by Year AND Quarter to find the 1st trading day of each quarter
rebal_dates = returns_total.groupby([returns_total.index.year, returns_total.index.quarter]).apply(lambda x: x.index[0]).tolist()

daily_dates = returns_total.index
rebal_set = set(rebal_dates)

print(f" -> Rebalancing Dates ({len(rebal_dates)} periods): {[d.date() for d in rebal_dates]}")

# 2. BOOTSTRAP (FORCED INITIALIZATION AT DAY 0)
print(" -> Initializing portfolio weights at Start Date...")
# Force calculation for the very first day
current_weights_daily = get_black_litterman_weights(daily_dates[0])

# Prepare costs
costs_list = []
for t in tickers:
    clean_ticker = t.replace('=X', '')
    cost_val = SPECIFIC_SPREADS.get(clean_ticker, DEFAULT_COST)
    costs_list.append(cost_val)
cost_vector = np.array(costs_list)

# Initial fees
initial_cost = np.sum(np.abs(current_weights_daily) * cost_vector)
capital = TARGET_BUDGET * (1 - initial_cost)

equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY ($)':<12} | {'ALLOCATION (Effective for NEXT Period)'}")
print("-" * 100)

# Display initial position
pos_longs = [f"{tickers[i].replace('=X','')}:{w:.0%}" for i, w in enumerate(current_weights_daily) if w > 0.05]
pos_shorts = [f"{tickers[i].replace('=X','')}:{w:.0%}" for i, w in enumerate(current_weights_daily) if w < -0.05]
print(f"{daily_dates[0].date()} | {capital:,.0f} | L: {pos_longs} | S: {pos_shorts}")

# 3. BACKTEST LOOP (Starts at Day+1)
for i, d in enumerate(daily_dates[1:]):
    
    # A. Daily PnL
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector)
    
    capital *= (1 + port_ret)
    
    # B. Rebalancing (If start of a new quarter)
    cost = 0.0
    if d in rebal_set:
        print(f" -> Rebalancing triggered on {d.date()}") # Debug info
        target_weights = get_black_litterman_weights(d)
        
        target_weights[np.abs(target_weights) < 0.02] = 0 
        
        turnover_vector = np.abs(target_weights - current_weights_daily)
        cost = np.sum(turnover_vector * cost_vector)
        
        capital -= (capital * cost)
        current_weights_daily = target_weights
        
        # LOGGING
        pos_longs = []
        pos_shorts = []
        for idx, w in enumerate(target_weights):
            t_name = tickers[idx].replace('=X', '')
            if w > 0.05: pos_longs.append(f"{t_name}:{w:.0%}")
            elif w < -0.05: pos_shorts.append(f"{t_name}:{w:.0%}")
        
        l_str = ", ".join(pos_longs) if pos_longs else "None"
        s_str = ", ".join(pos_shorts) if pos_shorts else "None"
        print(f"{d.date()} | {capital:,.0f} | L: [{l_str}] | S: [{s_str}]")

    # Storage
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
# 7. RISK ANALYSIS
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
# 8. CHARTS
# ==============================================================================
def generate_charts():
    print("\n[5] Generating Charts...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    
    # 1. Equity & DD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df_equity.index, df_equity['Equity'], label='BL Strategy (Real Rates)', color='#1f77b4', linewidth=1.5)
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"Strategy Performance: Black-Litterman ({profile_name} Profile)", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    ax2.plot(dd.index, dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_PERF_FRED.png"), dpi=150)
    
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
    plt.title('Risk/Return Profile (Including Real Carry)', fontsize=14)
    plt.xlabel('Volatility')
    plt.ylabel('Total Return (Price + Carry)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_SCATTER_FRED.png"), dpi=150)
    
    print(f"-> Charts saved to {OUTPUT_DIR}")
    plt.show()

# ==============================================================================
# 9. LIVE ALLOCATION (CURRENT DAY)
# ==============================================================================
def show_current_allocation():
    print("\n" + "="*80)
    print(" [6] LIVE ALLOCATION (WHAT TO TRADE TODAY)")
    print("="*80)

    # Récupérer la toute dernière date du dataset
    last_date = returns_total.index[-1]
    print(f" -> Calculation Date: {last_date.date()}")
    print(f" -> Account Value Used: {capital:,.0f} $ (from backtest end state)")

    # Calculer les poids pour cette date précise
    current_weights = get_black_litterman_weights(last_date)
    
    # Nettoyer les poids insignifiants (< 2%)
    current_weights[np.abs(current_weights) < 0.02] = 0 

    pos_longs = []
    pos_shorts = []
    
    for idx, w in enumerate(current_weights):
        if w == 0: continue
        
        t_name = tickers[idx].replace('=X', '')
        # Valeur en dollars de la position basée sur le capital de fin de backtest
        dollar_val = w * capital 
        
        if w > 0:
            pos_longs.append({'Pair': t_name, 'Weight': w, 'Position ($)': dollar_val})
        elif w < 0:
            pos_shorts.append({'Pair': t_name, 'Weight': w, 'Position ($)': dollar_val})

    print("\n 🟢 LONG POSITIONS (BUY):")
    if not pos_longs:
        print("    None")
    else:
        # Trier par poids absolu décroissant
        pos_longs.sort(key=lambda x: x['Weight'], reverse=True)
        for p in pos_longs:
            print(f"    - {p['Pair']:<6} : {p['Weight']:>6.1%} | {p['Position ($)']:>10,.0f} $")

    print("\n 🔴 SHORT POSITIONS (SELL):")
    if not pos_shorts:
        print("    None")
    else:
        # Trier par poids absolu croissant (les plus négatifs d'abord)
        pos_shorts.sort(key=lambda x: x['Weight']) 
        for p in pos_shorts:
            print(f"    - {p['Pair']:<6} : {p['Weight']:>6.1%} | {p['Position ($)']:>10,.0f} $")

    print("\n" + "-"*80)
    print(" -> Note: Apply these weights to your real broker account.")
    print(" -> Next Rebalancing: In exactly 3 months.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_full_risk_analysis()
    show_current_allocation()
    generate_charts()