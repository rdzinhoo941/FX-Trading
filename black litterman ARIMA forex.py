import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
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
OUTPUT_DIR = "backtest_results_arima_bl"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_ARIMA.csv")

# --- Strategy Parameters ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005    # 5 bps
LOOKBACK_COV = 126           # 6 Mois pour la matrice de risque
ARIMA_ORDER = (1, 0, 0)      # Modèle AR(1) simple pour la rapidité et éviter l'overfitting
ARIMA_WINDOW = 60            # Fenêtre glissante pour l'entraînement (3 mois)

# --- Black-Litterman Parameters ---
RISK_AVERSION_DELTA = 2.5    
TAU = 0.05                   
MAX_WEIGHT = 0.4            # On autorise un peu plus de concentration
GROSS_LEVERAGE = 1.5         # Levier 1.5x
# note : ici, limiter à 0.4 et 1.5 le total revient en théorie à limiter à 1.2 les poids si on limite le levier total à 4.5, 
# pour autant, on n'a pas forcément 3* plus ou 3* moins de bénéfice à la fin, ce qui rend l'équation assez complexe.  
# ici, on fait par exemple, pour 0.4 et 1.5, +5.6 %, contre -4% pour 1.2 et 4.5
SIGNAL_AMPLIFIER = 100.0     # ARIMA sort des petits chiffres (ex: 0.0001), on booste pour la Vue

# --- Risk Analysis Parameters ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252        

print("="*80)
print(f" 🚀 STARTING: BLACK-LITTERMAN FX STRATEGY POWERED BY ARIMA")
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
    
    # On commence plus tôt pour avoir assez d'historique pour le premier ARIMA
    prices = prices.loc["2017-01-01":]
    rates_daily = rates_daily.loc["2017-01-01":]
    
    tickers = prices.columns.tolist()
    print(f"   ✅ Data Loaded: {len(tickers)} pairs, {len(prices)} days.")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: Data files not found.")
    exit()

# ==============================================================================
# 3. SIGNAL GENERATION (ARIMA ROLLING FORECAST)
# ==============================================================================
print("[2] Generating ARIMA Signals (This may take a few minutes)...")

returns_total = pd.DataFrame(index=prices.index, columns=tickers)
arima_signals = pd.DataFrame(index=prices.index, columns=tickers)

def parse_pair(ticker):
    clean = ticker.replace('=X', '')
    return clean[:3], clean[3:]

# 1. Calcul des Rendements Totaux (Prix + Carry)
for t in tickers:
    base, quote = parse_pair(t)
    r_price = prices[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = (rates_daily[base] - rates_daily[quote]) / 252.0
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# 2. ARIMA Rolling Loop
# Pour gagner du temps, on ne recalcule pas ARIMA chaque jour, mais on utilise
# une logique simplifiée ou on le fait uniquement sur les dates de rebalancement (Vendredi)
# Mais pour être précis, voici la version optimisée vecteur :
# On va entraîner un modèle sur une fenêtre glissante.

# Note : Pour que ce script ne prenne pas 10 heures, on va tricher intelligemment.
# On va utiliser une régression auto-régressive simple (AR1) via OLS qui est 1000x plus rapide que ARIMA().fit()
# et donne le même résultat pour un ARIMA(1,0,0).

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

print("   -> Computing Rolling AR(1) Forecasts...")

for t in tickers:
    y = returns_total[t]
    # Lagged returns (Predictor)
    x = y.shift(1)
    
    # Rolling regression (Window = ARIMA_WINDOW)
    # Formule : R_t = alpha + beta * R_{t-1} + epsilon
    
    # On utilise Rolling de Pandas pour calculer Covariance et Variance glissantes
    cov_xy = y.rolling(ARIMA_WINDOW).cov(x)
    var_x = x.rolling(ARIMA_WINDOW).var()
    mean_x = x.rolling(ARIMA_WINDOW).mean()
    mean_y = y.rolling(ARIMA_WINDOW).mean()
    
    # Beta = Cov(x,y) / Var(x)
    beta = cov_xy / var_x
    # Alpha = Mean(y) - Beta * Mean(x)
    alpha = mean_y - beta * mean_x
    
    # Forecast = Alpha + Beta * Last_Return
    # On shift(1) les params car on utilise les params calculés hier pour prédire aujourd'hui
    # Mais attention au Look-Ahead !
    # Le Rolling calcule la stat à la date T en incluant T.
    # Donc pour prédire T+1, on doit utiliser les stats à T.
    # Donc forecast_T+1 = alpha_T + beta_T * return_T
    
    forecast = alpha + beta * y
    
    # On stocke le résultat SHIFTÉ de 1 jour.
    # Valeur à l'index T = Prédiction faite le soir de T-1 pour la journée T.
    arima_signals[t] = forecast.shift(1)

# Nettoyage et période de Backtest
start_bt = "2018-01-01"
returns_total = returns_total.loc[start_bt:]
arima_signals = arima_signals.loc[start_bt:]

print(f"   ✅ ARIMA Signals Ready (Rolling AR1 Proxy).")

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_black_litterman_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # --- ARIMA VIEWS ---
    if curr_date not in arima_signals.index: return np.zeros(n_assets)
    
    # Signal Brut (Rendement espéré journalier)
    raw_signal = arima_signals.loc[curr_date]
    
    # On filtre : On ne prend position que si le signal est "fort" (relatif)
    # On trie par force du signal (absolue ou relative)
    ranked = raw_signal.sort_values(ascending=False)
    
    # Top 3 Long et Top 3 Short
    views_assets = ranked.head(3).index.tolist() + ranked.tail(3).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        val = raw_signal[t]
        
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # Le signal ARIMA est petit (ex: 0.0005 journalier).
        # On l'annualise (*252) ou on l'amplifie pour en faire une "Vue" BL convaincante
        # On utilise tanh pour saturer à +/- 30% annuel
        annualized_signal = val * 252
        view_ret = np.tanh(annualized_signal * 2.0) * 0.30 
        Q.append(view_ret)
        
        variance = sigma.iloc[idx, idx]
        Omega_list.append(variance * TAU)

    P = np.array(P); Q = np.array(Q); Omega = np.diag(Omega_list)
    
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
# 5. BACKTEST ENGINE (TRADE-ON-CLOSE)
# ==============================================================================
print("\n[3] Executing ARIMA Backtest...")

rebal_dates = returns_total.resample('W-FRI').last().index
daily_dates = returns_total.index
rebal_set = set(rebal_dates)

current_weights_daily = np.zeros(len(tickers))
capital = TARGET_BUDGET
equity_curve = [capital]
equity_dates = [daily_dates[0]]
strategy_returns = []

last_log_month = 0

print("-" * 100)
print(f"{'DATE':<12} | {'EQUITY':<12} | {'LEV':<5} | {'ALLOCATION (ARIMA Views)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    
    # 1. PnL
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector)
    capital *= (1 + port_ret)
    
    # 2. Rebal
    cost = 0.0
    if d in rebal_set:
        target_weights = get_black_litterman_weights(d)
        target_weights[np.abs(target_weights) < 0.02] = 0 
        
        turnover = np.sum(np.abs(target_weights - current_weights_daily))
        cost = turnover * TRANSACTION_COST
        capital -= (capital * cost)
        
        current_weights_daily = target_weights
        
        if d.month != last_log_month:
            lev = np.sum(np.abs(target_weights))
            pos_l = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w > 0.05]
            pos_s = [f"{tickers[x][:-2]}:{w:.0%}" for x, w in enumerate(target_weights) if w < -0.05]
            l_str = ",".join(pos_l) if pos_l else "."
            s_str = ",".join(pos_s) if pos_s else "."
            print(f"{d.date()} | {capital:,.0f} | {lev:.1f}x  | L:[{l_str}] S:[{s_str}]")
            last_log_month = d.month

    equity_curve.append(capital)
    equity_dates.append(d)
    strategy_returns.append(port_ret - cost)

# Export
df_export = returns_total.loc[daily_dates[1:]].copy()
df_export['STRATEGY'] = strategy_returns
df_export.to_csv(FILE_RETURNS)

print("-" * 100)
print(f"Final Capital: {capital:,.0f} $")

# ==============================================================================
# 6. REPORTING & CHARTS
# ==============================================================================
def generate_reports():
    print("\n[4] Generating Reports...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = returns['STRATEGY']
    
    # Metrics
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    ret_ann = strat_ret.mean() * TRADING_DAYS
    sharpe = ret_ann / vol if vol > 0 else 0
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    max_dd = dd.min()
    
    print(f"\n--- PERFORMANCE (ARIMA BL) ---")
    print(f"Total Return       : {(capital/TARGET_BUDGET)-1:+.2%}")
    print(f"Annualized Return  : {ret_ann:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Max Drawdown       : {max_dd:.2%}")
    
    # Charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df_equity['Equity'], color='purple', linewidth=1.5, label='ARIMA BL Strategy')
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--')
    ax1.set_title("Strategy Performance: ARIMA-Driven Black-Litterman", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_ARIMA_PERF.png"), dpi=150)
    print("-> Charts saved.")
    plt.show()

if __name__ == "__main__":
    generate_reports()