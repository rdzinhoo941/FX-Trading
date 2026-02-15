import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import xgboost as xgb
import os
import warnings

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

# Directory
OUTPUT_DIR = "backtest_results_xgboost"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_XGB.csv")

# --- Strategy Params ---
TARGET_BUDGET = 1000000.0   
TRANSACTION_COST = 0.0005    
LOOKBACK_COV = 126           # Pour la matrice de risque BL

# --- XGBoost Params ---
# On réentraîne le modèle tous les 20 jours de trading (1 mois)
RETRAIN_FREQ = 20            
TRAIN_WINDOW = 500           # On utilise les 2 dernières années pour entraîner (Rolling Window)

# --- Black-Litterman Params ---
RISK_AVERSION_DELTA = 2.5    
TAU = 0.05                   
MAX_WEIGHT = 0.30            
GROSS_LEVERAGE = 5         # Levier 5x

# --- Risk Params ---
CONFIDENCE_LEVEL = 0.95  
TRADING_DAYS = 252        

print("="*80)
print(f" 🚀 STARTING: XGBOOST AI BLACK-LITTERMAN STRATEGY")
print("="*80)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
print("\n[1] Loading Local Data...")
try:
    prices = pd.read_csv('data/data_forex_prices.csv', index_col=0, parse_dates=True)
    rates_daily = pd.read_csv('data/data_fred_rates.csv', index_col=0, parse_dates=True)
    
    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx]
    rates_daily = rates_daily.loc[common_idx]
    
    # On commence large pour le Feature Engineering
    prices = prices.loc["2015-01-01":]
    rates_daily = rates_daily.loc["2015-01-01":]
    
    tickers = prices.columns.tolist()
    print(f"   ✅ Data Loaded: {len(tickers)} pairs, {len(prices)} days.")

except FileNotFoundError:
    print("❌ CRITICAL ERROR: Data files not found.")
    exit()

# ==============================================================================
# 3. FEATURE ENGINEERING & XGBOOST SIGNALS
# ==============================================================================
print("\n[2] Training XGBoost Models (Walk-Forward)...")

# A. Préparation des Features
# ---------------------------
returns_total = pd.DataFrame(index=prices.index, columns=tickers)
for t in tickers:
    clean = t.replace('=X', '')
    base, quote = clean[:3], clean[3:]
    r_price = prices[t].pct_change()
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = (rates_daily[base] - rates_daily[quote]) / 252.0
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# On va créer un gros DataFrame avec toutes les features
# Pour simplifier, on entraîne un modèle PAR PAIRE
xgboost_predictions = pd.DataFrame(index=returns_total.index, columns=tickers)

def create_features(series_returns, carry_series):
    df = pd.DataFrame(index=series_returns.index)
    df['ret_1d'] = series_returns.shift(1)
    df['ret_5d'] = series_returns.rolling(5).mean().shift(1)
    df['ret_21d'] = series_returns.rolling(21).mean().shift(1) # Momentum mois
    df['vol_21d'] = series_returns.rolling(21).std().shift(1)  # Volatilité
    df['carry'] = carry_series.shift(1) # Le taux d'intérêt connu hier
    return df.dropna()

print("   -> Starting Rolling Training (This prevents Look-Ahead Bias)...")
# Période de prédiction (Backtest)
start_predict = "2018-01-01"
predict_dates = returns_total.loc[start_predict:].index

# On avance par blocs de RETRAIN_FREQ (ex: 1 mois)
# C'est beaucoup plus rapide que de réentraîner chaque jour
for i in range(0, len(predict_dates), RETRAIN_FREQ):
    current_date = predict_dates[i]
    # Date de fin du bloc (ou fin des données)
    next_date_idx = min(i + RETRAIN_FREQ, len(predict_dates))
    block_end_date = predict_dates[next_date_idx - 1]
    
    # Fenêtre d'entraînement : [current_date - TRAIN_WINDOW : current_date]
    train_end = current_date
    train_start = train_end - pd.Timedelta(days=TRAIN_WINDOW*1.5) # *1.5 pour couvrir les jours fériés
    
    # Logging léger
    if i % (RETRAIN_FREQ * 6) == 0: 
        print(f"      Training block: {current_date.date()} -> {block_end_date.date()}")
    
    for t in tickers:
        # 1. Création Features
        clean = t.replace('=X', '')
        base, quote = clean[:3], clean[3:]
        carry_s = (rates_daily[base] - rates_daily[quote])/252 if (base in rates_daily and quote in rates_daily) else pd.Series(0, index=rates_daily.index)
        
        X = create_features(returns_total[t], carry_s)
        y = returns_total[t].shift(-1) # Target = Return de DEMAIN
        
        # Alignement
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        
        # Split Train / Predict
        # Train : Tout ce qui est AVANT la date courante
        X_train = X.loc[train_start:current_date]
        y_train = y.loc[train_start:current_date]
        
        # Predict : Le bloc actuel (ex: le mois de Janvier)
        X_predict = X.loc[current_date:block_end_date]
        
        if len(X_train) < 100 or len(X_predict) == 0: continue
        
        # 2. Modèle XGBoost (Light Regressor)
        model = xgb.XGBRegressor(
            n_estimators=50,       # Pas besoin de 1000 arbres (trop lent + bruit)
            max_depth=3,           # Arbres peu profonds (évite overfitting)
            learning_rate=0.05,
            objective='reg:squarederror',
            n_jobs=1               # 1 core par modèle (car on boucle déjà)
        )
        
        model.fit(X_train, y_train)
        
        # 3. Prédiction
        preds = model.predict(X_predict)
        
        # Stockage
        xgboost_predictions.loc[X_predict.index, t] = preds

# Nettoyage final
xgboost_predictions.dropna(how='all', inplace=True)
returns_total = returns_total.loc[xgboost_predictions.index]

print(f"   ✅ AI Signals Ready.")

# ==============================================================================
# 4. BLACK-LITTERMAN ENGINE
# ==============================================================================
def get_bl_xgboost_weights(curr_date):
    tickers_list = returns_total.columns.tolist()
    n_assets = len(tickers_list)
    
    # Covariance Historique
    hist_window_start = curr_date - pd.Timedelta(days=LOOKBACK_COV*1.5)
    history = returns_total.loc[hist_window_start:curr_date].tail(LOOKBACK_COV)
    
    if len(history) < LOOKBACK_COV * 0.9: return np.zeros(n_assets)

    sigma = history.cov() * 252 
    w_mkt = np.ones(n_assets) / n_assets 
    pi = RISK_AVERSION_DELTA * sigma.dot(w_mkt)
    
    # --- XGBOOST VIEWS ---
    if curr_date not in xgboost_predictions.index: return np.zeros(n_assets)
    
    # Le signal brut est la prédiction de rendement journalier (ex: 0.001 pour 0.1%)
    raw_signal = xgboost_predictions.loc[curr_date].fillna(0)
    
    # On sélectionne les meilleures opportunités (Top/Bottom)
    ranked = raw_signal.sort_values(ascending=False)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()
    
    P, Q, Omega_list = [], [], []
    
    for t in views_assets:
        idx = tickers_list.index(t)
        val = raw_signal[t]
        
        row = np.zeros(n_assets); row[idx] = 1; P.append(row)
        
        # On annualise la prédiction journalière pour l'utiliser comme Vue BL
        # Si XGB prédit +0.1% demain -> ~ +25% annuel. On sature avec tanh.
        annualized_pred = val * 252 
        view_ret = np.tanh(annualized_pred) * 0.30 
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
# 5. BACKTEST ENGINE (Trade-On-Close)
# ==============================================================================
print("\n[3] Executing Backtest...")

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
print(f"{'DATE':<12} | {'EQUITY':<12} | {'LEV':<5} | {'ALLOCATION (XGBoost Views)'}")
print("-" * 100)

for i, d in enumerate(daily_dates[1:]):
    
    # 1. PnL
    day_ret_vector = returns_total.loc[d]
    port_ret = np.dot(current_weights_daily, day_ret_vector)
    capital *= (1 + port_ret)
    
    # 2. Rebal
    cost = 0.0
    if d in rebal_set:
        target_weights = get_bl_xgboost_weights(d)
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
# 6. REPORTING
# ==============================================================================
def generate_reports():
    print("\n[4] Generating Reports...")
    df_equity = pd.DataFrame({'Equity': equity_curve}, index=equity_dates)
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    strat_ret = returns['STRATEGY']
    
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    ret_ann = strat_ret.mean() * TRADING_DAYS
    sharpe = ret_ann / vol if vol > 0 else 0
    dd = (df_equity['Equity'] - df_equity['Equity'].cummax()) / df_equity['Equity'].cummax()
    max_dd = dd.min()
    
    print(f"\n--- PERFORMANCE (XGBOOST BL) ---")
    print(f"Annualized Return  : {ret_ann:+.2%}")
    print(f"Sharpe Ratio       : {sharpe:.2f}")
    print(f"Max Drawdown       : {max_dd:.2%}")
    
    # Charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df_equity['Equity'], color='#2ca02c', linewidth=1.5, label='XGBoost Strategy')
    ax1.axhline(TARGET_BUDGET, color='black', linestyle='--')
    ax1.set_title("Strategy Performance: XGBoost AI + Black-Litterman", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dd, color='red', linewidth=1)
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "CHART_XGB_PERF.png"), dpi=150)
    print("-> Charts saved.")
    plt.show()

if __name__ == "__main__":
    generate_reports()