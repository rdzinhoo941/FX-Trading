import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# --- PARAMÈTRES ---
TARGET_BUDGET = 1000000.0  
RISK_AVERSION = 2.5     
#pairs_list_raw = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
pairs_list_raw = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD","CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD","USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"]
tickers = [f"{pair}=X" for pair in pairs_list_raw]
benchmark_ticker = "DX-Y.NYB"

# --- 1. TÉLÉCHARGEMENT ---
print("📥 Téléchargement des données (Forex + Dollar index)...")

# Téléchargement Forex
raw_data = yf.download(tickers, start="2010-01-01", end="2025-01-01", progress=False)
if 'Adj Close' in raw_data.columns.levels[0] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.columns:
    data = raw_data['Adj Close']
else:
    data = raw_data['Close']
data = data.ffill().dropna()
returns_daily = data.pct_change().dropna()

# Téléchargement Benchmark (dollar index)
bench_raw = yf.download(benchmark_ticker, start="2020-01-01", end="2025-01-01", progress=False)
bench_data = bench_raw['Adj Close'] if 'Adj Close' in bench_raw else bench_raw['Close']
bench_returns = bench_data.pct_change().dropna()

# Périodes
start_test_date = "2020-01-01"
train_data = returns_daily.loc[:start_test_date] 
test_data = returns_daily.loc[start_test_date:]

# --- 2. OPTIMISATION MARKOWITZ ---
def optimize_markowitz(mu, sigma, risk_aversion, max_leverage=1.5):
    num_assets = len(mu)
    init_guess = np.repeat(1/num_assets, num_assets)
    
    def objective(weights):
        utility = (np.dot(weights, mu) * 252) - (risk_aversion / 2) * (np.dot(weights.T, np.dot(sigma, weights)) * 252)
        return -utility

    # Contrainte 1 : Somme nette = 100% (Budget investi)
    cons_net = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    # Contrainte 2 : Somme des valeurs absolues <= Levier Max (ex: 1.5 ou 2.0)
    # C'est LA contrainte qui empêche l'algo de tricher
    cons_gross = {'type': 'ineq', 'fun': lambda x: max_leverage - np.sum(np.abs(x))}
    
    constraints = [cons_net, cons_gross]
    bounds = tuple((-1.0, 1.0) for _ in range(num_assets))
    
    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
"""
# --- 3. RECHERCHE DE LA FENÊTRE OPTIMALE (GRID SEARCH) ---
print("\n⚙️ Lancement du comparatif des fenêtres d'apprentissage...")

# Liste des fenêtres à tester (en jours de bourse)
# 63 = 3 mois, 126 = 6 mois, 252 = 1 an, 756 = 3 ans, 0 = Tout l'historique (Expanding)
windows_to_test = [63, 126, 252, 504, 756, 0]
results_store = {}

monthly_dates = test_data.resample('M').last().index
# On concatène tout pour pouvoir piocher dedans
full_returns = pd.concat([train_data, test_data])

for w in windows_to_test:
    window_name = f"Window {w}j" if w > 0 else "Full History"
    print(f"   > Test de : {window_name}...")
    
    capital = TARGET_BUDGET
    curve = []
    dates = []
    
    for i in range(len(monthly_dates) - 1):
        current_date = monthly_dates[i]
        next_date = monthly_dates[i+1]
        
        # --- SÉLECTION DES DONNÉES D'APPRENTISSAGE ---
        available_data = full_returns.loc[:current_date]
        
        if w > 0:
            # Fenêtre Glissante (Rolling)
            learning_data = available_data.tail(w)
        else:
            # Fenêtre Expansive (Tout l'historique dispo)
            learning_data = available_data
            
        # Calcul Mu / Sigma sur cette fenêtre spécifique
        mu = learning_data.mean() * 252
        sigma = learning_data.cov() * 252
        
        # Optimisation
        try:
            weights = optimize_markowitz(mu, sigma, RISK_AVERSION)
        except:
            weights = np.repeat(1/len(tickers), len(tickers))
            
        # Forward Test (Mois suivant)
        mask = (test_data.index > current_date) & (test_data.index <= next_date)
        month_returns = test_data.loc[mask]
        
        if month_returns.empty: continue
            
        # Calcul Perf
        daily_pnl = month_returns.dot(weights)
        for ret in daily_pnl:
            capital *= (1 + ret)
            curve.append(capital)
            dates.append(month_returns.index[daily_pnl.values == ret][0])
            
    # Stockage du résultat
    results_store[window_name] = pd.Series(curve, index=dates)

# --- 4. AFFICHAGE DU VAINQUEUR ---
print("\n" + "="*60)
print(f"{'FENÊTRE':<20} | {'RENDEMENT TOTAL':<15} | {'SHARPE':<10}")
print("-" * 60)

plt.figure(figsize=(12, 6))

for name, series in results_store.items():
    if series.empty: continue
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    daily_ret = series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    print(f"{name:<20} | {total_ret:>14.2%} | {sharpe:>9.2f}")
    plt.plot(series, label=f"{name} ({total_ret:.0%})")

print("="*60)

# Ajout du Benchmark pour comparer
bench_rebased = bench_data.reindex(dates).ffill()
if not bench_rebased.empty:
    bench_rebased = bench_rebased / bench_rebased.iloc[0] * TARGET_BUDGET
    plt.plot(bench_rebased, label='Benchmark DXY', color='black', linestyle='--', linewidth=2, alpha=0.5)

plt.title('Comparatif des Périodes d\'Apprentissage (Rolling Windows)')
plt.ylabel('Valeur Portefeuille')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
# --- 3. BACKTEST COMPARATIF ---
print("\n⚙️ Calcul des stratégies...")

monthly_dates = test_data.resample('M').last().index
history_returns = train_data.copy()

# Portefeuilles
capital_markowitz = TARGET_BUDGET
capital_equal = TARGET_BUDGET
values_markowitz = []
values_equal = []
dates_history = []

for i in range(len(monthly_dates) - 1):
    current_date = monthly_dates[i]
    next_date = monthly_dates[i+1]
    
    # --- Stratégie A: Markowitz ---
    mu = history_returns.mean()
    sigma = history_returns.cov()
    w_markowitz = optimize_markowitz(mu, sigma, RISK_AVERSION)
    
    # --- Stratégie B: Equal Weight (1/N) ---
    n = len(tickers)
    w_equal = np.repeat(1/n, n) 
    
    # --- Forward Test ---
    mask = (test_data.index > current_date) & (test_data.index <= next_date)
    month_returns = test_data.loc[mask]
    
    if month_returns.empty:
        continue
        
    # Calcul PnL Markowitz
    daily_ret_markowitz = month_returns.dot(w_markowitz)
    for ret in daily_ret_markowitz:
        capital_markowitz *= (1 + ret)
        values_markowitz.append(capital_markowitz)
        dates_history.append(month_returns.index[daily_ret_markowitz.values == ret][0])

    # Calcul PnL Equal Weight
    daily_ret_equal = month_returns.dot(w_equal)
    for ret in daily_ret_equal:
        capital_equal *= (1 + ret)
        values_equal.append(capital_equal)
    
    # Update History
    history_returns = pd.concat([history_returns, month_returns])

# Création DataFrame final
df_results = pd.DataFrame(index=dates_history)
df_results['Markowitz'] = values_markowitz
df_results['Equal Weight'] = values_equal

# --- CORRECTION DU JOIN ICI ---
# Au lieu de .join() et .rename(), on assigne directement.
# Pandas aligne les dates automatiquement. Si une date manque, il met NaN.
df_results['Dollar index'] = bench_data

# On supprime les lignes où il manque des données (ex: jours fériés différents entre Forex et Bourse)
df_results = df_results.dropna()

# Rebaser le Dollar index à 1M$ pour comparer (Base 100 au début de la période commune)
# On utilise iloc[0] pour prendre le premier point valide après le dropna
df_results['Dollar index'] = df_results['Dollar index'] / df_results['Dollar index'].iloc[0] * TARGET_BUDGET
# --- 4. MÉTRIQUES ET AFFICHAGE ---
def calculate_metrics(series):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    daily_ret = series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    dd = (series / series.cummax() - 1).min()
    return total_ret, sharpe, dd

print("\n" + "="*50)
print(f"{'STRATÉGIE':<15} | {'RENDEMENT':<10} | {'SHARPE':<8} | {'MAX DD':<10}")
print("-" * 50)

for col in df_results.columns:
    ret, sharpe, dd = calculate_metrics(df_results[col])
    print(f"{col:<15} | {ret:,.2%}    | {sharpe:.2f}     | {dd:.2%}")

print("="*50)

plt.figure(figsize=(12, 6))
plt.plot(df_results['Markowitz'], label='Markowitz (Long/Short)', color='blue', linewidth=1.5)
plt.plot(df_results['Equal Weight'], label='Equal Weight (Naïf)', color='gray', linestyle='--', alpha=0.7)
plt.plot(df_results['Dollar index'], label='Dollar index (Benchmark)', color='orange', alpha=0.6)
plt.axhline(y=TARGET_BUDGET, color='r', linestyle=':', label='Budget Initial')
plt.title('Performance : Markowitz vs Benchmarks (2020-2025)')
plt.ylabel('Valeur Portefeuille ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 5. GÉNÉRATION DES ORDRES POUR DEMAIN ---
print("\n" + "█"*80)
print(f"🔮 ORDRES À EXÉCUTER IMMÉDIATEMENT (Budget: {TARGET_BUDGET:,.0f} $)")
print("█"*80)

# 1. On recalcul sur TOUTE la donnée disponible jusqu'à ce soir
full_mu = returns_daily.mean()
full_sigma = returns_daily.cov()

# 2. Optimisation finale
try:
    optimal_weights = optimize_markowitz(full_mu, full_sigma, RISK_AVERSION)
except:
    # Fallback si l'optimiseur échoue (rare)
    optimal_weights = np.repeat(1/len(tickers), len(tickers))

# 3. Récupération des derniers prix
last_prices = data.iloc[-1]

# 4. Affichage du Ticket d'Ordre
print(f"{'ACTION':<10} | {'PAIRE':<10} | {'PRIX ACTUEL':<12} | {'POIDS CIBLE':<12} | {'MONTANT ($)':<15} | {'UNITÉS'}")
print("-" * 90)

total_invested = 0
for i, ticker in enumerate(tickers):
    weight = optimal_weights[i]
    
    # On ignore les positions minuscules (< 1%)
    if abs(weight) < 0.01: continue
        
    pair_name = ticker.replace('=X', '')
    price = last_prices[ticker]
    amount = weight * TARGET_BUDGET
    units = int(amount / price) # Approximation (sur Forex c'est notionnel)
    
    action = "🟢 ACHAT" if weight > 0 else "🔴 VENTE"
    
    print(f"{action:<10} | {pair_name:<10} | {price:<12.4f} | {weight*100:>10.1f} % | {amount:>14,.2f} $ | {units:>10,}")
    total_invested += abs(amount)

print("-" * 90)
print(f"EXPOSITION TOTALE (LEVIER) : {total_invested / TARGET_BUDGET:.2f}x")
print("Note : 'VENTE' signifie Vente à Découvert (Short).")
plt.show()
