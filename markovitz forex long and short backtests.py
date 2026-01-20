import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARAMÈTRES GLOBAUX
# ==============================================================================
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5     
LEVERAGES_TO_TEST = [1.0, 2.0, 3.0, 4.0, 5.0]
YEARS_TO_TEST = [2020, 2021, 2022, 2023, 2024]

# Liste Univers Global Macro
pairs_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", 
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", 
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" 
]
tickers = [f"{pair}=X" for pair in pairs_list]
benchmark_ticker = "DX-Y.NYB"

# On prend large pour avoir l'historique nécessaire au calcul de cov
start_date = "2018-01-01" 
end_date = "2025-01-01"

print("█"*80)
print(f"🚀 DÉMARRAGE DU STRESS TEST MULTI-LEVIER ({len(LEVERAGES_TO_TEST)} Scénarios x {len(YEARS_TO_TEST)} Années)")
print("█"*80)

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES (UNE SEULE FOIS)
# ==============================================================================
print("\n📥 1. Téléchargement des Données...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
# Nettoyage
data = raw_data.dropna(axis=1, how='all').ffill().dropna()
valid_tickers = list(data.columns)
print(f"   ✅ {len(valid_tickers)} paires valides.")

# Taux (Hybride FRED + Manuel)
print("📥 2. Construction Taux d'Intérêt...")
fred_codes = {
    'USD': 'IR3TIB01USM156N', 'EUR': 'IR3TIB01EZM156N', 'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N', 'CHF': 'IR3TIB01CHM156N', 'AUD': 'IR3TIB01AUM156N',
    'CAD': 'IR3TIB01CAM156N', 'NZD': 'IR3TIB01NZM156N'
}
rates_daily = pd.DataFrame(0.0, index=data.index, columns=fred_codes.keys())
try:
    df_fred = web.DataReader(list(fred_codes.values()), 'fred', start_date, end_date)
    df_fred.columns = list(fred_codes.keys())
    rates_daily.update((df_fred / 100.0).resample('D').ffill())
except:
    rates_daily['USD'] = 0.05 # Fallback

# Taux Exotiques (Indispensable)
exotic_rates = {
    'TRY': 0.40, 'BRL': 0.11, 'ZAR': 0.08, 'MXN': 0.11,
    'INR': 0.06, 'NOK': 0.04, 'SEK': 0.04, 'SGD': 0.03
}
for curr, rate in exotic_rates.items():
    rates_daily[curr] = rate

# Calcul Rendements Totaux (Prix + Carry)
print("⚙️ Calcul de la matrice des Rendements Totaux...")
returns_total = pd.DataFrame(index=data.index, columns=valid_tickers)
price_returns = data.pct_change()

for t in valid_tickers:
    clean_t = t.replace('=X', '')
    base, quote = clean_t[:3], clean_t[3:]
    r_price = price_returns[t]
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        r_carry = (rates_daily[base] - rates_daily[quote]) / 252
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# ==============================================================================
# 3. FONCTION D'OPTIMISATION PARAMÉTRABLE
# ==============================================================================
def optimize_markowitz(mu, sigma, risk_aversion, max_leverage):
    n = len(mu)
    init_guess = np.repeat(1/n, n)
    
    def objective(weights):
        utility = (np.dot(weights, mu) * 252) - (risk_aversion / 2) * (np.dot(weights.T, np.dot(sigma, weights)) * 252)
        return -utility

    cons_net = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    # CONTRAINTE DYNAMIQUE : LEVIER GLOBAL MAX
    cons_lev = {'type': 'ineq', 'fun': lambda x: max_leverage - np.sum(np.abs(x))}
    
    constraints = [cons_net, cons_lev]
    # Limite individuelle pour éviter la concentration (ex: max 60% par actif)
    # On laisse assez large pour que le levier x5 puisse s'exprimer
    bounds = tuple((-max_leverage/2, max_leverage/2) for _ in range(n))
    
    try:
        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
    except:
        return init_guess

# ==============================================================================
# 4. BOUCLE DE STRESS TEST (ANNÉES x LEVIERS)
# ==============================================================================
# Création de la figure (3 lignes, 2 colonnes pour 5 années)
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten() # Pour itérer facilement

print("\n🏁 Lancement des simulations...")

for i, year in enumerate(YEARS_TO_TEST):
    ax = axes[i]
    print(f"   > Simulation Année {year}...")
    
    # Définition des dates pour cette année spécifique
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    
    # Dates de rebalancement (Vendredi)
    try:
        rebal_dates = returns_total.loc[year_start:year_end].resample('D').last().index
    except:
        print(f"Données manquantes pour {year}")
        continue
        
    if len(rebal_dates) < 2: continue

    # Pour chaque levier, on lance un backtest indépendant
    final_perfs = []
    
    for lev in LEVERAGES_TO_TEST:
        capital = TARGET_BUDGET
        curve = [capital]
        dates = [rebal_dates[0]]
        
        # --- Backtest Loop ---
        for j in range(len(rebal_dates) - 1):
            curr_date = rebal_dates[j]
            next_date = rebal_dates[j+1]
            
            # Données passées pour l'optimisation (Expanding ou Rolling window)
            # On prend les 6 derniers mois pour la covariance
            past_data = returns_total.loc[:curr_date].tail(126) 
            
            mu = past_data.mean()
            sigma = past_data.cov()
            
            # Optimisation avec le Levier Spécifique 'lev'
            w = optimize_markowitz(mu, sigma, RISK_AVERSION, max_leverage=lev)
            
            # Calcul Perf
            period_ret = returns_total.loc[curr_date:next_date].dot(w).sum()
            capital *= (1 + period_ret)
            
            curve.append(capital)
            dates.append(next_date)
        
        # Trace la courbe pour ce levier
        perf_pct = (capital / TARGET_BUDGET) - 1
        ax.plot(dates, curve, label=f"Lev {lev}x (Perf: {perf_pct:+.1%})")
        final_perfs.append(perf_pct)

    # Cosmétique du graphique annuel
    ax.set_title(f"Année {year}")
    ax.axhline(y=TARGET_BUDGET, color='black', linestyle=':', alpha=0.5)
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)

# Suppression du 6ème graphique vide (car on a 5 années et 6 emplacements)
if len(YEARS_TO_TEST) < 6:
    fig.delaxes(axes[5])

plt.tight_layout()
plt.suptitle(f"Comparatif Performance par Levier ({min(YEARS_TO_TEST)}-{max(YEARS_TO_TEST)})", y=1.02, fontsize=16)
plt.show()