import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import warnings
import time

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARAMÈTRES DE LA GRID SEARCH
# ==============================================================================
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5     

# PLAGE DE LEVIERS : De 1x à 15x
LEVERAGES = list(range(1, 16)) # [1, 2, ..., 15]

# PLAGE DE FENÊTRES D'ENTRAÎNEMENT (En jours ouvrés approx)
# 1m=21, 2m=42, 3m=63, 4m=84, 6m=126, 9m=189, 1y=252, 1.5y=378, 2y=504
WINDOWS = {
    '1 Mois': 21,
    '2 Mois': 42,
    '3 Mois': 63,
    '4 Mois': 84,
    '6 Mois': 126,
    '9 Mois': 189,
    '1 An': 252,
    '1.5 Ans': 378,
    '2 Ans': 504
}

YEARS_TO_TEST = [2020, 2021, 2022, 2023, 2024]

# UNIVERS D'INVESTISSEMENT
pairs_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", 
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", 
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" 
]
tickers = [f"{pair}=X" for pair in pairs_list]
benchmark_ticker = "DX-Y.NYB"

# On charge large pour avoir assez d'historique pour la fenêtre de 2 ans
start_date = "2017-01-01" 
end_date = "2025-01-01"

print("█"*80)
print(f"🚀 DÉMARRAGE GRID SEARCH MASSIVE")
print(f"🎯 Leviers: 1x à 15x")
print(f"⏳ Fenêtres: {list(WINDOWS.keys())}")
print(f"📅 Années: {YEARS_TO_TEST}")
print("█"*80)

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
print("\n📥 1. Téléchargement des Données (Prix)...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
data = raw_data.dropna(axis=1, how='all').ffill().dropna()
valid_tickers = list(data.columns)
print(f"   ✅ {len(valid_tickers)} paires valides.")

print("📥 2. Construction Taux d'Intérêt (Hybride)...")
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
    rates_daily['USD'] = 0.05 

exotic_rates = {
    'TRY': 0.40, 'BRL': 0.11, 'ZAR': 0.08, 'MXN': 0.11,
    'INR': 0.06, 'NOK': 0.04, 'SEK': 0.04, 'SGD': 0.03
}
for curr, rate in exotic_rates.items():
    rates_daily[curr] = rate

print("⚙️ Calcul Rendements Totaux (Prix + Carry)...")
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
# 3. MOTEUR D'OPTIMISATION
# ==============================================================================
def get_optimal_weights(mu, sigma, risk_aversion, max_leverage):
    n = len(mu)
    init_guess = np.repeat(1/n, n)
    
    def objective(w):
        utility = (np.dot(w, mu) * 252) - (risk_aversion / 2) * (np.dot(w.T, np.dot(sigma, w)) * 252)
        return -utility # On veut maximiser l'utilité

    # Ici, petite astuce pour accélérer : On utilise une formule analytique simplifiée si possible
    # Mais avec les contraintes de levier, on doit utiliser minimize.
    # On va réduire la précision (tol) pour gagner du temps, car on fait du "grossier"
    
    def objective_func(w):
        return -((np.dot(w, mu) * 252) - (risk_aversion / 2) * (np.dot(w.T, np.dot(sigma, w)) * 252))

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}, # Investissement net = 100%
        {'type': 'ineq', 'fun': lambda x: max_leverage - np.sum(np.abs(x))} # Levier Max
    ]
    # Limite par actif pour éviter la concentration extrême (ex: +/- 60% du levier max)
    limit = max_leverage * 0.6
    bounds = tuple((-limit, limit) for _ in range(n))
    
    try:
        res = minimize(objective_func, init_guess, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-4)
        return res.x
    except:
        return init_guess

# ==============================================================================
# 4. GRID SEARCH EXECUTION
# ==============================================================================
results_log = [] # Pour stocker (Année, Fenêtre, Levier, Perf, Sharpe)

print("\n🏁 Lancement de la Simulation...")
start_time_total = time.time()

for year in YEARS_TO_TEST:
    print(f"\n" + "="*60)
    print(f"📅 TRAITEMENT ANNÉE {year}")
    print("="*60)
    
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    
    # On récupère les dates de trading de l'année (Daily)
    trading_days = returns_total.loc[year_start:year_end].index
    
    if len(trading_days) < 10:
        print(f"⚠️ Pas assez de données pour {year}")
        continue

    # --- BOUCLE FENÊTRES (TIME WINDOWS) ---
    for win_name, win_days in WINDOWS.items():
        print(f"   🔎 Fenêtre: {win_name} ({win_days}j)...")
        
        # Initialisation des capitaux pour chaque levier
        capitals = {lev: TARGET_BUDGET for lev in LEVERAGES}
        
        # --- BOUCLE JOUR PAR JOUR (DAILY REBALANCING) ---
        # C'est ici que ça se joue. On ne rebalance PAS tous les jours pour gagner du temps ? 
        # NON, tu as demandé du Daily. On fait du Daily.
        
        for i in range(len(trading_days) - 1):
            curr_date = trading_days[i]
            next_date = trading_days[i+1]
            
            # 1. Calcul des données historiques (Mu, Sigma) -> FAIT UNE SEULE FOIS PAR JOUR
            # On prend les 'win_days' derniers jours
            hist_data = returns_total.loc[:curr_date].tail(win_days)
            
            # Sécurité : Si pas assez d'historique (ex: début 2020 avec fenêtre 2 ans)
            if len(hist_data) < win_days * 0.9: 
                continue 
            
            mu = hist_data.mean()
            sigma = hist_data.cov()
            
            # Rendement du marché pour le jour suivant (vecteur)
            day_returns = returns_total.loc[curr_date:next_date].sum() # approx du jour
            
            # 2. Boucle sur les Leviers (Calcul des poids et mise à jour)
            for lev in LEVERAGES:
                # Optimisation
                w = get_optimal_weights(mu, sigma, RISK_AVERSION, max_leverage=lev)
                
                # Performance
                port_ret = np.dot(w, day_returns)
                capitals[lev] *= (1 + port_ret)
        
        # --- FIN DE L'ANNÉE POUR CETTE FENÊTRE ---
        # On loggue les résultats
        print(f"      [RÉSULTATS {win_name}]")
        best_lev = 0
        best_perf = -999.0
        
        for lev in LEVERAGES:
            final_cap = capitals[lev]
            perf = (final_cap / TARGET_BUDGET) - 1
            
            # On sauvegarde pour le CSV
            results_log.append({
                'Annee': year,
                'Fenetre': win_name,
                'Jours_Entrainement': win_days,
                'Levier': lev,
                'Capital_Fin': final_cap,
                'Performance': perf
            })
            
            # Affichage console intelligent (pas tout, juste les bornes et le best)
            if perf > best_perf:
                best_perf = perf
                best_lev = lev
                
            # On affiche quelques étapes clés pour contrôler
            if lev in [1, 5, 10, 15]: 
                print(f"      - Levier {lev}x : {perf:>7.2%}")
        
        print(f"      🏆 MEILLEUR LEVIER : {best_lev}x ({best_perf:+.2%})")

# ==============================================================================
# 5. ANALYSE ET EXPORT
# ==============================================================================
print("\n" + "█"*80)
print("✅ SIMULATION TERMINÉE.")
print("█"*80)

df_results = pd.DataFrame(results_log)
# Sauvegarde
csv_filename = "grid_search_forex_results.csv"
df_results.to_csv(csv_filename, index=False)
print(f"💾 Résultats détaillés sauvegardés dans : {csv_filename}")

# --- AFFICHAGE DU TABLEAU RECAPITULATIF (Moyenne par Fenêtre) ---
print("\n📊 MOYENNE DE PERFORMANCE PAR FENÊTRE (Toutes années confondues, Levier 3x vs 10x)")
print(f"{'FENÊTRE':<15} | {'LEV 3x (Moy)':<15} | {'LEV 10x (Moy)':<15}")
print("-" * 50)

for win_name in WINDOWS.keys():
    mask = df_results['Fenetre'] == win_name
    
    # Perf moyenne levier 3
    avg_3 = df_results[mask & (df_results['Levier'] == 3)]['Performance'].mean()
    # Perf moyenne levier 10
    avg_10 = df_results[mask & (df_results['Levier'] == 10)]['Performance'].mean()
    
    print(f"{win_name:<15} | {avg_3:>14.2%} | {avg_10:>14.2%}")

# --- GRAPHIQUE FINAL : LE PLAFOND DE PERFORMANCE ---
# On prend l'année 2022 (la meilleure) et 2023 (la pire) pour voir la courbe Levier/Perf
plt.figure(figsize=(14, 6))

years_to_plot = [2022, 2023]
markers = ['o', 's', '^', 'D']

for i, year in enumerate(years_to_plot):
    plt.subplot(1, 2, i+1)
    subset = df_results[df_results['Annee'] == year]
    
    # On trace une ligne par fenêtre
    for win_name in ['1 Mois', '6 Mois', '2 Ans']:
        data_win = subset[subset['Fenetre'] == win_name]
        plt.plot(data_win['Levier'], data_win['Performance'], marker='o', label=win_name)
    
    plt.title(f"Impact du Levier - Année {year}")
    plt.xlabel("Levier Max")
    plt.ylabel("Performance Annuelle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.show()