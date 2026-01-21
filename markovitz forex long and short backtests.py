import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp  # <--- LE NOUVEAU MOTEUR
import pandas_datareader.data as web
import warnings
import time
import sys

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARAMÈTRES DE LA GRID SEARCH ÉTENDUE
# ==============================================================================
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5      

# PLAGE DE LEVIERS : De 1x à 15x
LEVERAGES = list(range(1, 16)) 

# PLAGE DE FENÊTRES D'ENTRAÎNEMENT (LOOKBACK PERIOD)
# C'est ici que l'on définit combien de passé on regarde pour calculer Mu et Sigma
# J'ai ajouté jusqu'à 3 ans comme demandé.
WINDOWS = {
    '2 Mois': 42,
    '3 Mois': 63,
    '6 Mois': 126,
    '1 An': 252,
    '1.5 Ans': 378,
    '2 Ans': 504,
    '3 Ans': 756  # <--- Ajout demandé
}

YEARS_TO_TEST = [2020, 2021, 2022, 2023, 2024]

# UNIVERS D'INVESTISSEMENT
pairs_list = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", 
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", 
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" 
]
tickers = [f"{pair}=X" for pair in pairs_list]

# On charge plus large (2016) pour que la fenêtre de 3 ans ait assez de données pour commencer en 2020
start_date = "2016-01-01" 
end_date = "2025-01-01"

print("█"*80)
print(f"🚀 DÉMARRAGE GRID SEARCH MASSIVE (MOTEUR CVXPY)")
print(f"🎯 Leviers: 1x à 15x")
print(f"⏳ Fenêtres (Lookback): {list(WINDOWS.keys())}")
print(f"📅 Années testées: {YEARS_TO_TEST}")
print("█"*80)

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
print("\n📥 1. Téléchargement des Données (Prix)...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
data = raw_data.dropna(axis=1, how='all').ffill().dropna()
valid_tickers = list(data.columns)
print(f"   ✅ {len(valid_tickers)} paires valides récupérées.")

print("📥 2. Construction Taux d'Intérêt (Hybride)...")
fred_codes = {
    'USD': 'IR3TIB01USM156N', 'EUR': 'IR3TIB01EZM156N', 'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N', 'CHF': 'IR3TIB01CHM156N', 'AUD': 'IR3TIB01AUM156N',
    'CAD': 'IR3TIB01CAM156N', 'NZD': 'IR3TIB01NZM156N'
}
rates_daily = pd.DataFrame(0.0, index=data.index, columns=fred_codes.keys())

# Bloc Try/Except pour FRED
try:
    df_fred = web.DataReader(list(fred_codes.values()), 'fred', start_date, end_date)
    df_fred.columns = list(fred_codes.keys())
    # Forward fill pour convertir mensuel en quotidien
    rates_daily.update((df_fred / 100.0).resample('D').ffill())
except Exception as e:
    print(f"⚠️ Erreur FRED: {e}. Utilisation taux fixes USD par défaut.")
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
    
    # Calcul du Carry
    r_carry = 0.0
    if base in rates_daily.columns and quote in rates_daily.columns:
        # On aligne les dates des taux sur les dates des prix
        r_base = rates_daily[base].reindex(data.index).ffill()
        r_quote = rates_daily[quote].reindex(data.index).ffill()
        r_carry = (r_base - r_quote) / 252
        
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# ==============================================================================
# 3. MOTEUR D'OPTIMISATION (VERSION CVXPY - RAPIDE)
# ==============================================================================
def get_optimal_weights_cvxpy(mu, sigma, risk_aversion, max_leverage):
    """
    Résout le problème Markowitz avec contraintes de levier via CVXPY.
    Beaucoup plus rapide et robuste que scipy.optimize.
    """
    n = len(mu)
    
    # Variables de décision (Poids)
    w = cp.Variable(n)
    
    # Conversion inputs en numpy pour CVXPY
    mu_np = mu.values
    sigma_np = sigma.values
    
    # Stabilisation de la matrice (Covariance doit être PSD)
    # On ajoute un epsilon minuscule sur la diagonale pour éviter les erreurs numériques
    sigma_np += np.eye(n) * 1e-8
    
    # Fonction Objectif: Maximiser (Rendement - Penalité Risque)
    # Note: mu est daily, sigma est daily. On annualise dans la formule.
    ret = w @ mu_np
    risk = cp.quad_form(w, sigma_np)
    
    # Maximiser Utilité
    objective = cp.Maximize((ret * 252) - (risk_aversion / 2) * (risk * 252))
    
    # Contraintes
    constraints = [
        cp.sum(w) == 1.0,               # Investissement Net = 100%
        cp.norm(w, 1) <= max_leverage,  # Levier Brut (Somme des val abs) <= Max
        cp.abs(w) <= max_leverage * 0.6 # Pas plus de 60% du levier sur un seul actif
    ]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        # OSQP est un solveur très robuste pour les problèmes quadratiques
        prob.solve(solver=cp.OSQP, verbose=False)
        
        if w.value is None:
            # Fallback si échec (rare)
            return np.ones(n) / n
        
        # Petit nettoyage pour mettre les 0.0000001 à 0
        weights = w.value
        weights[np.abs(weights) < 1e-5] = 0
        return weights
        
    except Exception:
        return np.ones(n) / n

# ==============================================================================
# 4. GRID SEARCH EXECUTION
# ==============================================================================
results_log = [] 

print("\n🏁 Lancement de la Simulation...")
start_time_total = time.time()

for year in YEARS_TO_TEST:
    print(f"\n" + "="*60)
    print(f"📅 TRAITEMENT ANNÉE {year}")
    print("="*60)
    
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    
    trading_days = returns_total.loc[year_start:year_end].index
    
    if len(trading_days) < 10:
        print(f"⚠️ Pas assez de données pour {year}")
        continue

    # --- BOUCLE FENÊTRES DE DONNÉES (LOOKBACK) ---
    for win_name, win_days in WINDOWS.items():
        print(f"   🔎 Grid Lookback: {win_name} ({win_days} jours d'historique)")
        
        # Init capitaux
        capitals = {lev: TARGET_BUDGET for lev in LEVERAGES}
        
        # --- BOUCLE JOUR PAR JOUR (DAILY REBALANCING) ---
        n_days = len(trading_days)
        
        for i in range(n_days - 1):
            curr_date = trading_days[i]
            next_date = trading_days[i+1]
            
            # LOGGING DYNAMIQUE (Pour voir ce qui se passe)
            if i % 20 == 0: # Affiche tous les 20 jours
                sys.stdout.write(f"\r      ⏳ Progression {year} | {win_name}: Jour {i}/{n_days} traité...")
                sys.stdout.flush()
            
            # 1. Extraction des données (La fenêtre glissante)
            hist_data = returns_total.loc[:curr_date].tail(win_days)
            
            if len(hist_data) < win_days * 0.95: 
                continue 
            
            mu = hist_data.mean()
            sigma = hist_data.cov()
            
            # Rendement réel du marché pour le lendemain
            day_returns = returns_total.loc[next_date]            
            # 2. Boucle sur les Leviers (Calcul des poids et mise à jour)
            for lev in LEVERAGES:
                # Utilisation de CVXPY ici
                w = get_optimal_weights_cvxpy(mu, sigma, RISK_AVERSION, max_leverage=lev)
                
                # Performance
                port_ret = np.dot(w, day_returns)
                capitals[lev] *= (1 + port_ret)
        
        print(f"\n      ✅ Terminé pour {win_name}.")

        # --- SAUVEGARDE DES RÉSULTATS POUR CETTE FENÊTRE ---
        best_lev = 0
        best_perf = -999.0
        
        for lev in LEVERAGES:
            final_cap = capitals[lev]
            perf = (final_cap / TARGET_BUDGET) - 1
            
            results_log.append({
                'Annee': year,
                'Fenetre_Lookback': win_name,
                'Jours_Entrainement': win_days,
                'Levier': lev,
                'Capital_Fin': final_cap,
                'Performance': perf
            })
            
            if perf > best_perf:
                best_perf = perf
                best_lev = lev

        # Petit résumé console
        print(f"      🏆 Best: Levier {best_lev}x -> {best_perf:+.2%}")
        worst_lev = min(capitals, key=capitals.get)
        print(f"      💀 Worst: Levier {worst_lev}x -> {(capitals[worst_lev]/TARGET_BUDGET)-1:+.2%}")
        print("-" * 40)

# ==============================================================================
# 5. ANALYSE ET EXPORT
# ==============================================================================
print("\n" + "█"*80)
print("✅ SIMULATION TERMINÉE.")
print("█"*80)

df_results = pd.DataFrame(results_log)
csv_filename = "grid_search_forex_cvxpy.csv"
df_results.to_csv(csv_filename, index=False)
print(f"💾 Résultats détaillés sauvegardés dans : {csv_filename}")

# --- TABLEAU RECAPITULATIF CROISÉ (Moyenne Performance par Fenêtre et par Levier) ---
print("\n📊 TABLEAU DE SYNTHÈSE (Performance Moyenne Annuelle)")
pivot_table = df_results.pivot_table(
    index='Fenetre_Lookback', 
    columns='Levier', 
    values='Performance', 
    aggfunc='mean'
)

# On affiche une sélection de leviers pour la lisibilité
cols_to_show = [1, 3, 5, 10, 15]
print(pivot_table[cols_to_show].applymap(lambda x: f"{x:.2%}"))

# --- GRAPHIQUE FINAL ---
plt.figure(figsize=(15, 8))

# On va plotter la moyenne des performances (toutes années confondues) pour chaque fenêtre
# pour voir quelle quantité de données historique est la meilleure.
for win_name in WINDOWS.keys():
    subset = df_results[df_results['Fenetre_Lookback'] == win_name]
    # Groupe par levier pour avoir la courbe
    avg_perf_by_lev = subset.groupby('Levier')['Performance'].mean()
    plt.plot(avg_perf_by_lev.index, avg_perf_by_lev.values, marker='o', label=f"Lookback: {win_name}")

plt.title("Impact de l'Historique de Données (Lookback) sur la Performance Moyenne (2020-2024)")
plt.xlabel("Levier Max")
plt.ylabel("Performance Moyenne Annuelle")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.show()