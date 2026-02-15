import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration pour l'affichage
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [14, 10]

print("="*80)
print("🧐 AUDIT ET VISUALISATION DES DONNÉES (CHECK-UP)")
print("="*80)

# 1. CHARGEMENT
# ==============================================================================
try:
    print("📂 Chargement des fichiers CSV...")
    prices = pd.read_csv('data/data_forex_prices.csv', index_col=0, parse_dates=True)
    rates = pd.read_csv('data/data_fred_rates.csv', index_col=0, parse_dates=True)
    
    # On multiplie les taux par 100 pour l'affichage (0.05 -> 5%)
    rates_pct = rates * 100
    
    print(f"   ✅ Prix chargés : {prices.shape[0]} lignes, {prices.shape[1]} colonnes")
    print(f"   ✅ Taux chargés : {rates.shape[0]} lignes, {rates.shape[1]} colonnes")
    
except FileNotFoundError:
    print("❌ ERREUR : Les fichiers CSV sont introuvables. Lance 'data_downloader.py' d'abord.")
    exit()

# 2. STATISTIQUES RAPIDES (SANITY CHECK)
# ==============================================================================
print("\n📊 Vérification des Plages de Dates :")
if not prices.empty and not rates.empty:
    print(f"   Prix  : du {prices.index[0].date()} au {prices.index[-1].date()}")
    print(f"   Taux  : du {rates.index[0].date()} au {rates.index[-1].date()}")

    print("\n🔍 Recherche de valeurs manquantes (NaN) récentes :")
    missing_prices = prices.iloc[-100:].isna().sum().sum()
    missing_rates = rates.iloc[-100:].isna().sum().sum()
    print(f"   Prix manquants (100 derniers jours) : {missing_prices}")
    print(f"   Taux manquants (100 derniers jours) : {missing_rates}")
else:
    print("⚠️ Fichiers vides ou illisibles.")

# 3. VISUALISATION
# ==============================================================================
print("\n🎨 Génération des graphiques...")

fig, axes = plt.subplots(3, 1, figsize=(12, 16))

# GRAPHIQUE 1 : LES TAUX D'INTÉRÊT
major_rates = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD']
cols_to_plot = [c for c in major_rates if c in rates_pct.columns]

if cols_to_plot:
    rates_pct[cols_to_plot].plot(ax=axes[0], linewidth=1.5)
    axes[0].set_title("1. Évolution des Taux Directeurs (FRED / OCDE)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Taux (%)")
    axes[0].legend(loc='upper left', ncol=2)
    axes[0].grid(True, alpha=0.3)

# GRAPHIQUE 2 : PRIX FOREX NORMALISÉS (Base 100)
pairs_to_plot = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X']
valid_pairs = [p for p in pairs_to_plot if p in prices.columns]

if valid_pairs:
    normalized_prices = (prices[valid_pairs] / prices[valid_pairs].iloc[0]) * 100
    normalized_prices.plot(ax=axes[1], linewidth=1.5)
    axes[1].set_title("2. Performance Relative des Paires (Base 100)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Base 100")
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

# GRAPHIQUE 3 : CARRY SPREAD (CORRIGÉ)
if 'USD' in rates_pct.columns and 'JPY' in rates_pct.columns:
    spread = rates_pct['USD'] - rates_pct['JPY']
    
    # Correction ici : On utilise plot simple + fill_between manuel
    axes[2].plot(spread.index, spread.values, color='darkgreen', linewidth=1.5, label='Spread USD - JPY')
    
    # Remplissage vert si positif, rouge si négatif (optionnel, mais joli)
    axes[2].fill_between(spread.index, 0, spread, where=(spread >= 0), color='green', alpha=0.3, interpolate=True)
    axes[2].fill_between(spread.index, 0, spread, where=(spread < 0), color='red', alpha=0.3, interpolate=True)
    
    axes[2].set_title("3. Le 'Moteur' du Carry Trade : Différentiel USD vs JPY (%)", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Différentiel (%)")
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Vérification terminée. Affichage des graphiques.")