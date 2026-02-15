import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
START_DATE = "2010-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d") # Aujourd'hui

# Création du dossier 'data' si il n'existe pas
if not os.path.exists('data'):
    os.makedirs('data')

# --- LISTE DES PAIRES FOREX ---
pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURJPY", "EURGBP", "AUDJPY", "NZDJPY",
    "USDMXN", "USDTRY", "USDZAR", "USDSGD", "USDNOK", "USDSEK"
]
tickers = [f"{pair}=X" for pair in pairs_list]

# --- LISTE DES CODES FRED (OCDE & FED) ---
# On utilise les codes les plus robustes pour l'historique long
fred_map = {
    'USD': 'DFF',             # Daily Federal Funds Rate
    'EUR': 'IRSTCI01EZM156N', # Euro Area Immediate Rates
    'JPY': 'IRSTCI01JPM156N', # Japan Immediate Rates
    'GBP': 'IRSTCI01GBM156N', # UK Immediate Rates (Plus d'historique que SONIA)
    'CHF': 'IRSTCI01CHM156N', # Switzerland Immediate Rates
    'AUD': 'IRSTCI01AUM156N', # Australia Immediate Rates
    'CAD': 'IRSTCI01CAM156N', # Canada Immediate Rates
    'NZD': 'IRSTCI01NZM156N', # New Zealand Immediate Rates
    'MXN': 'IRSTCI01MXM156N', # Mexico Immediate Rates
    'ZAR': 'IRSTCI01ZAM156N', # South Africa Immediate Rates
    'TRY': 'IRSTCI01TRM156N', # Turkey Immediate Rates
    'NOK': 'IRSTCI01NOM156N', # Norway Immediate Rates
    'SEK': 'IRSTCI01SEM156N', # Sweden Immediate Rates
}

print("="*80)
print(f"📥 MODULE DE TÉLÉCHARGEMENT DE DONNÉES ({START_DATE} -> {END_DATE})")
print("="*80)

# ==============================================================================
# 2. TÉLÉCHARGEMENT DES PRIX FOREX
# ==============================================================================
print(f"\n[1/3] Téléchargement des Prix Forex ({len(tickers)} paires)...")
try:
    prices = yf.download(tickers, start=START_DATE, end=END_DATE, progress=True)['Close']
    
    # Nettoyage : On remplit les weekends/jours fériés avec la valeur précédente
    prices = prices.ffill().dropna(how='all')
    
    # Sauvegarde
    file_prices = os.path.join('data', 'data_forex_prices.csv')
    prices.to_csv(file_prices)
    print(f"   ✅ Prix sauvegardés dans : {file_prices}")
    print(f"   📊 Dimensions : {prices.shape}")

except Exception as e:
    print(f"   ❌ Erreur lors du téléchargement Yahoo: {e}")

# ==============================================================================
# 3. TÉLÉCHARGEMENT DES TAUX FRED
# ==============================================================================
print(f"\n[2/3] Téléchargement des Taux Directeurs FRED ({len(fred_map)} devises)...")

rates_df = pd.DataFrame()
failed_rates = []

for curr, code in fred_map.items():
    print(f"   ... Récupération {curr} ({code})")
    try:
        # Téléchargement via FRED
        df = web.DataReader(code, 'fred', START_DATE, END_DATE)
        
        # Renommage colonne
        df.columns = [curr]
        
        # Fusion
        if rates_df.empty:
            rates_df = df
        else:
            rates_df = rates_df.join(df, how='outer')
            
    except Exception as e:
        print(f"      ⚠️ Échec pour {curr}: {e}")
        failed_rates.append(curr)

# ==============================================================================
# 4. NETTOYAGE ET SAUVEGARDE DES TAUX
# ==============================================================================
print(f"\n[3/3] Traitement et Sauvegarde des Taux...")

if not rates_df.empty:
    # 1. Remplissage des fréquences (certains taux sont mensuels, d'autres journaliers)
    # On resample en journalier ('D') et on propage la dernière valeur connue (ffill)
    rates_daily = rates_df.resample('D').ffill()
    
    # 2. On aligne les dates sur celles du Forex (Optionnel mais propre)
    # On garde toutes les dates pour avoir un historique complet
    rates_daily = rates_daily.ffill() 
    
    # 3. Conversion en Décimal (Les données brutes sont en %, ex: 5.0 pour 5%)
    # On divise par 100 pour avoir 0.05
    rates_decimal = rates_daily / 100.0
    
    # Sauvegarde
    file_rates = os.path.join('data', 'data_fred_rates.csv')
    rates_decimal.to_csv(file_rates)
    
    print(f"   ✅ Taux sauvegardés dans : {file_rates}")
    print(f"   📊 Dimensions : {rates_decimal.shape}")
    print(f"   📝 Note : Les taux sont sauvegardés en format DÉCIMAL (ex: 0.05 pour 5%)")
    
    if failed_rates:
        print(f"   ⚠️ Attention, taux manquants pour : {failed_rates}")
else:
    print("   ❌ Aucune donnée de taux récupérée.")

print("\n" + "="*80)
print("🏁 TÉLÉCHARGEMENT TERMINÉ AVEC SUCCÈS")
print("="*80)