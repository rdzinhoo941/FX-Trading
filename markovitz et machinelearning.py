import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PARAMÈTRES & UNIVERS ÉLARGI
# ==============================================================================
TARGET_BUDGET = 1000000.0   
RISK_AVERSION = 2.5     

# TA LISTE COMPLÈTE (Majeurs + Cross + Exotiques)
pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", # Majeurs
    "EURJPY", "EURGBP", "EURCHF", "AUDJPY", "NZDJPY", # Cross
    "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDBRL", "USDTRY", "USDINR", "USDSGD" # Exotiques / Yield
]

tickers = [f"{pair}=X" for pair in pairs_list]
benchmark_ticker = "DX-Y.NYB"
start_date = "2010-01-01"
end_date = "2025-01-01"

print("█"*80)
print(f"🚀 DÉMARRAGE 'GLOBAL MACRO UNIVERSE' ({len(pairs_list)} Paires) | Budget: {TARGET_BUDGET:,.0f} $")
print("█"*80)

# ==============================================================================
# 2. TÉLÉCHARGEMENT DONNÉES PRIX
# ==============================================================================
print("\n📥 1. Téléchargement des Prix (Yahoo Finance)...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close'].ffill().dropna()

print("📥 2. Téléchargement du Benchmark (Dollar Index)...")
bench_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Close'].ffill().dropna()

# ==============================================================================
# 3. TÉLÉCHARGEMENT TAUX D'INTÉRÊT (FRED - MAJEURS & EXOTIQUES)
# ==============================================================================
print("📥 3. Téléchargement des Taux d'Intérêt (FRED - Central Bank & Interbank Rates)...")

# Dictionnaire complet pour couvrir ton univers
fred_codes = {
    # Majeurs (Interbank 3M)
    'USD': 'IR3TIB01USM156N', 'EUR': 'IR3TIB01EZM156N', 'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N', 'CHF': 'IR3TIB01CHM156N', 'AUD': 'IR3TIB01AUM156N',
    'CAD': 'IR3TIB01CAM156N', 'NZD': 'IR3TIB01NZM156N',
    
    # Exotiques & Scandies (Policy Rates / Short Term Rates)
    'SEK': 'IR3TIB01SEM156N', # Suède
    'NOK': 'IR3TIB01NOM156N', # Norvège
    'MXN': 'INTDSRMXM193N',   # Mexique (Taux Directeur)
    'ZAR': 'INTDSRZAM193N',   # Afrique du Sud
    'BRL': 'INTDSRBRM193N',   # Brésil
    'TRY': 'INTDSRTRM193N',   # Turquie (Crucial pour le Carry !)
    'INR': 'INTDSRINM193N',   # Inde
    'SGD': 'INTDSRSGM193N'    # Singapour
}

try:
    # Téléchargement
    rates_df = web.DataReader(list(fred_codes.values()), 'fred', start_date, end_date)
    # Mapping des colonnes (Code FRED -> Symbole Devise)
    inv_map = {v: k for k, v in fred_codes.items()}
    rates_df.rename(columns=inv_map, inplace=True)
    
    # Conversion % -> Décimal et remplissage des trous (fréquence mensuelle -> journalière)
    rates_daily = (rates_df / 100.0).resample('D').ffill().reindex(data.index).ffill()
    
    # Gestion des devises manquantes (si FRED échoue sur certaines)
    for curr in fred_codes.keys():
        if curr not in rates_daily.columns:
            rates_daily[curr] = 0.0
            
    print("   ✅ Taux FRED récupérés (Majeurs + Exotiques).")
    
except Exception as e:
    print(f"   ⚠️ Erreur partielle FRED: {e}. Carry Trade limité.")
    rates_daily = pd.DataFrame(0.0, index=data.index, columns=fred_codes.keys())

# ==============================================================================
# 4. CALCUL RENDEMENTS NETS (PRIX + CARRY)
# ==============================================================================
print("⚙️ Calcul des rendements Totaux (Prix + Différentiel Taux)...")
returns_total = pd.DataFrame(index=data.index, columns=data.columns)
price_returns = data.pct_change()

for t in tickers:
    # Nettoyage du ticker (USDTRY=X -> Base:USD, Quote:TRY)
    clean_t = t.replace('=X', '')
    
    # Logique spéciale pour identifier Base et Quote
    # Yahoo met souvent la devise forte en premier, mais pas toujours.
    # On assume le standard ISO : EURUSD, USDJPY, USDTRY...
    base = clean_t[:3]
    quote = clean_t[3:]
    
    r_price = price_returns[t]
    r_carry = 0.0
    
    if base in rates_daily.columns and quote in rates_daily.columns:
        # Différentiel annuel ramené au jour (252 jours ouvrés)
        r_carry = (rates_daily[base] - rates_daily[quote]) / 252
    
    returns_total[t] = r_price + r_carry

returns_total.dropna(inplace=True)

# ==============================================================================
# 5. FEATURE ENGINEERING (ML)
# ==============================================================================
print("🧠 Génération des Features (RSI, Vol, Trend)...")

def create_features(prices, rets):
    feats = {}
    for t in prices.columns:
        df = pd.DataFrame(index=prices.index)
        p = prices[t]
        r = rets[t]
        
        # Momentum
        df['Ret_5d'] = p.pct_change(5).shift(1)
        df['Ret_21d'] = p.pct_change(21).shift(1)
        
        # Volatilité
        df['Vol_21d'] = r.rolling(21).std().shift(1)
        
        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = (100 - (100 / (1 + gain/loss))).shift(1)
        
        # Distance Moyenne Mobile (Mean Reversion)
        sma50 = p.rolling(50).mean().shift(1)
        df['Dist_SMA50'] = (p.shift(1) / sma50) - 1
        
        feats[t] = df.dropna()
    return feats

features_all = create_features(data, returns_total)

# ==============================================================================
# 6. FONCTIONS (XGBOOST, HURST, MARKOWITZ)
# ==============================================================================

def calculate_hurst_complex(ts, max_lag=20):
    ts = ts.dropna().values
    if len(ts) < 50: return 0.5
    lags = range(2, min(max_lag, len(ts)//2))
    tau = []
    for lag in lags:
        # Simple R/S proxy via Standard Deviation of differences
        pp = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.std(pp))
    if len(tau) < 2: return 0.5
    try:
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return m[0] * 2.0 # Approximation Hurst
    except:
        return 0.5

def predict_returns_xgb(curr_date):
    preds = []
    train_start = curr_date - pd.DateOffset(years=2)
    
    for t in tickers:
        if t not in features_all: 
            preds.append(0.0)
            continue
            
        df = features_all[t]
        mask = (df.index >= train_start) & (df.index < curr_date)
        X_train = df.loc[mask]
        y_train = returns_total[t].reindex(X_train.index)
        
        if len(X_train) < 50 or curr_date not in df.index:
            preds.append(0.0)
            continue
            
        model = xgb.XGBRegressor(n_estimators=40, max_depth=3, learning_rate=0.05, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(df.loc[[curr_date]])[0]
        preds.append(pred)
    return np.array(preds)

def optimize_markowitz_sport(mu, sigma, risk_aversion, max_lev=1.0):
    n = len(mu)
    def objective(w):
        ret = np.dot(w, mu) * 52
        vol = np.sqrt(np.dot(w.T, np.dot(sigma, w))) * np.sqrt(52)
        penalty = 0.05 * np.sum(w**2) 
        return -(ret - (risk_aversion/2)*vol**2 - penalty)
    
    cons_net = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    # Levier Global Max : 3.0x (Mode Sport)
    GLOBAL_MAX = 3.0
    actual_limit = min(GLOBAL_MAX, max_lev)
    cons_lev = {'type': 'ineq', 'fun': lambda x: actual_limit - np.sum(np.abs(x))}
    
    constraints = [cons_net, cons_lev]
    # Diversification forcée (Max 40% par actif) pour éviter le suicide sur USDTRY
    bounds = tuple((-0.4, 0.4) for _ in range(n))
    
    try:
        init_guess = np.repeat(1/n, n)
        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
    except:
        return np.repeat(1/n, n)

# ==============================================================================
# 7. BACKTEST (2023-2025)
# ==============================================================================
print("\n🤖 Lancement Backtest sur UNIVERS ÉLARGI (Mode Sport : Levier 3x)...")

test_start_date = "2023-01-01"
rebal_dates = returns_total.loc[test_start_date:].resample('W-FRI').last().index

capital = TARGET_BUDGET
history_values = []
history_dates = []

for i in range(len(rebal_dates) - 1):
    curr_date = rebal_dates[i]
    next_date = rebal_dates[i+1]
    
    # 1. IA Prediction
    mu_pred = predict_returns_xgb(curr_date)
    
    # 2. Hurst (Régime sur PRIX pur)
    hursts = []
    for t in tickers:
        series = np.log(data.loc[:curr_date, t].tail(100))
        hursts.append(calculate_hurst_complex(series))
    avg_hurst = np.nanmean(hursts)
    
    # 3. Réglage Levier Dynamique
    lev_target = 2.0
    regime = "NEUTRAL"
    if avg_hurst > 0.6: 
        lev_target = 3.0
        regime = "TRENDING"
    elif avg_hurst < 0.45:
        lev_target = 1.0
        regime = "MEAN-REV"
        
    # 4. Optimisation
    sigma = returns_total.loc[:curr_date].tail(120).cov()
    w = optimize_markowitz_sport(mu_pred, sigma, RISK_AVERSION, max_lev=lev_target)
    
    # 5. Simulation
    week_ret = returns_total.loc[curr_date:next_date].dot(w).sum()
    capital *= (1 + week_ret)
    
    history_values.append(capital)
    history_dates.append(next_date)
    
    # 6. Logs (Mensuel)
    if i % 4 == 0:
        tot_lev = np.sum(np.abs(w))
        print(f"\n📅 {curr_date.date()} | Cap: {capital:,.0f} $ | Hurst: {avg_hurst:.2f} ({regime})")
        print(f"   🔥 Levier: {tot_lev:.2f}x (Max: {lev_target}x)")
        
        # Affichage Top Convictions
        sorted_idx = np.argsort(w)
        tops = []
        for idx in np.concatenate([sorted_idx[:2], sorted_idx[-2:][::-1]]):
            if abs(w[idx]) > 0.05:
                # Affichage du Carry Annuel Estimé pour la paire
                t = tickers[idx]
                pair_base, pair_quote = t.replace('=X','')[:3], t.replace('=X','')[3:]
                carry = 0.0
                if pair_base in rates_daily.columns:
                    carry = rates_daily.loc[curr_date, pair_base] - rates_daily.loc[curr_date, pair_quote]
                
                # Sens du trade
                side = "🟢" if w[idx] > 0 else "🔴" # Long ou Short
                # Si on short, le carry est inversé
                net_carry = carry if w[idx] > 0 else -carry
                
                tops.append(f"{side}{t[:-2]}:{w[idx]:.0%} (Carry:{net_carry:.1%})")
                
        print(f"   ⚖️ {', '.join(tops)}")

# ==============================================================================
# 8. RÉSULTATS FINAUX
# ==============================================================================
df_res = pd.DataFrame({'Global Macro AI': history_values}, index=history_dates)
bench_rebased = bench_data.reindex(history_dates).ffill()
bench_rebased = bench_rebased / bench_rebased.iloc[0] * TARGET_BUDGET
df_res['Dollar Index'] = bench_rebased

print("\n" + "="*80)
print(f"RÉSULTAT FINAL : {capital:,.2f} $")
print(f"PERFORMANCE    : {(capital/TARGET_BUDGET - 1):.2%}")
print("="*80)

plt.figure(figsize=(12, 6))
plt.plot(df_res['Global Macro AI'], label='AI Markowitz (20 Paires)', color='blue', linewidth=1.5)
plt.plot(df_res['Dollar Index'], label='Dollar Index', color='orange', linestyle='--', alpha=0.7)
plt.title(f"Performance Global Macro AI (Majeurs + Exotiques)")
plt.ylabel("Valeur Portefeuille ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()