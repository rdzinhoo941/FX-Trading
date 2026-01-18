import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import itertools

# --- 1. CONFIGURATION FOREX ---
TARGET_BUDGET = 1000.0  
RISK_AVERSION = 2.5     
TAU = 0.05              
CONFIDENCE_LEVEL = 0.95 

# Liste des paires fournies (Format Yahoo Finance avec =X)
pairs_list_raw = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD",
    "USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"
]
FOREX_TICKERS = [f"{p}=X" for p in pairs_list_raw]
BENCHMARK_TICKER = "DX-Y.NYB" # US Dollar Index (Standard de marché pour le Forex)

print(f"1. CONFIG : BL + SIM + CVaR | Mode: FOREX | Budget ~{TARGET_BUDGET}€")

# --- 2. DONNÉES HISTORIQUES ---
print("2. TÉLÉCHARGEMENT DONNÉES FOREX...")
all_tickers = FOREX_TICKERS + [BENCHMARK_TICKER]
try:
    # On prend large pour avoir de l'historique
    data = yf.download(all_tickers, start="2002-01-01", progress=False)['Close'].ffill()
    # Nettoyage : Suppression des colonnes vides si certaines paires n'existent pas depuis 2000
    data = data.dropna(axis=1, how='all')
    
    # Séparation Bench / Paires
    if BENCHMARK_TICKER in data.columns:
        prices_bench = data[BENCHMARK_TICKER]
        data = data.drop(columns=[BENCHMARK_TICKER])
    else:
        # Fallback si le Dollar Index bug: on prend l'EURUSD comme proxy marché inverse ou moyenne
        print("Warning: Dollar Index introuvable, utilisation Moyenne Simple comme Bench.")
        prices_bench = data.mean(axis=1)

    # On s'assure que toutes les paires demandées sont là (si téléchargées)
    valid_tickers = [t for t in FOREX_TICKERS if t in data.columns]
    data = data[valid_tickers]
    
    last_date = data.index[-1]
    print(f"   Données OK : {len(valid_tickers)} paires chargées jusqu'au {last_date.date()}")
except Exception as e:
    print(f"Erreur Download: {e}")
    exit()

# --- 3. FONCTIONS MATHÉMATIQUES (INCHANGÉES) ---

def get_covariance_sim(asset_returns, benchmark_returns):
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10: return None 
    
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    
    var_mkt = x.var()
    if var_mkt < 1e-6: return asset_returns.cov() 
    
    covs_xy = y.apply(lambda col: col.cov(x))
    betas = covs_xy / var_mkt
    
    beta_outer = np.outer(betas, betas) * var_mkt
    
    var_assets = y.var()
    idio_vars = var_assets - (betas ** 2) * var_mkt
    idio_vars = np.maximum(idio_vars, 0) 
    
    sim_cov = beta_outer + np.diag(idio_vars)
    return pd.DataFrame(sim_cov, index=asset_returns.columns, columns=asset_returns.columns)

def optimize_black_litterman(prices_train, prices_bench_train):
    valid = prices_train.dropna(axis=1, how='any')
    if valid.shape[1] < 2: return None
    
    returns = valid.pct_change(fill_method=None).dropna()
    if returns.empty: return None

    returns = returns.loc[:, returns.var() > 1e-6]
    if returns.shape[1] < 2: return None 

    n_observations = returns.shape[0]
    tau_dynamic = 1 / n_observations 
    
    bench_slice = prices_bench_train.loc[returns.index]
    bench_rets = bench_slice.pct_change(fill_method=None).dropna()
    
    if bench_rets.empty:
        cov_mat_annual = returns.cov() * 252
        risk_aversion_dyn = 2.5
    else:
        mkt_ret = bench_rets.mean() * 252
        mkt_var = bench_rets.var() * 252
        if mkt_var == 0: mkt_var = 1e-6
        raw_lambda = (mkt_ret - 0.01) / mkt_var # Risk free rate plus bas sur Forex
        risk_aversion_dyn = np.clip(raw_lambda, 1.0, 10.0)
        
        cov_mat_annual = get_covariance_sim(returns, bench_rets) * 252
        if cov_mat_annual is None: 
            cov_mat_annual = returns.cov() * 252

    mu_historical = returns.mean() * 252 
    n_assets = len(mu_historical)
    
    jitter = 1e-5 * np.eye(n_assets)
    cov_mat_safe = cov_mat_annual + jitter
    
    weights_eq = np.array([1/n_assets] * n_assets).reshape(-1, 1)
    pi = risk_aversion_dyn * cov_mat_safe.dot(weights_eq)
    
    Q = mu_historical.values.reshape(-1, 1)
    P = np.identity(n_assets)
    omega = np.dot(np.dot(P, tau_dynamic * cov_mat_safe), P.T) * np.eye(n_assets)
    
    try:
        tau_cov_inv = np.linalg.pinv(tau_dynamic * cov_mat_safe)
        omega_inv = np.linalg.pinv(omega)
        M1 = np.linalg.pinv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
        M2 = np.dot(tau_cov_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
        bl_returns = np.dot(M1, M2).flatten()
    except Exception:
        return None
    
    def neg_sharpe_bl(w):
        p_ret = np.sum(bl_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_safe, w))) 
        if p_vol < 1e-6: return 0 
        return - (p_ret) / p_vol

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    try:
        res = minimize(neg_sharpe_bl, [1/n_assets]*n_assets, method='SLSQP', bounds=bounds, constraints=cons)
        return pd.Series(res.x, index=returns.columns).sort_values(ascending=False)
    except:
        return None

def clean_portfolio(allocations, threshold):
    if allocations is None: return None
    clean = allocations[allocations >= threshold]
    if clean.empty: return allocations 
    return clean / clean.sum()

def calculate_risk_metrics(returns_series, confidence=0.95):
    if returns_series.empty: return np.nan, np.nan
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    if np.isnan(cvar): cvar = var 
    return var, cvar

# --- 4. GRID SEARCH HYPER-PARAMÈTRES (2010-2020) ---

def backtest_parameters(window_y, threshold_pct, start_y=2005, end_y=2022):
    """ Simule la stratégie sur une période donnée et renvoie le Rendement Total """
    capital = 100.0
    current_year = start_y
    
    # On boucle sur les années
    while current_year <= end_y:
        test_start = f"{current_year}-01-01"
        test_end = f"{current_year}-12-31"
        
        test_start_dt = pd.Timestamp(test_start)
        start_train = test_start_dt - pd.DateOffset(months=int(window_y * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        # Données
        train_slice = data.loc[start_train:end_train]
        bench_slice = prices_bench.loc[start_train:end_train]
        test_slice = data.loc[test_start:test_end]
        
        if test_slice.empty: break
        
        # Optim
        alloc = optimize_black_litterman(train_slice, bench_slice)
        alloc = clean_portfolio(alloc, threshold_pct)
        
        if alloc is not None:
            daily_returns = test_slice[alloc.index].pct_change(fill_method=None).dropna()
            port_ret = daily_returns.dot(alloc.values)
            year_perf = (1 + port_ret).prod() - 1
            capital *= (1 + year_perf)
        
        current_year += 1
        
    return capital # Capital final

print("\n" + "█"*80)
print("3. RECHERCHE DES MEILLEURS PARAMÈTRES (GRID SEARCH 2005-2022)")
print("   Objectif : Maximiser la performance sur 10 ans de Forex")
print("█"*80)

# Grille de paramètres à tester
windows_test = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9]     # 6 mois, 1 an, 2 ans, 3.5 ans
filters_test = [0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2] # 0%, 5%, 10%, 15% (Seuil Conviction)

best_score = -999
best_params = (3.5, 0.10) # Valeurs par défaut

print(f"{'WINDOW (Ans)':<15} | {'FILTRE (%)':<15} | {'CAPITAL FINAL (Base 100)':<25}")
print("-" * 65)

for w in windows_test:
    for f in filters_test:
        try:
            final_cap = backtest_parameters(w, f, 2005, 2022)
            print(f"{w:<15} | {f*100:<15.0f} | {final_cap:.2f} €")
            
            if final_cap > best_score:
                best_score = final_cap
                best_params = (w, f)
        except Exception as e:
            continue

print("-" * 65)
print(f"🏆 VAINQUEUR : Window={best_params[0]} ans | Filtre={best_params[1]*100:.0f}%")
print(f"   (Ces paramètres seront utilisés pour la suite)")

# Application des paramètres gagnants
WINDOW_YEARS = best_params[0]
SEUIL_FILTRE = best_params[1]

# --- 5. BACKTEST VISUEL ENGINE (Adapté) ---

def run_scenario(scenario_id, name, start_year, end_year):
    print(f"\n[{scenario_id}] {name}...")
    periods = []
    current_year = start_year
    while current_year <= end_year:
        periods.append((f"{current_year}-01-01", f"{current_year}-12-31"))
        current_year += 1
            
    cols = 3
    rows = math.ceil(len(periods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.canvas.manager.set_window_title(name)
    fig.suptitle(name, fontsize=14, fontweight='bold', color='darkblue')
    axs_flat = axes.flatten()
    plot_idx = 0
    
    cap_strat = 100.0
    cap_bench = 100.0
    risk_audit = [] 
    
    print(f"{'ANNÉE':<6} | {'BENCH':<8} | {'STRAT':<8} | {'VaR 95%':<8} | {'CVaR 95%':<9} | {'TOP POSITIONS':<30}")
    print("-" * 120)
    
    for start_test, end_test in periods:
        bench_slice = prices_bench.loc[start_test:end_test]
        if bench_slice.empty: continue
        
        b_rets = bench_slice.pct_change(fill_method=None).dropna()
        per_bench = (1 + b_rets).prod() - 1
        cap_bench *= (1 + per_bench)
        
        test_start_dt = pd.Timestamp(start_test)
        start_train = test_start_dt - pd.DateOffset(months=int(WINDOW_YEARS * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        # Données Training
        train_data = data.loc[start_train:end_train]
        bench_train = prices_bench.loc[start_train:end_train]
        
        # APPEL A L'OPTIMISEUR BL
        alloc = optimize_black_litterman(train_data, bench_train)
        
        if alloc is not None:
            alloc = clean_portfolio(alloc, SEUIL_FILTRE)
        
        period_strat_curve = []
        period_bench_curve = []
        period_dates = []
        
        var_forecast = np.nan
        cvar_forecast = np.nan
        top_holdings_str = "CASH"
        
        if alloc is not None:
            top3 = alloc.head(3)
            # Affichage nettoyé du suffixe =X pour la lisibilité
            top_holdings_str = ", ".join([f"{t.replace('=X','')} {w*100:.0f}%" for t, w in top3.items()])
            
            # Prévision Risque
            train_prices_alloc = train_data[alloc.index]
            train_rets_alloc = train_prices_alloc.pct_change(fill_method=None).dropna()
            train_port_rets = train_rets_alloc.dot(alloc.values)
            var_forecast, cvar_forecast = calculate_risk_metrics(train_port_rets, CONFIDENCE_LEVEL)
            
            # Test Réel
            test_prices = data.loc[start_test:end_test, alloc.index]
            s_rets = test_prices.pct_change(fill_method=None).dropna()
            common_idx = s_rets.index.intersection(b_rets.index)
            
            if len(common_idx) > 0:
                s_rets = s_rets.loc[common_idx]
                b_rets_aligned = b_rets.loc[common_idx]
                daily_perf = s_rets.dot(alloc.values)
                
                # Audit Risk
                worst_loss = daily_perf.min()
                breach_var = daily_perf[daily_perf < var_forecast]
                pct_breach = len(breach_var) / len(daily_perf) if len(daily_perf) > 0 else 0
                
                year_start = int(start_test[:4])
                if year_start >= 2021:
                    risk_audit.append({
                        "Period": f"{start_test[:4]}",
                        "Pred_VaR": var_forecast,
                        "Pred_CVaR": cvar_forecast,
                        "Real_Worst": worst_loss,
                        "Breach_Pct": pct_breach,
                        "Status": "FAIL" if pct_breach > (1-CONFIDENCE_LEVEL) else "OK"
                    })

                per_strat = (1 + daily_perf).prod() - 1
                cap_strat *= (1 + per_strat)
                
                curve_s = (1 + daily_perf).cumprod() * 100
                curve_b = (1 + b_rets_aligned).cumprod() * 100
                period_strat_curve = curve_s.tolist()
                period_bench_curve = curve_b.tolist()
                period_dates = common_idx

        per_bench_abs = per_bench
        per_strat_abs = per_strat if alloc is not None else 0
        sign = "✅" if per_strat_abs > per_bench_abs else "🔻"
        
        print(f"{start_test[:4]:<6} | {per_bench_abs*100:+.1f}%   | {per_strat_abs*100:+.1f}% {sign} | {var_forecast*100:.1f}%    | {cvar_forecast*100:.1f}%     | {top_holdings_str}")
        
        if plot_idx < len(axs_flat) and len(period_strat_curve) > 0:
            ax = axs_flat[plot_idx]
            ax.plot(period_dates, period_strat_curve, color='blue', linewidth=1.5)
            ax.plot(period_dates, period_bench_curve, color='gray', linestyle='--', alpha=0.7)
            col_tit = 'blue' if per_strat_abs > per_bench_abs else 'black'
            ax.set_title(f"{start_test[:4]} : {per_strat_abs*100:+.1f}% vs {per_bench_abs*100:+.1f}%", color=col_tit, fontsize=9, fontweight='bold')
            ax.xaxis.set_visible(False)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
    for j in range(plot_idx, len(axs_flat)): fig.delaxes(axs_flat[j])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("-" * 120)
    print(f"BILAN : Stratégie {cap_strat:.0f} € vs Bench (USD Index) {cap_bench:.0f} €")
    
    if len(risk_audit) > 0:
        print("\n>>> AUDIT DE RISQUE AVANCÉ (VaR vs CVaR)")
        print(f"{'Année':<8} | {'VaR (Seuil)':<12} | {'CVaR (Crash)':<13} | {'Pire Jour':<12} | {'Analyse'}")
        for r in risk_audit:
            cvar_breach = "⚠️ >CVaR" if r['Real_Worst'] < r['Pred_CVaR'] else "OK"
            print(f"{r['Period']:<8} | {r['Pred_VaR']*100:.2f}%      | {r['Pred_CVaR']*100:.2f}%       | {r['Real_Worst']*100:.2f}%      | {r['Status']} / {cvar_breach}")
            
    print("=" * 120)

# --- 6. OPTIMISEUR ENTIER (LOGIQUE CONSERVÉE) ---
def smart_integer_optimizer(allocation, current_prices, target_budget):
    # Note: Sur le Forex, "shares" représente le nombre d'unités (ex: 1000 EURUSD)
    results = []
    tickers = allocation.index.tolist()
    target_weights = allocation.values
    ideal_amounts = target_weights * target_budget
    ideal_shares = ideal_amounts / current_prices[tickers].values
    
    ranges = []
    for share_count in ideal_shares:
        floor_s = math.floor(share_count)
        ceil_s = math.ceil(share_count)
        if floor_s == 0 and share_count > 0.1:
            ranges.append([1]) # Forcer au moins 1 unité
        else:
            ranges.append(list(set([floor_s, ceil_s])))
    
    # Si trop complexe, arrondi simple
    if len(ranges) > 12:
         shares = np.round(ideal_shares)
         total_cost = np.sum(shares * current_prices[tickers].values)
         return {"shares": shares, "cost": total_cost, "score": 0}

    for combination in itertools.product(*ranges):
        shares = np.array(combination)
        total_cost = np.sum(shares * current_prices[tickers].values)
        if total_cost < target_budget * 0.8 or total_cost > target_budget * 1.2: continue
        
        real_weights = (shares * current_prices[tickers].values) / total_cost
        weight_error = np.sum(np.abs(real_weights - target_weights))
        budget_error = abs(total_cost - target_budget) / target_budget
        score = (2.0 * weight_error) + (1.0 * budget_error)
        
        results.append({"shares": shares, "cost": total_cost, "score": score, "real_weights": real_weights})
        
    if not results: return None
    return sorted(results, key=lambda x: x["score"])[0]

# --- 7. EXÉCUTION ---
print("4. LANCEMENT DU BACKTEST AVEC PARAMÈTRES OPTIMISÉS...")

# On lance uniquement sur la période récente (2023-2025) car 2005-2022 a servi à l'entrainement des paramètres
scenarios_list = [
    {"id": 1, "years": (2023, 2025), "desc": f"FOREX | Window={WINDOW_YEARS}y | Filtre={SEUIL_FILTRE*100:.0f}%"}
]

for s in scenarios_list:
    run_scenario(s["id"], s["desc"], s["years"][0], s["years"][1])

# --- 8. PRÉVISIONS 2026 ---
print("\n" + "█"*80)
print(f"🔮 PRÉVISIONS 2026 : BL + SIM (Optimisé Forex)")
print("█"*80)

end_train_26 = last_date
start_train_26 = last_date - pd.DateOffset(months=int(WINDOW_YEARS * 12))
train_data_26 = data.loc[start_train_26:end_train_26]
bench_train_26 = prices_bench.loc[start_train_26:end_train_26]

# Appel Optimiseur BL
alloc_base = optimize_black_litterman(train_data_26, bench_train_26)

if alloc_base is not None:
    budget = TARGET_BUDGET
    alloc_conviction = clean_portfolio(alloc_base, SEUIL_FILTRE)
    
    if alloc_conviction is not None:
        
        # --- DASHBOARD DE RISQUE 2026 ---
        train_prices_alloc = train_data_26[alloc_conviction.index]
        train_port_26 = train_prices_alloc.pct_change(fill_method=None).dropna().dot(alloc_conviction.values)
        p_var_26, p_cvar_26 = calculate_risk_metrics(train_port_26, CONFIDENCE_LEVEL)
        p_vol_26 = train_port_26.std() * np.sqrt(252)
        p_ret_26 = train_port_26.mean() * 252
        
        print(f"\n📊 DASHBOARD RISQUE (Window {WINDOW_YEARS} ans)")
        print("-" * 50)
        print(f"   Rendement Espéré (Hist) : {p_ret_26*100:.1f}%")
        print(f"   Volatilité Annuelle     : {p_vol_26*100:.1f}%")
        print(f"   VaR 95% (Jour)          : {p_var_26*100:.2f}%")
        print(f"   CVaR 95% (Jour)         : {p_cvar_26*100:.2f}%")
        print("-" * 50)
        
        print(f"\nA. PORTEFEUILLE CIBLE")
        print("-" * 75)
        for t, w in alloc_conviction.items():
            price = data[t].iloc[-1]
            print(f"   👉 {t.replace('=X',''):<10} | {w*100:.1f}%      | {budget*w:.0f} €          | {price:.4f}")
    
        current_prices = data.iloc[-1][alloc_conviction.index]
        best_plan = smart_integer_optimizer(alloc_conviction, current_prices, TARGET_BUDGET)
        
        print("\n" + "█"*80)
        print(f"C. PLAN D'ACHAT (Unités) 2026 (Budget ~{TARGET_BUDGET}€)")
        print("█"*80)
        
        if best_plan:
            print(f"   Coût Total : {best_plan['cost']:.2f} €")
            print("-" * 80)
            print(f"{'PAIRE':<10} | {'COURS':<12} | {'UNITÉS':<10} | {'MONTANT':<12} | {'POIDS'}")
            print("-" * 80)
            
            tickers = alloc_conviction.index.tolist()
            shares = best_plan["shares"]
            
            for i, t in enumerate(tickers):
                n = int(shares[i])
                p = current_prices[t]
                amt = n * p
                w_real = amt / best_plan['cost']
                print(f"{t.replace('=X',''):<10} | {p:<10.4f}   | {n:<10} | {amt:<10.2f} € | {w_real*100:.1f}%")
            print("-" * 80)
        else:
            print("Budget trop serré.")
else:
    print("Erreur données ou Optimisation échouée (Pas assez de variance).")

plt.show()