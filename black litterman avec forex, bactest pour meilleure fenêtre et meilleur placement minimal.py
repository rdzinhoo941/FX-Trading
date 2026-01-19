import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math
import itertools

TARGET_BUDGET = 1000000.0  
RISK_AVERSION = 2.5     
TAU = 0.05              
CONFIDENCE_LEVEL = 0.95 

# Liste des paires MAJEURES uniquement (Plus stable, moins de frais cachés)
pairs_list_raw = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", 
    "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"
]
FOREX_TICKERS = [f"{p}=X" for p in pairs_list_raw]
BENCHMARK_TICKER = "DX-Y.NYB" 

print(f"1. CONFIG : BL Trend-Following | Mode: FOREX MAJORS | Budget ~{TARGET_BUDGET}€")

# --- 2. DONNÉES HISTORIQUES ---
print("2. TÉLÉCHARGEMENT DONNÉES FOREX...")
all_tickers = FOREX_TICKERS + [BENCHMARK_TICKER]
try:
    data = yf.download(all_tickers, start="2002-01-01", progress=False)['Close'].ffill()
    data = data.dropna(axis=1, how='all')
    
    if BENCHMARK_TICKER in data.columns:
        prices_bench = data[BENCHMARK_TICKER]
        data = data.drop(columns=[BENCHMARK_TICKER])
    else:
        print("Warning: Dollar Index introuvable, utilisation Moyenne Simple.")
        prices_bench = data.mean(axis=1)

    valid_tickers = [t for t in FOREX_TICKERS if t in data.columns]
    data = data[valid_tickers]
    last_date = data.index[-1]
    print(f"   Données OK : {len(valid_tickers)} paires chargées jusqu'au {last_date.date()}")
except Exception as e:
    print(f"Erreur Download: {e}")
    exit()

# --- 3. FONCTIONS MATHÉMATIQUES ---

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
    """
    Version LONG / SHORT TACTIQUE :
    - Si Tendance Hausse (Prix > Moyenne) : Vue Positive -> Achat
    - Si Tendance Baisse (Prix < Moyenne) : Vue Négative -> Vente (Short)
    """
    valid = prices_train.dropna(axis=1, how='any')
    if valid.shape[1] < 2: return None
    
    returns = valid.pct_change(fill_method=None).dropna()
    if returns.empty: return None

    # --- 1. PARAMÈTRES BL ---
    recent_returns = returns.tail(252) 
    cov_mat_annual = recent_returns.cov() * 252
    
    n_assets = len(returns.columns)
    risk_aversion = 2.5
    tau = 0.05
    
    # Equilibrium : On part neutre (0 partout) au lieu de 1/N
    # Car en Long/Short, le marché neutre n'est pas "tout acheter"
    weights_eq = np.zeros((n_assets, 1))
    pi = risk_aversion * cov_mat_annual.dot(weights_eq)

    # --- 2. VUES (Tendance Long/Short) ---
    current_prices = valid.iloc[-1]
    # Moyenne Mobile 100 jours pour la tendance
    sma = valid.rolling(window=100).mean().iloc[-1]
    
    pick_matrix = []   
    view_returns = [] 
    view_confidences = []

    for i, asset in enumerate(returns.columns):
        if pd.isna(sma[asset]): continue
        
        # Force du signal : (Prix - Moyenne) / Moyenne
        # Si positif = Hausse, Si négatif = Baisse
        trend_strength = (current_prices[asset] / sma[asset]) - 1
        
        # On ne joue que si la tendance est un minimum marquée (> 0.5% d'écart)
        if abs(trend_strength) > 0.005:
            p_row = np.zeros(n_assets)
            p_row[i] = 1 
            pick_matrix.append(p_row)
            
            # Projection : on espère capter ce momentum
            # On amplifie un peu le signal pour la vue (x2)
            expected_ret = np.clip(trend_strength * 2.0, -0.40, 0.40)
            view_returns.append(expected_ret)
            
            asset_vol = np.sqrt(cov_mat_annual.iloc[i, i])
            view_confidences.append((asset_vol * tau) ** 2)

    # --- 3. CALCUL BL ---
    if len(pick_matrix) > 0:
        P = np.array(pick_matrix)
        Q = np.array(view_returns).reshape(-1, 1)
        Omega = np.diag(view_confidences)
        
        try:
            tau_cov = tau * cov_mat_annual
            inv_tau_cov = np.linalg.inv(tau_cov + np.eye(n_assets)*1e-6)
            inv_omega = np.linalg.inv(Omega)
            
            left = inv_tau_cov + P.T.dot(inv_omega).dot(P)
            minv = np.linalg.inv(left + np.eye(n_assets)*1e-6)
            right = inv_tau_cov.dot(pi) + P.T.dot(inv_omega).dot(Q)
            
            bl_returns = minv.dot(right).flatten()
        except:
            bl_returns = pi.flatten()
    else:
        # Pas de tendance claire nulle part -> Cash
        return None

    # --- 4. OPTIMISATION LONG/SHORT ---
    def neg_sharpe(w):
        p_ret = np.sum(bl_returns * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_annual, w)))
        if p_vol < 1e-6: return 0
        return - (p_ret) / p_vol

    # Contrainte : Somme des valeurs ABSOLUES = 1 (Gross Exposure = 100%)
    # Cela veut dire qu'on utilise tout le capital, soit en achat, soit en short.
    # Ex: 50% Long EURUSD + 50% Short GBPUSD = 100% exposé.
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
    
    # Bornes : On peut aller de -100% (Short total) à +100% (Long total) par actif
    bounds = tuple((-1.0, 1.0) for _ in range(n_assets)) 
    
    try:
        # Initial guess : on suit le signe des retours espérés
        # Si BL prévoit positif -> on commence positif, sinon négatif
        init_guess = np.sign(bl_returns) * (1/n_assets)
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        return pd.Series(res.x, index=returns.columns).sort_values(ascending=False)
    except:
        return None
    
def clean_portfolio(allocations, threshold):
    if allocations is None: return None
    # On garde si la valeur ABSOLUE est supérieure au seuil
    clean = allocations[allocations.abs() >= threshold]
    if clean.empty: return None
    
    # Renormalisation pour que la somme des valeurs absolues soit 1
    # (Gross leverage = 1)
    total_exposure = clean.abs().sum()
    if total_exposure == 0: return None
    
    return clean / total_exposure

def calculate_risk_metrics(returns_series, confidence=0.95):
    if returns_series.empty: return np.nan, np.nan
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    if np.isnan(cvar): cvar = var 
    return var, cvar
# --- 4. GRID SEARCH HYPER-PARAMÈTRES (2010-2022) ---

def backtest_parameters(window_y, threshold_pct, start_y=2010, end_y=2022):
    """ Simule la stratégie avec prise en compte du Spread (Frais) """
    capital = 100.0
    current_year = start_y
    
    while current_year <= end_y:
        test_start = f"{current_year}-01-01"
        test_end = f"{current_year}-12-31"
        test_start_dt = pd.Timestamp(test_start)
        start_train = test_start_dt - pd.DateOffset(months=int(window_y * 12))
        end_train = test_start_dt - pd.DateOffset(days=1)
        
        train_slice = data.loc[start_train:end_train]
        bench_slice = prices_bench.loc[start_train:end_train]
        test_slice = data.loc[test_start:test_end]
        
        if test_slice.empty: break
        
        # Optimisation
        alloc = optimize_black_litterman(train_slice, bench_slice)
        alloc = clean_portfolio(alloc, threshold_pct)
        
        if alloc is not None:
            daily_returns = test_slice[alloc.index].pct_change(fill_method=None).dropna()
            port_ret = daily_returns.dot(alloc.values)
            
            # --- REALISME : COUT DE TRANSACTION (SPREAD) ---
            # On simule un coût de 0.05% à l'entrée en position (début d'année)
            if not port_ret.empty:
                port_ret.iloc[0] -= 0.0005 

            year_perf = (1 + port_ret).prod() - 1
            capital *= (1 + year_perf)
        
        current_year += 1
        
    return capital

print("\n" + "█"*80)
print("3. RECHERCHE DES MEILLEURS PARAMÈTRES (GRID SEARCH 2010-2022)")
print("   Objectif : Maximiser la performance Réaliste (Net de frais)")
print("█"*80)

# Nouvelle grille : Plus courte (0.5 an à 3 ans) car l'algo suit la tendance
windows_test = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,2.5, 2.75,3.0,3.25,3.5,3.75,4.25,4.5,4.75,5,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8] 
filters_test = [0.00] # On garde 0% pour laisser le BL décider

best_score = -999
best_params = (1.5, 0.00) 

print(f"{'WINDOW (Ans)':<15} | {'FILTRE (%)':<15} | {'CAPITAL FINAL (Base 100)':<25}")
print("-" * 65)

for w in windows_test:
    for f in filters_test:
        try:
            final_cap = backtest_parameters(w, f, 2010, 2022)
            print(f"{w:<15} | {f*100:<15.0f} | {final_cap:.2f} €")
            
            if final_cap > best_score:
                best_score = final_cap
                best_params = (w, f)
        except Exception as e:
            continue

print("-" * 65)
print(f"🏆 VAINQUEUR : Window={best_params[0]} ans | Filtre={best_params[1]*100:.0f}%")
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
        if alloc is None:
            # CAS 1 : Tout baisse, on reste en CASH (Performance = 0%)
            daily_perf = pd.Series(0.0, index=b_rets.index)
            top_holdings_str = "🛑 100% CASH (Tendance Baissière)"
            var_forecast, cvar_forecast = 0.0, 0.0
            
            # Mise à jour graphique
            cap_strat *= 1.0 # Le capital ne bouge pas
            curve_s = pd.Series(cap_strat, index=b_rets.index).tolist() # Ligne plate
            
             # On force la création de listes pour le plot même si c'est plat
            period_strat_curve = [cap_strat] * len(b_rets)
            period_bench_curve = (1 + b_rets).cumprod() * cap_bench # Le bench continue de bouger
            period_bench_curve = period_bench_curve.tolist()
            period_dates = b_rets.index
            
            # On met à jour cap_bench pour le tour suivant
            cap_bench *= (1 + ((1 + b_rets).prod() - 1))
            
            per_bench_abs = (1 + b_rets).prod() - 1
            per_strat_abs = 0.0
            
        else:
            # CAS 2 : On est investi
            alloc = clean_portfolio(alloc, SEUIL_FILTRE)
            # --- MODIF AFFICHAGE LONG/SHORT ---
            longs = alloc[alloc > 0].sort_values(ascending=False).head(2)
            shorts = alloc[alloc < 0].sort_values().head(2)
            
            txt_l = " ".join([f"L:{t.replace('=X','')}({w*100:.0f}%)" for t, w in longs.items()])
            txt_s = " ".join([f"S:{t.replace('=X','')}({w*100:.0f}%)" for t, w in shorts.items()])
            top_holdings_str = f"{txt_l} | {txt_s}"
            
            # Prévision Risque (sur train data)
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
                
                # Performance Stratégie
                daily_perf = s_rets.dot(alloc.values)
                
                # PAS DE FRAIS (Supprimés comme demandé)
                # if not daily_perf.empty: daily_perf.iloc[0] -= 0.0005 
                
                per_strat = (1 + daily_perf).prod() - 1
                cap_strat *= (1 + per_strat)
                
                # Mise à jour Bench
                per_bench = (1 + b_rets_aligned).prod() - 1
                cap_bench *= (1 + per_bench)
                
                # Courbes
                curve_s = (1 + daily_perf).cumprod() * (cap_strat / (1+per_strat))
                curve_b = (1 + b_rets_aligned).cumprod() * (cap_bench / (1+per_bench))
                
                period_strat_curve = curve_s.tolist()
                period_bench_curve = curve_b.tolist()
                period_dates = common_idx
                
                per_strat_abs = per_strat
                per_bench_abs = per_bench

        period_strat_curve = []
        period_bench_curve = []
        period_dates = []
        
        var_forecast = np.nan
        cvar_forecast = np.nan
        top_holdings_str = "CASH"
        
        if alloc is not None:
            top3 = alloc.head(3)
            # Affichage nettoyé du suffixe =X pour la lisibilité
# On prend les 2 plus gros achats ET les 2 plus gros shorts
            shorts = alloc[alloc < -0.05].sort_values().head(2)
            longs = alloc[alloc > 0.05].sort_values(ascending=False).head(2)

            txt_long = " ".join([f"Long {t.replace('=X','')}({w*100:.0f}%)" for t, w in longs.items()])
            txt_short = " ".join([f"Short {t.replace('=X','')}({w*100:.0f}%)" for t, w in shorts.items()])
            top_holdings_str = f"{txt_long} | {txt_short}"            
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
            print(f"   👉 {t.replace('=X',''):<10} | {w*100:.3f}%      | {budget*w:.0f} €          | {price:.4f}")
    
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
           # ... (juste après tickers = alloc_conviction...)
            shares = best_plan["shares"]
            
            for i, t in enumerate(tickers):
                n = int(shares[i])
                p = current_prices[t]
                amt = n * p
                w_real = abs(amt) / best_plan['cost'] # Poids en valeur absolue
                
                # Action : ACHAT ou VENTE (Short)
                action = "ACHAT (Long)" if n > 0 else "VENTE (Short)"
                
                print(f"{action:<13} | {t.replace('=X',''):<10} | {p:<10.4f}   | {n:<10} | {amt:<10.2f} € | {w_real*100:.1f}%")
        else:
            print("Budget trop serré.")
else:
    print("Erreur données ou Optimisation échouée (Pas assez de variance).")

plt.show()