import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
import os

# --- CONFIGURATION ---
FILE_RETURNS = "MASTER_RETURNS.csv"
CONFIDENCE_LEVEL = 0.95  # Pour la VaR (95%)
RISK_FREE_RATE = 0.0     # Taux sans risque (0% pour simplifier sur le Forex)
TRADING_DAYS = 252       # Jours de trading par an
MC_SIMULATIONS = 10000   # Nombre de simulations pour Monte Carlo

def run_full_analysis():
    print(">>> DÉMARRAGE DE L'ANALYSE COMPLÈTE (4 BLOCS) <<<")
    
    # 1. CHARGEMENT
    if not os.path.exists(FILE_RETURNS):
        print(f"Erreur : {FILE_RETURNS} introuvable.")
        return
    
    df_ret = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    print(f"Données chargées : {df_ret.shape[1]} devises sur {df_ret.shape[0]} jours.")

    # ---------------------------------------------------------
    # BLOC 1 : VOLATILITÉ
    # ---------------------------------------------------------
    print("\n--- BLOC 1 : ANALYSE DE LA VOLATILITÉ ---")
    # Volatilité Annualisée
    vol_ann = df_ret.std() * np.sqrt(TRADING_DAYS)
    
    # Semi-Déviation (Risque à la baisse)
    downside_returns = df_ret[df_ret < 0]
    semi_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
    
    # EWMA (Volatilité pondérée, lambda=0.94)
    # alpha = 1 - lambda = 0.06
    ewma_vol = df_ret.ewm(alpha=0.06).std().iloc[-1] * np.sqrt(TRADING_DAYS)

    df_block1 = pd.DataFrame({
        'Volatilité (Ann)': vol_ann,
        'Semi-Déviation': semi_dev,
        'EWMA (Actuelle)': ewma_vol,
        'Ratio Symétrie (Vol/Semi)': vol_ann / semi_dev
    })

    # ---------------------------------------------------------
    # BLOC 2 : TAIL RISK (QUEUES DE DISTRIBUTION)
    # ---------------------------------------------------------
    print("--- BLOC 2 : ANALYSE DES QUEUES (VaR & CVaR) ---")
    
    # A. VaR Historique (95%)
    # On cherche le percentile 5%
    var_hist = df_ret.quantile(1 - CONFIDENCE_LEVEL)
    
    # B. VaR Paramétrique (Méthode Variance-Covariance / Gaussienne)
    # Formule : Mu - z * Sigma
    mu = df_ret.mean()
    sigma = df_ret.std()
    z_score = norm.ppf(1 - CONFIDENCE_LEVEL) # -1.645 pour 95%
    var_param = mu + z_score * sigma
    
    # C. VaR Monte Carlo (Simulation)
    var_mc = []
    for col in df_ret.columns:
        # On simule 10000 rendements possibles basés sur mu et sigma
        simulated_rets = np.random.normal(mu[col], sigma[col], MC_SIMULATIONS)
        var_mc.append(np.percentile(simulated_rets, (1 - CONFIDENCE_LEVEL) * 100))
    
    var_mc_series = pd.Series(var_mc, index=df_ret.columns)

    # D. Expected Shortfall (CVaR) - Historique
    # Moyenne des pertes pires que la VaR Historique
    cvar_list = []
    for col in df_ret.columns:
        cutoff = var_hist[col]
        tail_losses = df_ret[col][df_ret[col] <= cutoff]
        cvar_list.append(tail_losses.mean())
    cvar_series = pd.Series(cvar_list, index=df_ret.columns)

    df_block2 = pd.DataFrame({
        f'VaR Hist {CONFIDENCE_LEVEL:.0%}': var_hist,
        f'VaR Param {CONFIDENCE_LEVEL:.0%}': var_param,
        f'VaR MonteCarlo {CONFIDENCE_LEVEL:.0%}': var_mc_series,
        f'CVaR (Expected Shortfall)': cvar_series
    })

    # ---------------------------------------------------------
    # BLOC 3 : PERFORMANCE & CORRÉLATION
    # ---------------------------------------------------------
    print("--- BLOC 3 : PERFORMANCE & CORRÉLATIONS ---")
    
    # Rendement Annualisé Moyen (Géométrique c'est mieux, mais Arithmétique simple ici)
    mean_ret_ann = df_ret.mean() * TRADING_DAYS
    
    # Ratio de Sharpe
    sharpe = (mean_ret_ann - RISK_FREE_RATE) / vol_ann
    
    # Ratio de Sortino (Rendement / Semi-Déviation)
    sortino = (mean_ret_ann - RISK_FREE_RATE) / semi_dev
    
    # Beta
    # On crée un Benchmark "Synthétique" (Moyenne équipondérée de toutes les devises)
    market_index = df_ret.mean(axis=1)
    market_var = market_index.var()
    
    betas = []
    for col in df_ret.columns:
        cov = df_ret[col].cov(market_index)
        beta = cov / market_var
        betas.append(beta)
    
    beta_series = pd.Series(betas, index=df_ret.columns)

    df_block3 = pd.DataFrame({
        'Rendement Moyen (Ann)': mean_ret_ann,
        'Ratio Sharpe': sharpe,
        'Ratio Sortino': sortino,
        'Beta (vs Market Avg)': beta_series
    })

    # Matrice de Corrélation (sauvegardée à part)
    corr_matrix = df_ret.corr()

    # ---------------------------------------------------------
    # BLOC 4 : DISTRIBUTION & DRAWDOWNS
    # ---------------------------------------------------------
    print("--- BLOC 4 : TRAJECTOIRES & DRAWDOWNS ---")
    
    # Skewness (Asymétrie) et Kurtosis (Aplatissement)
    # Skew < 0 : Queue épaisse à gauche (risque de krach)
    # Kurtosis > 3 : Leptokurtique (événements extrêmes fréquents)
    skew_s = df_ret.apply(skew)
    kurt_s = df_ret.apply(kurtosis) # C'est l'Excess Kurtosis (Normal = 0 ici)
    
    # Calcul des Drawdowns
    # 1. NAV (Base 100)
    nav = (1 + df_ret).cumprod() * 100
    # 2. Pic Courant (Running Max)
    running_max = nav.cummax()
    # 3. Drawdown (%)
    drawdown = (nav - running_max) / running_max
    # 4. Max Drawdown
    max_drawdown = drawdown.min()
    
    # Ratio de Calmar (Rendement Ann / Abs(Max Drawdown))
    # Attention aux divisions par zéro
    calmar = mean_ret_ann / abs(max_drawdown)
    
    df_block4 = pd.DataFrame({
        'Skewness': skew_s,
        'Kurtosis (Excess)': kurt_s,
        'Max Drawdown': max_drawdown,
        'Ratio Calmar': calmar
    })

    # ---------------------------------------------------------
    # CONSOLIDATION & SAUVEGARDE
    # ---------------------------------------------------------
    print("\n>>> SAUVEGARDE DES RÉSULTATS... <<<")
    
    # On fusionne tout dans un énorme fichier Global
    global_report = pd.concat([df_block1, df_block2, df_block3, df_block4], axis=1)
    
    # Tri par Sharpe Ratio (les meilleurs en haut)
    global_report = global_report.sort_values(by='Ratio Sharpe', ascending=False)
    
    # Sauvegarde CSV
    global_report.to_csv("RAPPORT_GLOBAL_RISQUE.csv")
    corr_matrix.to_csv("MATRICE_CORRELATION.csv")
    
    print("\n--- RÉSUMÉ DES MEILLEURS ACTIFS (TOP 3 SHARPE) ---")
    print(global_report[['Ratio Sharpe', 'Volatilité (Ann)', 'Max Drawdown']].head(3))
    
    print("\n--- FICHIERS GÉNÉRÉS ---")
    print("1. RAPPORT_GLOBAL_RISQUE.csv (Toutes les métriques)")
    print("2. MATRICE_CORRELATION.csv (Corrélations croisées)")
    print("\nAnalyse terminée avec succès.")

if __name__ == "__main__":
    run_full_analysis()