import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD",
    "USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"
]

def update_forex_data(pairs):
    for pair in pairs:
        ticker = f"{pair}=X"
        filename = f"{ticker}.csv"
        
        if os.path.exists(filename):
            try:
                df_existing = pd.read_csv(filename, index_col=0, parse_dates=True)
                df_existing.index = pd.to_datetime(df_existing.index, utc=True).tz_localize(None)
                
                last_date = df_existing.index.max()
                start_date = last_date + timedelta(days=1)
                
                if start_date < datetime.now():
                    df_new = yf.download(ticker, start=start_date, progress=False)
                    if not df_new.empty:
                        df_new.index = df_new.index.tz_localize(None)
                        df_updated = pd.concat([df_existing, df_new])
                        df_updated.to_csv(filename)
            except Exception:
                pass
        else:
            df_full = yf.download(ticker, period="10y", progress=False)
            if not df_full.empty:
                df_full.index = df_full.index.tz_localize(None)
                df_full.to_csv(filename)

def get_returns_and_covariance(pairs):
    all_returns = []
    for pair in pairs:
        filename = f"{pair}=X.csv"
        if os.path.exists(filename):
            # On lit le CSV
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            
            close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            
            # --- CORRECTION ICI ---
            # On force la conversion en numérique, les erreurs deviennent des NaN
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            # On supprime les lignes vides ou erronées
            df = df.dropna(subset=[close_col])
            
            if not df.empty:
                # Calcul des log-returns
                returns = np.log(df[close_col] / df[close_col].shift(1))
                returns.name = pair
                all_returns.append(returns)
    
    if not all_returns:
        raise ValueError("Aucune donnée numérique valide n'a pu être extraite des CSV.")

    df_merged_returns = pd.concat(all_returns, axis=1).dropna()
    mu = df_merged_returns.mean() * 252
    sigma = df_merged_returns.cov() * 252
    
    return mu, sigma

if __name__ == "__main__":
    update_forex_data(pairs_list)
    mu, sigma = get_returns_and_covariance(pairs_list)
    
    print("--- VECTEUR MU ---")
    print(mu)
    print("\n--- MATRICE SIGMA ---")
    print(sigma)