import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
pairs_list = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "CHFJPY", "EURJPY", "EURAUD", "GBPAUD", "CADJPY", "NZDJPY", "EURCAD", "GBPCAD",
    "USDTRY", "USDINR", "USDBRL", "USDZAR", "USDSEK"
]

def update_forex_data(pair):
    ticker_symbol = f"{pair}=X" #on ajoute =X au nom de la paire car c'est comme ça que sont les paires sur yahoo finance
    filename = f"{ticker_symbol}.csv"
    
    print(f"--- Traitement de {ticker_symbol} ---")
    if os.path.exists(filename):#si le ficher existe : (cas 2 : on le met à jour : )

        print(f"Fichier {filename} trouvé. Vérification des mises à jour...")
        try : 
            df_existing = pd.read_csv(filename, index_col=0, parse_dates=True)# a. Charger le fichier CSV existant
            df_existing.index = pd.to_datetime(df_existing.index, utc=True, errors='coerce')# On force la conversion de l'index en datetime pour éviter les erreurs de concaténation
            df_existing = df_existing[df_existing.index.notnull()]
            df_existing.index = df_existing.index.tz_localize(None) # On retire le fuseau horaire (tz-naive) pour éviter les conflits de calcul
            if not df_existing.empty:# b. Identifier la date la plus récente
                last_date = df_existing.index.max()
                start_date = last_date + timedelta(days=1)# On veut commencer à télécharger à partir du lendemain
                if start_date > datetime.now():# Vérifier si la date de début est dans le futur par rapport à aujourd'hui
                    print("Déjà à jour, pas de changements à faire.")
                    return
                print(f"Dernière date connue : {last_date.date()}. Téléchargement depuis {start_date.date()}...")
                df_new = yf.download(ticker_symbol, start=start_date, progress=False) # c. Télécharger uniquement les nouvelles données               # c. Télécharger uniquement les nouvelles données
                if not df_new.empty:
                    df_updated = pd.concat([df_existing, df_new])# d. Ajouter les nouvelles données
                    df_updated.to_csv(filename)# e. Sauvegarder le fichier mis à jour au même nom
                    print(f"Mise à jour réussie. {len(df_new)} nouvelles lignes ajoutées.")
                else:
                    print("Aucune nouvelle donnée disponible sur Yahoo Finance.")
            else:
                print("Fichier existant vide ou corrompu. Relance complète.")# Si le fichier existe mais est vide, on relance un téléchargement complet
                df_full = yf.download(ticker_symbol, period="10y", progress=False)
                df_full.to_csv(filename)
        except Exception as e:
            print(f"Erreur lors de la lecture/mise à jour de {filename} : {e}")
    else:
        # Cas 1 : Le fichier n'existe pas (uniquement quand c'est le premier lancement
        print(f"Fichier {filename} non trouvé. Téléchargement de l'historique complet (10 ans)...")
        df_full = yf.download(ticker_symbol, period="10y", progress=False)# a. Télécharger l'historique complet (10 ans)
        if not df_full.empty:
            df_full.to_csv(filename)# b. Sauvegarder dans le nouveau fichier CSV
            print(f"Fichier créé avec succès ({len(df_full)} lignes).")
        else:
            print("Erreur : Aucune donnée récupérée depuis Yahoo Finance.")

# Exécution du script pour toutes les paires
if __name__ == "__main__":
    for pair in pairs_list:
        update_forex_data(pair)
    print("Toutes les paires ont été chargées.")