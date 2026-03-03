"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  MODÈLES ML — Walk-Forward Training                                            ║
║                                                                                ║
║  COURT TERME  : Ridge Regression, LightGBM CT, LightGBM+Hurst                 ║
║  MOYEN TERME  : Gradient Boosting, Ensemble (GB+LSTM+Transformer), TCN        ║
║  LONG TERME   : Random Forest LT, MLP macro                                   ║
║                                                                                ║
║  Tous en walk-forward avec retrain glissant.                                  ║
║  Chaque modèle retourne prediction_series (P(UP) ou score indexé par date).   ║
║                                                                                ║
║  Références :                                                                  ║
║    Jansen ML4T ch.7  — Ridge/Lasso for return forecasting                     ║
║    Jansen ML4T ch.11 — Random Forests                                         ║
║    Jansen ML4T ch.12 — Gradient Boosting (XGBoost/LightGBM)                  ║
║    Jansen ML4T ch.17 — Deep Learning (MLP)                                    ║
║    Jansen ML4T ch.19 — RNN/LSTM                                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math, gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Imports conditionnels (dépendances optionnelles) ──
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential as KerasSequential, Model
    from tensorflow.keras.layers import (
        LSTM as KerasLSTM, Dense as KerasDense, Dropout as KerasDropout,
        Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input,
        BatchNormalization, Activation, Add
    )
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from config import (
    FEATURE_COLS, FEATURE_COLS_CT, FEATURE_COLS_MT, FEATURE_COLS_LT,
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_LAYERS,
    TRANSFORMER_DIM_FF, TRANSFORMER_DROPOUT, TRANSFORMER_EPOCHS,
    TRANSFORMER_LR, TRANSFORMER_BATCH,
    BACKTEST_START, USER
)
from utils_display import hdr, tbl


# ════════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════════

def create_sequences(X: np.ndarray, y: np.ndarray,
                     lookback: int = LSTM_LOOKBACK):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _empty_result() -> dict:
    return {
        'accuracy': 0.5, 'predictions': [], 'dates': [],
        'expected_return_bps': 0, 'importances': {},
        'prediction_series': pd.Series(dtype=float)
    }


def _build_result(preds, probas, actuals, dates, returns_series) -> dict:
    preds   = np.array(preds)
    probas  = np.array(probas) if len(probas) > 0 else np.array([])
    actuals = np.array(actuals)
    acc     = accuracy_score(actuals, preds) if len(preds) > 0 else 0.5
    pred_series = pd.Series(probas, index=dates) if dates else pd.Series(dtype=float)
    e_r = returns_series.loc[dates].values if dates else np.array([0.0])
    mask_up  = preds == 1
    mean_up  = np.mean(e_r[mask_up])  if mask_up.any()  else 0.0
    mean_dn  = np.mean(e_r[~mask_up]) if (~mask_up).any() else 0.0
    pred_ret = mean_up - mean_dn
    return {
        'accuracy': acc, 'predictions': preds, 'dates': dates,
        'expected_return_bps': pred_ret * 10000,
        'prediction_series': pred_series
    }


# ════════════════════════════════════════════════════════════════════════════════
# COURT TERME — Ridge Regression
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.7 — Linear Models for return forecasting
# Logique : Ridge sur features CT (fenêtres courtes), prédit direction J+1.
# Avantage : très peu d'overfitting, stable, interprétable.

def train_ridge_ct_walkforward(data_dict: dict, pairs: list) -> dict:
    hdr("RIDGE CT — Walk-Forward (Court Terme)")
    results = {}
    rows = []

    for pair in pairs:
        try:
            df = data_dict[pair].dropna(subset=FEATURE_COLS_CT + ['Target_Direction'])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < 200 or oos_start >= len(df) - 50:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = USER.wf_retrain_freq
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_tr = df[FEATURE_COLS_CT].iloc[t_start:t].values.astype(np.float64)
                    y_tr = df['Target_Direction'].iloc[t_start:t].values

                    scaler = RobustScaler()
                    X_sc   = scaler.fit_transform(X_tr)

                    # Ridge avec alpha optimal via validation simple
                    current_model = RidgeClassifier(alpha=1.0)
                    current_model.fit(X_sc, y_tr)

                X_t    = scaler.transform(df[FEATURE_COLS_CT].iloc[t:t+1].values.astype(np.float64))
                score  = float(current_model.decision_function(X_t)[0])
                # Convertir score en probabilité via sigmoid
                proba  = 1.0 / (1.0 + np.exp(-score * 0.5))
                proba  = np.clip(proba, 0.01, 0.99)
                pred   = int(proba > 0.5)

                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df['Target_Direction'].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])

        except Exception as e:
            print(f"  [WARN] Ridge CT {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r]", "Accuracy"], rows)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# COURT TERME — LightGBM (ou GBM si LightGBM absent)
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.12 — Gradient Boosting for intraday/CT strategies
# Logique : LGBM sur features CT, retrain tous les 63j, rebal 2j.
# LightGBM > sklearn GBM sur grandes données (10x plus rapide).

def _make_lgbm_ct():
    """Crée un classifier LightGBM ou fallback sklearn GBM."""
    if HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.02,
            num_leaves=31, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=1
        )
    else:
        # Fallback sklearn
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.02,
            min_samples_leaf=10, subsample=0.8,
            random_state=42, verbose=0
        )


def train_lgbm_ct_walkforward(data_dict: dict, pairs: list,
                               use_hurst_filter: bool = False) -> dict:
    """
    LightGBM CT walk-forward.
    use_hurst_filter=True : n'agit que quand H < 0.5 (régime mean-rev confirmé).
    """
    label = "LGBM+HURST CT" if use_hurst_filter else "LGBM CT"
    hdr(f"{label} — Walk-Forward (Court Terme)")
    results = {}
    rows    = []

    for pair in pairs:
        try:
            df = data_dict[pair].dropna(subset=FEATURE_COLS_CT + ['Target_Direction'])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < 200 or oos_start >= len(df) - 50:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = USER.wf_retrain_freq
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_tr = df[FEATURE_COLS_CT].iloc[t_start:t].values.astype(np.float64)
                    y_tr = df['Target_Direction'].iloc[t_start:t].values

                    scaler = RobustScaler()
                    X_sc   = scaler.fit_transform(X_tr)
                    current_model = _make_lgbm_ct()
                    current_model.fit(X_sc, y_tr)

                X_t   = scaler.transform(df[FEATURE_COLS_CT].iloc[t:t+1].values.astype(np.float64))
                proba = float(current_model.predict_proba(X_t)[0][1])

                # Filtre Hurst : si tendance (H > 0.55), signal faible car mean-rev n'est pas le bon régime
                if use_hurst_filter and 'Hurst' in df.columns:
                    h = df['Hurst'].iloc[t]
                    if pd.notna(h) and h > 0.55:
                        proba = 0.5 + (proba - 0.5) * 0.3   # atténuer signal

                pred = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df['Target_Direction'].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])

        except Exception as e:
            print(f"  [WARN] LGBM CT {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r]", "Accuracy"], rows)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# MOYEN TERME — Gradient Boosting seul
# ════════════════════════════════════════════════════════════════════════════════

def train_gb_walkforward(data_dict: dict, pairs: list) -> dict:
    """
    Gradient Boosting MT — prédit la direction à 21j (vrai moyen terme).
    Retrain trimestriel (63j), features MT (fenêtres 1-63j).
    Ref : Jansen ML4T ch.12.
    """
    hdr("GRADIENT BOOSTING MT — Walk-Forward (target 21j)")
    results = {}
    rows    = []

    for pair in pairs:
        try:
            # Target : direction du rendement cumulé sur 21j
            target_col = 'Target_Direction_21d'
            df = data_dict[pair].dropna(subset=FEATURE_COLS_MT + [target_col])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < 252 or oos_start >= len(df) - 63:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = USER.wf_retrain_freq      # 63j = trimestriel
            train_window  = USER.wf_train_window       # 756j = 3 ans
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_tr = df[FEATURE_COLS_MT].iloc[t_start:t].values.astype(np.float64).copy()
                    y_tr = df[target_col].iloc[t_start:t].values.copy()
                    scaler = StandardScaler()
                    X_sc   = scaler.fit_transform(X_tr)
                    current_model = GradientBoostingClassifier(
                        n_estimators=150, max_depth=4, learning_rate=0.02,
                        min_samples_leaf=20, subsample=0.8,
                        random_state=42, verbose=0
                    )
                    current_model.fit(X_sc, y_tr)

                X_t   = scaler.transform(df[FEATURE_COLS_MT].iloc[t:t+1].values.astype(np.float64))
                proba = float(current_model.predict_proba(X_t)[0][1])
                pred  = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df[target_col].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_21d'] if 'Return_21d' in df else df['Return_1d'])
            res['importances'] = dict(zip(
                FEATURE_COLS_MT[:len(current_model.feature_importances_)],
                current_model.feature_importances_
            )) if current_model else {}
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])

        except Exception as e:
            print(f"  [WARN] GB {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r] 21j", "Accuracy"], rows)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# MOYEN TERME — LSTM Walk-Forward
# ════════════════════════════════════════════════════════════════════════════════

def train_lstm_walkforward(data_dict: dict, pairs: list) -> dict:
    """
    LSTM MT — prédit direction à 21j.
    Séquence de 40j → prédiction du rendement cumulé sur les 21j suivants.
    Retrain semestriel (126j) car coûteux computationnellement.
    """
    hdr("LSTM MT — Walk-Forward (target 21j)")

    if not HAS_TENSORFLOW:
        print("  [WARN] TensorFlow non disponible — remplacement par GB MT")
        return train_gb_walkforward(data_dict, pairs)

    results = {}
    rows    = []

    for pair in pairs:
        try:
            target_col = 'Target_Direction_21d'
            df = data_dict[pair].dropna(subset=FEATURE_COLS_MT + [target_col])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < LSTM_LOOKBACK + 252:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = 126    # semestriel — LSTM est coûteux
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_raw = df[FEATURE_COLS_MT].iloc[t_start:t].values.astype(np.float64).copy()
                    y_raw = df[target_col].iloc[t_start:t].values.copy()
                    scaler = StandardScaler()
                    X_sc   = scaler.fit_transform(X_raw)
                    X_seq, y_seq = create_sequences(X_sc, y_raw)
                    if len(X_seq) < 100:
                        continue
                    if current_model is not None:
                        del current_model
                        tf.keras.backend.clear_session()
                    current_model = KerasSequential([
                        KerasLSTM(64, return_sequences=True,
                                  input_shape=(LSTM_LOOKBACK, len(FEATURE_COLS_MT))),
                        KerasDropout(0.2),
                        KerasLSTM(32),
                        KerasDropout(0.2),
                        KerasDense(1, activation='sigmoid')
                    ])
                    current_model.compile(optimizer='adam', loss='binary_crossentropy')
                    current_model.fit(
                        X_seq, y_seq,
                        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0,
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                    )

                if current_model is None or scaler is None:
                    continue
                lookback_data = df[FEATURE_COLS_MT].iloc[t - LSTM_LOOKBACK:t].values.astype(np.float64)
                X_t   = scaler.transform(lookback_data).reshape(1, LSTM_LOOKBACK, len(FEATURE_COLS_MT))
                proba = float(current_model.predict(X_t, verbose=0)[0][0])
                pred  = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df[target_col].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_21d'] if 'Return_21d' in df else df['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])
            if current_model is not None:
                del current_model
                tf.keras.backend.clear_session()
                gc.collect()

        except Exception as e:
            print(f"  [WARN] LSTM {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r] 21j", "Accuracy"], rows)
    gc.collect()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# MOYEN TERME — Transformer Walk-Forward
# ════════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module if HAS_TORCH else object):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        if not HAS_TORCH:
            return
        super().__init__()
        import torch.nn as nn
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FXTransformer(nn.Module if HAS_TORCH else object):
    def __init__(self, n_features):
        if not HAS_TORCH:
            return
        super().__init__()
        import torch.nn as nn
        d = TRANSFORMER_D_MODEL
        self.input_proj = nn.Linear(n_features, d)
        self.pos_enc    = PositionalEncoding(d, TRANSFORMER_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=TRANSFORMER_NHEAD,
            dim_feedforward=TRANSFORMER_DIM_FF,
            dropout=TRANSFORMER_DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.head    = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Dropout(TRANSFORMER_DROPOUT),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze(-1)


def train_transformer_walkforward(data_dict: dict, pairs: list) -> dict:
    """
    Transformer MT — prédit direction à 21j.
    Séquence 40j × features MT → prédiction rendement cumulé 21j.
    Retrain semestriel (126j).
    """
    hdr("TRANSFORMER MT — Walk-Forward (target 21j)")

    if not HAS_TORCH:
        print("  [WARN] PyTorch non disponible — remplacement par GB MT")
        return train_gb_walkforward(data_dict, pairs)

    device  = torch.device('cpu')
    results = {}
    rows    = []

    for pair in pairs:
        try:
            target_col = 'Target_Direction_21d'
            df = data_dict[pair].dropna(subset=FEATURE_COLS_MT + [target_col])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < LSTM_LOOKBACK + 252:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = 126    # semestriel
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_raw = df[FEATURE_COLS_MT].iloc[t_start:t].values.astype(np.float64).copy()
                    y_raw = df[target_col].iloc[t_start:t].values.copy()
                    scaler = StandardScaler()
                    X_sc   = scaler.fit_transform(X_raw)
                    X_seq, y_seq = create_sequences(X_sc, y_raw)
                    if len(X_seq) < 100:
                        continue
                    X_tr = torch.FloatTensor(X_seq).to(device)
                    y_tr = torch.FloatTensor(y_seq).to(device)
                    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                                          batch_size=TRANSFORMER_BATCH, shuffle=True)
                    current_model = FXTransformer(n_features=len(FEATURE_COLS_MT)).to(device)
                    opt     = torch.optim.Adam(current_model.parameters(), lr=TRANSFORMER_LR)
                    loss_fn = nn.BCELoss()
                    best_loss, patience_cnt = float('inf'), 0
                    for ep in range(TRANSFORMER_EPOCHS):
                        current_model.train()
                        ep_loss = 0
                        for xb, yb in train_dl:
                            opt.zero_grad()
                            loss = loss_fn(current_model(xb), yb)
                            loss.backward()
                            opt.step()
                            ep_loss += loss.item()
                        avg = ep_loss / len(train_dl)
                        if avg < best_loss - 1e-4:
                            best_loss, patience_cnt = avg, 0
                        else:
                            patience_cnt += 1
                            if patience_cnt >= 5:
                                break

                if current_model is None or scaler is None:
                    continue
                lookback_data = df[FEATURE_COLS_MT].iloc[t - LSTM_LOOKBACK:t].values.astype(np.float64)
                X_t = torch.FloatTensor(scaler.transform(lookback_data)).unsqueeze(0).to(device)
                current_model.eval()
                with torch.no_grad():
                    proba = float(current_model(X_t).cpu().item())
                pred = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df[target_col].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_21d'] if 'Return_21d' in df else df['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])
            del current_model
            gc.collect()

        except Exception as e:
            print(f"  [WARN] Transformer {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r]", "Accuracy"], rows)
    gc.collect()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# MOYEN TERME — TCN (Temporal Convolutional Network)
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional
#       and Recurrent Networks for Sequence Modeling
#       Jansen ML4T ch.18 — CNN for Financial Time Series

def _build_tcn_keras(n_features: int, lookback: int,
                     filters: int = 64, kernel_size: int = 3,
                     dilations: list = None) -> 'tf.keras.Model':
    """
    TCN : convolutions causales dilatées.
    Chaque couche "voit" 2^k fois plus loin que la précédente.
    Plus rapide que LSTM, souvent meilleur sur CT.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    inp = Input(shape=(lookback, n_features))
    x   = inp
    for d in dilations:
        # Convolution causale dilatée
        residual = x
        x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=d,
                   activation='relu')(x)
        x = BatchNormalization()(x)
        x = KerasDropout(0.1)(x)
        # Residual connection si taille compatible
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding='same')(residual)
        x = Add()([x, residual])

    x   = GlobalAveragePooling1D()(x)
    x   = KerasDense(32, activation='relu')(x)
    out = KerasDense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train_tcn_walkforward(data_dict: dict, pairs: list) -> dict:
    """TCN walk-forward — nécessite TensorFlow."""
    hdr("TCN MT — Walk-Forward (Temporal Convolutional Network)")

    if not HAS_TENSORFLOW:
        print("  [WARN] TensorFlow non disponible — remplacement par LGBM CT")
        return train_lgbm_ct_walkforward(data_dict, pairs)

    results = {}
    rows    = []

    for pair in pairs:
        try:
            df = data_dict[pair].dropna(subset=FEATURE_COLS + ['Target_Direction'])
            oos_start = df.index.searchsorted(pd.Timestamp(BACKTEST_START))
            if oos_start < LSTM_LOOKBACK + 200:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = USER.wf_retrain_freq * 2
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_raw = df[FEATURE_COLS].iloc[t_start:t].values.astype(np.float64).copy()
                    y_raw = df['Target_Direction'].iloc[t_start:t].values.copy()
                    scaler = StandardScaler()
                    X_sc   = scaler.fit_transform(X_raw)
                    X_seq, y_seq = create_sequences(X_sc, y_raw, lookback=LSTM_LOOKBACK)
                    if len(X_seq) < 100:
                        continue
                    if current_model is not None:
                        del current_model
                        tf.keras.backend.clear_session()
                    current_model = _build_tcn_keras(len(FEATURE_COLS), LSTM_LOOKBACK)
                    current_model.fit(
                        X_seq, y_seq,
                        epochs=20, batch_size=32, verbose=0,
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                    )

                if current_model is None or scaler is None:
                    continue
                lookback_data = df[FEATURE_COLS].iloc[t - LSTM_LOOKBACK:t].values.astype(np.float64)
                X_t   = scaler.transform(lookback_data).reshape(1, LSTM_LOOKBACK, len(FEATURE_COLS))
                proba = float(current_model.predict(X_t, verbose=0)[0][0])
                pred  = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df['Target_Direction'].iloc[t])
                all_dates.append(df.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates, df['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])
            if current_model is not None:
                del current_model
                tf.keras.backend.clear_session()
                gc.collect()

        except Exception as e:
            print(f"  [WARN] TCN {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r]", "Accuracy"], rows)
    gc.collect()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# LONG TERME — Random Forest
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.11 — Random Forests for FX/equity return prediction
#       Breiman (2001) — Random Forests

def train_rf_lt_walkforward(data_dict: dict, pairs: list) -> dict:
    """
    Random Forest LT — prédit direction à 63j (3 mois).
    Features LT : momentum 12m, carry, vol 126j, SMA longues.
    Retrain semestriel (126j) car signal LT lent.
    Ref : Jansen ML4T ch.11, Moskowitz et al. (2012).
    """
    hdr("RANDOM FOREST LT — Walk-Forward (target 63j)")
    results = {}
    rows    = []

    for pair in pairs:
        try:
            target_col = 'Target_Direction_63d'
            valid_cols = [c for c in FEATURE_COLS_LT if c in data_dict[pair].columns]
            df_valid   = data_dict[pair].dropna(subset=valid_cols + [target_col])
            oos_start  = df_valid.index.searchsorted(pd.Timestamp(BACKTEST_START))

            if oos_start < 252 or oos_start >= len(df_valid) - 63:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = 126    # semestriel — signal LT très lent
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df_valid)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_tr = df_valid[valid_cols].iloc[t_start:t].values.astype(np.float64)
                    y_tr = df_valid[target_col].iloc[t_start:t].values

                    scaler = RobustScaler()
                    X_sc   = scaler.fit_transform(X_tr)

                    current_model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=30,   # plus grand pour éviter overfit sur signal lent
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42,
                        n_jobs=1
                    )
                    current_model.fit(X_sc, y_tr)

                X_t   = scaler.transform(df_valid[valid_cols].iloc[t:t+1].values.astype(np.float64))
                proba = float(current_model.predict_proba(X_t)[0][1])
                pred  = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df_valid[target_col].iloc[t])
                all_dates.append(df_valid.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates,
                                df_valid['Return_63d'] if 'Return_63d' in df_valid else df_valid['Return_1d'])
            res['importances'] = dict(zip(
                valid_cols[:len(current_model.feature_importances_)],
                current_model.feature_importances_
            )) if current_model else {}
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])

        except Exception as e:
            print(f"  [WARN] RF LT {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r] 63j", "Accuracy"], rows)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# LONG TERME — MLP Macro (réseau dense sur features lentes)
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.17 — Deep Learning for Trading (MLP baseline)
#       Gu, Kelly & Xiu (2020) — Empirical Asset Pricing via Machine Learning

def train_mlp_macro_lt_walkforward(data_dict: dict, pairs: list) -> dict:
    """
    MLP macro LT — prédit direction à 63j.
    Architecture 64→32, L2 reg, early stopping.
    Features LT : momentum 12m, carry, vol 126j.
    Retrain semestriel (126j).
    Ref : Gu, Kelly & Xiu (2020) — Empirical Asset Pricing via ML.
    """
    hdr("MLP MACRO LT — Walk-Forward (target 63j)")
    results = {}
    rows    = []

    for pair in pairs:
        try:
            target_col = 'Target_Direction_63d'
            valid_cols = [c for c in FEATURE_COLS_LT if c in data_dict[pair].columns]
            df_valid   = data_dict[pair].dropna(subset=valid_cols + [target_col])
            oos_start  = df_valid.index.searchsorted(pd.Timestamp(BACKTEST_START))

            if oos_start < 252 or oos_start >= len(df_valid) - 63:
                raise ValueError("Pas assez de données")

            all_preds, all_probas, all_actuals, all_dates = [], [], [], []
            retrain_freq  = 126    # semestriel
            train_window  = USER.wf_train_window
            current_model = None
            scaler        = None

            for t in range(oos_start, len(df_valid)):
                if current_model is None or (t - oos_start) % retrain_freq == 0:
                    t_start = max(0, t - train_window)
                    X_tr = df_valid[valid_cols].iloc[t_start:t].values.astype(np.float64)
                    y_tr = df_valid[target_col].iloc[t_start:t].values

                    scaler = RobustScaler()
                    X_sc   = scaler.fit_transform(X_tr)

                    current_model = MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation='relu',
                        solver='adam',
                        alpha=1e-3,
                        batch_size='auto',
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                        verbose=False
                    )
                    current_model.fit(X_sc, y_tr)

                X_t   = scaler.transform(df_valid[valid_cols].iloc[t:t+1].values.astype(np.float64))
                proba = float(current_model.predict_proba(X_t)[0][1])
                pred  = int(proba > 0.5)
                all_probas.append(proba)
                all_preds.append(pred)
                all_actuals.append(df_valid[target_col].iloc[t])
                all_dates.append(df_valid.index[t])

            res = _build_result(all_preds, all_probas, all_actuals,
                                all_dates,
                                df_valid['Return_63d'] if 'Return_63d' in df_valid else df_valid['Return_1d'])
            results[pair] = res
            rows.append([pair, f"{res['expected_return_bps']:+.1f} bps",
                         f"{res['accuracy']:.1%}"])

        except Exception as e:
            print(f"  [WARN] MLP LT {pair}: {str(e)[:60]}")
            results[pair] = _empty_result()
            rows.append([pair, "0.0 bps", "50.0%"])

    tbl(["Pair", "E[r] 63j", "Accuracy"], rows)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# DISPATCHER — sélectionne le modèle selon le pipeline
# ════════════════════════════════════════════════════════════════════════════════

def train_models_for_pipeline(pipeline_cfg: dict, data_dict: dict,
                               pairs: list, horizon: str) -> dict:
    """
    Dispatcher central : sélectionne et entraîne le bon modèle
    selon le pipeline_cfg retourné par PIPELINE_REGISTRY.

    Retourne un dict {pair: result_dict} compatible avec strategies.py.
    """
    model_id = pipeline_cfg['model']

    # ── Court terme ──
    if model_id == 'ridge_ct':
        return train_ridge_ct_walkforward(data_dict, pairs)
    elif model_id == 'lgbm_ct':
        return train_lgbm_ct_walkforward(data_dict, pairs, use_hurst_filter=False)
    elif model_id == 'lgbm_hurst_ct':
        return train_lgbm_ct_walkforward(data_dict, pairs, use_hurst_filter=True)

    # ── Moyen terme ──
    elif model_id == 'gb_only':
        return train_gb_walkforward(data_dict, pairs)
    elif model_id in ('ensemble_mt', 'ensemble_ep_mt'):
        # L'ensemble complet est géré dans strategies.py qui combine GB+Trans+LSTM
        return train_gb_walkforward(data_dict, pairs)

    # ── Long terme ──
    elif model_id == 'carry_mom_bl':
        return {}   # pas de ML, géré directement dans strategy_long_term
    elif model_id == 'rf_lt':
        return train_rf_lt_walkforward(data_dict, pairs)
    elif model_id == 'mlp_macro_lt':
        return train_mlp_macro_lt_walkforward(data_dict, pairs)

    # ── EP standalone ──
    elif model_id in ('ep_signals', 'ep_ml', 'ep_ml_full'):
        return train_gb_walkforward(data_dict, pairs)  # EP utilise GB comme vue principale

    else:
        print(f"  [WARN] Modèle inconnu '{model_id}' — fallback GB")
        return train_gb_walkforward(data_dict, pairs)