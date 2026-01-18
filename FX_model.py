import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.optimize import minimize
from hmmlearn import hmm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


print("bonjour")

pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']
dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
n = len(dates)
n_pairs = len(pairs)

data_dict = {}
for i, pair in enumerate(pairs):
    np.random.seed(42 + i)
    trend = np.linspace(1.10 + i*0.05, 1.20 + i*0.05, n)
    noise = np.random.normal(0, 0.015 + i*0.003, n)
    close = trend + noise
    
    cycle = 0.05 * np.sin(2 * np.pi * np.arange(n) / (252 * 2 + i * 50))
    close = close + cycle
    
    df = pd.DataFrame({
        'Open': close * (1 + np.random.normal(0, 0.003, n)),
        'High': close * (1 + np.abs(np.random.normal(0, 0.006, n))),
        'Low': close * (1 - np.abs(np.random.normal(0, 0.006, n))),
        'Close': close,
    }, index=dates)
    
    data_dict[pair] = df


# feature engineering
def create_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Return_1d'] = df['Return'].shift(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    df['SMA_21'] = df['Close'].rolling(21).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['RealizedVol_21d'] = df['Return'].rolling(21).std() * np.sqrt(252)
    
    df['Target_Direction'] = (df['Return'].shift(-1) > 0).astype(int)
    
    df.dropna(inplace=True)
    return df

for pair in pairs:
    data_dict[pair] = create_features(data_dict[pair])


#Méthode R/S (Rescaled Range) pour calculer l'exposant de Hurst
#Plus robuste que la méthode des moments
#H doit être entre 0 et 1
# Peters 1994
def calculate_hurst_exponent(ts, max_lag=20):
    n = len(ts)
    
    if n < 20:
        return 0.5
    
    lags = range(2, min(max_lag, n // 2))
    
    tau = []
    for lag in lags:
        n_chunks = n // lag
        if n_chunks == 0:
            continue
        
        rs_values = []
        for i in range(n_chunks):
            chunk = ts[i*lag:(i+1)*lag]
            if len(chunk) < 2:
                continue
            
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumsum_dev = np.cumsum(deviations)
            
            R = np.max(cumsum_dev) - np.min(cumsum_dev)
            S = np.std(chunk)
            
            if S > 0:
                rs_values.append(R / S)
        
        if len(rs_values) > 0:
            tau.append(np.mean(rs_values))
    
    if len(tau) < 2:
        return 0.5
    
    lags_array = np.array(list(lags)[:len(tau)])
    tau_array = np.array(tau)
    
    if np.any(tau_array <= 0):
        return 0.5
    
    log_lags = np.log(lags_array)
    log_tau = np.log(tau_array)
    
    coeffs = np.polyfit(log_lags, log_tau, 1)
    H_hat = coeffs[0]
    
    H_hat = np.clip(H_hat, 0.0, 1.0)
    
    return H_hat

def rolling_hurst(series, window=100):
    hurst_values = []
    for i in range(window, len(series)):
        subset = series[i-window:i].values
        try:
            h = calculate_hurst_exponent(subset)
            hurst_values.append(h)
        except:
            hurst_values.append(np.nan)
    
    hurst_series = pd.Series([np.nan] * window + hurst_values, index=series.index)
    return hurst_series

for pair in pairs:
    data_dict[pair]['Hurst'] = rolling_hurst(data_dict[pair]['Close'], window=100)
    data_dict[pair].dropna(inplace=True)


# ML models first HMM for the regime detection

hmm_regimes = {}

for pair in pairs:
    df = data_dict[pair]
    
    returns = df['Return'].values.reshape(-1, 1)
    
    returns_clean = returns[~np.isnan(returns).any(axis=1)]
    returns_clean = returns_clean[~np.isinf(returns_clean).any(axis=1)]
    
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(returns_clean)
    
    hidden_states = model.predict(returns_clean)
    
    hidden_states_full = np.full(len(df), -1)
    valid_indices = ~np.isnan(returns).flatten() & ~np.isinf(returns).flatten()
    hidden_states_full[valid_indices] = hidden_states
    
    data_dict[pair]['HMM_Regime'] = hidden_states_full
    
    regime_means = []
    for i in range(3):
        regime_returns = df[hidden_states_full == i]['Return']
        if len(regime_returns) > 0:
            regime_means.append(regime_returns.mean())
        else:
            regime_means.append(0)
    
    regime_order = np.argsort(regime_means)
    regime_map = {regime_order[0]: 'BEAR', regime_order[1]: 'NEUTRAL', regime_order[2]: 'BULL', -1: 'NEUTRAL'}
    
    data_dict[pair]['HMM_Regime_Label'] = data_dict[pair]['HMM_Regime'].map(regime_map)
    
    current_regime = data_dict[pair]['HMM_Regime_Label'].iloc[-1]
    hmm_regimes[pair] = current_regime
    
    print(pair, "- Regime actuel HMM:", current_regime)

# LSTM à améliorer car dépend du HMM et peut contenir du data leakage 

feature_cols = ['Return_1d', 'Return_5d', 'SMA_21', 'MACD', 'RSI', 
                'RealizedVol_21d', 'Hurst']

lstm_predictions = {}
lookback = 30

for pair in pairs:
    df = data_dict[pair]
    
    X = df[feature_cols].values
    y = df['Target_Direction'].values
    
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def create_sequences(X, y, lookback):
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:i+lookback])
            ys.append(y[i+lookback])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    
    y_pred_proba = model.predict(X_test_seq, verbose=0)
    
    expected_return = (y_pred_proba.mean() - 0.5) * 2 * df['RealizedVol_21d'].mean() / np.sqrt(252)
    
    lstm_predictions[pair] = expected_return
    
    print(pair, "- LSTM prediction:",expected_return)

xgb_predictions = {}

for pair in pairs:
    df = data_dict[pair]
    
    X = df[feature_cols]
    y = df['Target_Direction']
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)
    
    expected_return = (y_pred_proba[:, 1].mean() - 0.5) * 2 * df['RealizedVol_21d'].mean() / np.sqrt(252)
    
    xgb_predictions[pair] = expected_return
    
    print(pair, "- XGBoost prediction:", expected_return)


# Combinaison des 3 modèles précédents en accord avec le papier de Liu
print("\nEnsemble des predictions (LSTM + XGBoost + HMM)")

ensemble_predictions = {}

for pair in pairs:
    lstm_pred = lstm_predictions[pair]
    xgb_pred = xgb_predictions[pair]
    
    hmm_regime = hmm_regimes[pair]
    if hmm_regime == 'BULL':
        hmm_boost = 1.2
    elif hmm_regime == 'BEAR':
        hmm_boost = 0.8
    else:
        hmm_boost = 1.0
    
    ensemble_pred = (0.5 * lstm_pred + 0.5 * xgb_pred) * hmm_boost
    
    ensemble_predictions[pair] = ensemble_pred
    
    print(pair, "- Ensemble:", round(ensemble_pred, 6), "(HMM:", hmm_regime, ")")


returns_matrix = pd.DataFrame({pair: data_dict[pair]['Return'] for pair in pairs})
returns_matrix = returns_matrix.dropna()

Sigma = returns_matrix.cov().values
mu = np.array([ensemble_predictions[pair] for pair in pairs])

print("\nRendements esperes mu (ensemble LSTM + XGBoost + HMM):")
print(mu)


#Formule exacte du cours: min w^T Sigma w
def objective_function(weight_vector, mat_corr):
    return weight_vector @ mat_corr @ weight_vector

# Optimisation de Markowitz du cours
#min w^T Sigma w s.c. w^T 1 = 1, sum(w_i * r_i) >= r_target

def markowitz_optimization(vec_returns, mat_corr):
    n = len(vec_returns)
    initial_guess = np.ones(n) / n
    bounds = [(0.05, 0.5) for _ in range(n)]
    
    target_return = np.mean(vec_returns[vec_returns > 0]) if np.any(vec_returns > 0) else np.mean(vec_returns)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, vec_returns) - target_return}
    ]
    
    result = minimize(
        objective_function,
        initial_guess,
        args=(mat_corr,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

poids_optimaux = markowitz_optimization(mu, Sigma)

print("\nPoids optimaux Markowitz (formule du cours avec contrainte rendement):")
for i, pair in enumerate(pairs):
    print(pair, ": ", poids_optimaux[i])


# Ajustement en fonction de hurst
# en fonction de si le marché a tendance à persister dans une tendance ou non 
hurst_mean = np.mean([data_dict[pair]['Hurst'].mean() for pair in pairs])

print("\nExposant de Hurst moyen: ", hurst_mean)

if hurst_mean > 0.55:
    regime = "TRENDING"
    leverage_factor = 1.3
elif hurst_mean < 0.45:
    regime = "MEAN REVERTING"
    leverage_factor = 0.7
else:
    regime = "BROWNIAN"
    leverage_factor = 1.0

print("Regime de marche (Hurst): ", regime)
print("Facteur de levier: ", leverage_factor)

omega_final = poids_optimaux * leverage_factor
omega_final = omega_final / np.sum(omega_final)

print("\nPoids finaux ajustes avec Hurst:")
for i, pair in enumerate(pairs):
    print(pair, ": ", round(omega_final[i], 4))


# juste quelques métriques pour voir le fonctionnement
def calculate_sharpe_ratio(returns):
    n = len(returns)
    mean_return = np.sum(returns) / n
    variance = np.sum(returns**2) / n
    if variance == 0:
        return 0
    sharpe = mean_return / np.sqrt(variance)
    return sharpe

def calculate_var(returns, alpha=0.95):
    return -np.percentile(returns, (1-alpha)*100)

def calculate_expected_shortfall(returns, alpha=0.95):
    var_alpha = calculate_var(returns, alpha)
    losses = -returns
    tail_losses = losses[losses >= var_alpha]
    if len(tail_losses) == 0:
        return var_alpha
    return np.mean(tail_losses)

portfolio_returns_optimal = returns_matrix.values @ poids_optimaux
portfolio_returns_final = returns_matrix.values @ omega_final
sharpe_optimal = calculate_sharpe_ratio(portfolio_returns_optimal)
sharpe_final = calculate_sharpe_ratio(portfolio_returns_final)
var_95_optimal = calculate_var(portfolio_returns_optimal, 0.95)
es_95_optimal = calculate_expected_shortfall(portfolio_returns_optimal, 0.95)
var_95_final = calculate_var(portfolio_returns_final, 0.95)
es_95_final = calculate_expected_shortfall(portfolio_returns_final, 0.95)

print("\nSharpe Ratio Markowitz optimal: ", sharpe_optimal)
print("Sharpe Ratio avec ajustement Hurst: ", sharpe_final)

print("\nVaR 95% Markowitz optimal: ", var_95_optimal)
print("Expected Shortfall 95% Markowitz optimal: ", es_95_optimal)

print("\nVaR 95% avec ajustement Hurst: ", var_95_final)
print("Expected Shortfall 95% avec ajustement Hurst: ", es_95_final)

