"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  CONFIGURATION — Paramètres utilisateur + système                              ║
║                                                                                ║
║  L'utilisateur peut configurer :                                               ║
║    • Horizon d'investissement (court / moyen / long)                           ║
║    • Aversion au risque (1 à 5)                                                ║
║    • Budget, levier max, vol cible                                             ║
║    • Paires à inclure/exclure                                                  ║
║    • Fréquence de rebalancement                                                ║
║    • Pipeline de modèles (via PIPELINE_REGISTRY)                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Chemins ────────────────────────────────────────────────────────────────
DATA_DIR    = "."
OUTPUT_DIR  = "results"
FILE_PRICES = "/Users/thomasbelleville/fx_optimization/data_forex_prices.csv"
FILE_RATES  = "/Users/thomasbelleville/fx_optimization/data_fred_rates.csv"

# ─── Période ────────────────────────────────────────────────────────────────
DATA_START     = "2010-01-01"
BACKTEST_START = "2020-01-01"
TRAIN_END      = "2019-12-31"
TRADING_DAYS   = 252

# ─── Univers FX (17 paires) ─────────────────────────────────────────────────
PAIR_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "AUDJPY": "AUDJPY=X",
    "NZDJPY": "NZDJPY=X",
    "USDMXN": "USDMXN=X", "USDNOK": "USDNOK=X", "USDSEK": "USDSEK=X",
    "USDSGD": "USDSGD=X", "USDTRY": "USDTRY=X", "USDZAR": "USDZAR=X",
}
TICKER_TO_PAIR = {v: k for k, v in PAIR_MAP.items()}


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE REGISTRY
# ═════════════════════════════════════════════════════════════════════════════
# Chaque entrée décrit un pipeline complet pour un horizon + profil donné.
#
# Champs :
#   model       : identifiant du modèle (utilisé dans ml_models.py)
#   sizing      : méthode de sizing des positions
#   bl_mv       : utiliser Black-Litterman + Markowitz ?
#   ep          : utiliser Entropy Pooling pour la fusion ?
#   rebal_freq  : fréquence de rebalancement en jours
#   leverage    : multiplicateur de levier relatif au max_leverage du profil
#   description : texte affiché dans le menu
#   sharpe_est  : Sharpe estimé (indicatif, basé sur littérature / tests internes)
#   papers      : références académiques
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_REGISTRY = {

    # ── COURT TERME (1-5 jours) ───────────────────────────────────────────────
    # Prédiction : direction J+1 à J+5
    # Signal dominant : mean-reversion, microstructure
    # Rebalancement : tous les 2-3 jours
    'CT': {
        'conservateur': {
            'model':       'ridge_ct',
            'sizing':      'vol_target',
            'bl_mv':       False,
            'ep':          False,
            'rebal_freq':  3,
            'target_horizon': 1,       # jours forward pour la prédiction
            'leverage':    0.3,
            'description': 'Ridge CT — linéaire, turnover 3j, prédiction J+1',
            'sharpe_est':  1.2,
            'papers':      ['Jansen ML4T ch.7 — Ridge/Lasso for return forecasting'],
        },
        'equilibre': {
            'model':       'lgbm_ct',
            'sizing':      'tanh',
            'bl_mv':       False,
            'ep':          False,
            'rebal_freq':  2,
            'target_horizon': 1,
            'leverage':    0.4,
            'description': 'LightGBM CT — features 1-10j, prédiction J+1, rebal 2j',
            'sharpe_est':  2.0,
            'papers':      ['Jansen ML4T ch.12 — Gradient Boosting for intraday/CT'],
        },
        'agressif': {
            'model':       'lgbm_hurst_ct',
            'sizing':      'tanh',
            'bl_mv':       False,
            'ep':          False,
            'rebal_freq':  2,
            'target_horizon': 1,
            'leverage':    0.5,
            'description': 'LightGBM+Hurst CT — filtre régime mean-rev, prédiction J+1',
            'sharpe_est':  2.3,
            'papers':      [
                'Jansen ML4T ch.12 — Gradient Boosting',
                'Garcin (2021) — Forecasting with fBm',
            ],
        },
    },

    # ── MOYEN TERME (1-6 mois, 21-126 jours) ─────────────────────────────────
    # Prédiction : rendement à 21j ou 63j
    # Signal dominant : momentum 3-6 mois, carry, réversion macro lente
    # Rebalancement : toutes les 2-4 semaines
    'MT': {
        'conservateur': {
            'model':       'gb_only',
            'sizing':      'tanh',
            'bl_mv':       False,
            'ep':          False,
            'rebal_freq':  21,          # rebal mensuel
            'target_horizon': 21,       # prédiction à 21j (1 mois)
            'leverage':    0.4,
            'description': 'GB seul — prédiction rendement 21j, rebal mensuel',
            'sharpe_est':  1.8,
            'papers':      ['Jansen ML4T ch.11-12 — RF & GB for equity/FX signals'],
        },
        'equilibre': {
            'model':       'ensemble_mt',
            'sizing':      'tanh',
            'bl_mv':       False,
            'ep':          False,
            'rebal_freq':  10,          # rebal bi-mensuel
            'target_horizon': 21,       # prédiction à 21j
            'leverage':    0.5,
            'description': 'Ensemble GB+Transformer+LSTM — prédiction 21j, rebal 2 sem.',
            'sharpe_est':  3.2,
            'papers':      [
                'Jansen ML4T ch.19 — RNN/LSTM for time series',
                'Vaswani et al. (2017) — Attention is all you need',
            ],
        },
        'agressif': {
            'model':       'ensemble_ep_mt',
            'sizing':      'bl_mv',
            'bl_mv':       True,
            'ep':          True,
            'rebal_freq':  10,
            'target_horizon': 21,
            'leverage':    0.6,
            'description': 'Ensemble + EP calibré + BL/MV — prédiction 21j, corrélations gérées',
            'sharpe_est':  3.5,
            'papers':      [
                'Meucci (2008) — Entropy Pooling',
                'DMVFEP (2024) — Deep Multi-View Factor Entropy Pooling',
            ],
        },
    },

    # ── LONG TERME (6 mois - 2 ans, 126-504 jours) ───────────────────────────
    # Prédiction : rendement à 63j ou 126j
    # Signal dominant : carry trade pur, momentum 12 mois, valeur PPA/REER
    # Rebalancement : mensuel / trimestriel
    # Ref : Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum (CTA style)
    #        Koijen et al. (2012) — Carry trades in FX
    'LT': {
        'conservateur': {
            'model':       'carry_mom_bl',
            'sizing':      'bl_mv',
            'bl_mv':       True,
            'ep':          False,
            'rebal_freq':  63,          # rebal trimestriel
            'target_horizon': 63,       # prédiction à 63j (3 mois)
            'leverage':    0.4,
            'description': 'Carry + Momentum 12m + BL/MV — rebal trimestriel',
            'sharpe_est':  0.8,
            'papers':      [
                'Black & Litterman (1992)',
                'Koijen et al. (2012) — Carry trades in FX',
                'Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum',
            ],
        },
        'equilibre': {
            'model':       'rf_lt',
            'sizing':      'bl_mv',
            'bl_mv':       True,
            'ep':          False,
            'rebal_freq':  42,          # rebal ~6 semaines
            'target_horizon': 63,       # prédiction à 63j
            'leverage':    0.5,
            'description': 'Random Forest LT + BL/MV — prédiction 63j, rebal 6 sem.',
            'sharpe_est':  1.5,
            'papers':      [
                'Jansen ML4T ch.11 — Random Forests for return prediction',
                'Black & Litterman (1992)',
            ],
        },
        'agressif': {
            'model':       'mlp_macro_lt',
            'sizing':      'bl_mv',
            'bl_mv':       True,
            'ep':          True,
            'rebal_freq':  21,          # rebal mensuel (agressif = plus réactif)
            'target_horizon': 63,       # prédiction à 63j
            'leverage':    0.6,
            'description': 'MLP macro + BL/MV + EP — prédiction 63j, rebal mensuel',
            'sharpe_est':  2.0,
            'papers':      [
                'Jansen ML4T ch.17 — Deep Learning for Trading',
                'Meucci (2008) — Entropy Pooling',
                'Gu, Kelly & Xiu (2020) — Empirical Asset Pricing via ML',
            ],
        },
    },

    # ── ENTROPY POOLING STANDALONE ────────────────────────────────────────────
    'EP': {
        'conservateur': {
            'model':       'ep_signals',
            'sizing':      'cvar',
            'bl_mv':       False,
            'ep':          True,
            'rebal_freq':  10,
            'leverage':    0.3,
            'description': 'EP standalone CVaR conservateur — vues brutes sans ML',
            'sharpe_est':  1.0,
            'papers':      ['Meucci (2008) — Entropy Pooling', 'Rockafellar & Uryasev (2000) — CVaR'],
        },
        'equilibre': {
            'model':       'ep_ml',
            'sizing':      'cvar',
            'bl_mv':       False,
            'ep':          True,
            'rebal_freq':  5,
            'leverage':    0.5,
            'description': 'EP + ML views — EP calibré sur signaux GB+Transformer',
            'sharpe_est':  2.5,
            'papers':      [
                'Meucci (2008) — Entropy Pooling',
                'DMVFEP (2024) — Deep Multi-View Factor Entropy Pooling',
            ],
        },
        'agressif': {
            'model':       'ep_ml_full',
            'sizing':      'cvar',
            'bl_mv':       False,
            'ep':          True,
            'rebal_freq':  5,
            'leverage':    0.7,
            'description': 'EP full ML views — tous les modèles comme vues pondérées par accuracy',
            'sharpe_est':  3.0,
            'papers':      [
                'Meucci (2008) — Entropy Pooling',
                'DMVFEP (2024)',
                'Ardia et al. (2017) — Entropic Portfolio Optimization',
            ],
        },
    },

    # ── COMBINÉ (tous horizons) ────────────────────────────────────────────────
    'COMBINED': {
        'conservateur': {
            'model':       'combined_ep',
            'sizing':      'vol_target',
            'bl_mv':       True,
            'ep':          True,
            'rebal_freq':  5,
            'leverage':    0.5,
            'description': 'EP fusion CT+MT+LT conservateur — vol target 5%',
            'sharpe_est':  1.5,
            'papers':      ['Meucci (2008)', 'Black & Litterman (1992)'],
        },
        'equilibre': {
            'model':       'combined_ep',
            'sizing':      'vol_target',
            'bl_mv':       True,
            'ep':          True,
            'rebal_freq':  5,
            'leverage':    0.75,
            'description': 'EP fusion CT+MT+LT équilibré — vol target 10% (pipeline actuel)',
            'sharpe_est':  2.0,
            'papers':      ['Meucci (2008)', 'Black & Litterman (1992)'],
        },
        'agressif': {
            'model':       'combined_ep_full',
            'sizing':      'vol_target',
            'bl_mv':       True,
            'ep':          True,
            'rebal_freq':  5,
            'leverage':    1.0,
            'description': 'EP full fusion + ML vues calibrées — vol target 18%',
            'sharpe_est':  2.8,
            'papers':      [
                'Meucci (2008)',
                'DMVFEP (2024)',
                'Black & Litterman (1992)',
            ],
        },
    },
}


def get_pipeline(horizon: str, profil: str) -> dict:
    """Retourne le pipeline recommandé pour un horizon et profil donnés."""
    h = horizon.upper()
    p = profil.lower()
    if h not in PIPELINE_REGISTRY:
        raise ValueError(f"Horizon inconnu: {horizon}. Choix: {list(PIPELINE_REGISTRY.keys())}")
    if p not in PIPELINE_REGISTRY[h]:
        raise ValueError(f"Profil inconnu: {profil}. Choix: {list(PIPELINE_REGISTRY[h].keys())}")
    return PIPELINE_REGISTRY[h][p]


def print_pipeline_recommendations(profil: str):
    """Affiche toutes les recommandations pour un profil donné."""
    p = profil.lower()
    print(f"\n{'='*80}")
    print(f"  RECOMMANDATIONS PIPELINE — Profil : {profil.upper()}")
    print(f"{'='*80}")
    horizons_labels = {
        'CT': 'COURT TERME  (1-5 jours, pred J+1)',
        'MT': 'MOYEN TERME  (1-6 mois, pred 21j)',
        'LT': 'LONG TERME   (6m-2 ans, pred 63j)',
        'EP': 'ENTROPY POOLING (standalone)',
        'COMBINED': 'COMBINE (tous horizons)',
    }
    for h, label in horizons_labels.items():
        if h in PIPELINE_REGISTRY and p in PIPELINE_REGISTRY[h]:
            cfg = PIPELINE_REGISTRY[h][p]
            print(f"\n  ── {label} ──")
            print(f"     Modèle     : {cfg['model']}")
            print(f"     Sizing     : {cfg['sizing']}")
            print(f"     BL/MV      : {'Oui' if cfg['bl_mv'] else 'Non'}")
            print(f"     EP         : {'Oui' if cfg['ep'] else 'Non'}")
            print(f"     Rebal      : tous les {cfg['rebal_freq']}j")
            print(f"     Levier rel : {cfg['leverage']:.0%} du max")
            print(f"     Sharpe est.: ~{cfg['sharpe_est']:.1f}")
            print(f"     → {cfg['description']}")
            if cfg['papers']:
                print(f"     Réf.       : {cfg['papers'][0]}")


# ═════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES UTILISATEUR
# ═════════════════════════════════════════════════════════════════════════════

class UserConfig:
    """Configuration utilisateur — tous les paramètres ajustables."""

    def __init__(self):
        # ── Profil de risque ──
        self.risk_aversion  = 2.5
        self.target_vol     = 0.10
        self.max_leverage   = 1.5
        self.max_drawdown   = 0.15
        self.budget         = 1_000_000

        # ── Horizon ──
        self.horizon_weights = {
            'short':  0.30,
            'medium': 0.40,
            'long':   0.30,
        }

        # ── Pipeline sélectionné (défaut = équilibré) ──
        self.ct_pipeline   = 'equilibre'
        self.mt_pipeline   = 'equilibre'
        self.lt_pipeline   = 'equilibre'
        self.ep_pipeline   = 'equilibre'

        # ── Coûts de transaction ──
        self.transaction_cost = 0.0003

        # ── Paires sélectionnées ──
        self.selected_pairs = None

        # ── Walk-forward ──
        self.wf_train_window  = 756      # 3 ans de training (plus stable pour LT)
        self.wf_retrain_freq  = 63       # retrain trimestriel

        # ── Wavelet ──
        self.wavelet_method   = "sure"
        self.wavelet_name     = "db4"
        self.wavelet_level    = 4

        # ── Lookbacks par horizon ──
        self.lookback_cov          = 252   # covariance sur 1 an (plus stable)
        self.lookback_momentum_lt  = 252   # momentum LT : 12 mois (Moskowitz 2012)
        self.lookback_momentum_mt  = 63    # momentum MT : 3 mois
        self.lookback_carry        = 252   # carry annualisé
        self.lookback_momentum     = 252   # rétrocompat
        self.max_weight            = 0.25

        # ── Targets par horizon ──
        self.target_ct  = 1    # J+1  (court terme)
        self.target_mt  = 21   # J+21 (moyen terme, 1 mois)
        self.target_lt  = 63   # J+63 (long terme, 3 mois)

        # ── Hurst ──
        self.use_hurst        = True

        # ── HMM régimes (remplace le z-score simple) ──
        self.use_hmm_regimes  = True
        self.hmm_n_states     = 2   # 2 états : low-vol / high-vol

        # ── GARCH vol forecast ──
        self.use_garch        = True

    def set_profile(self, profile: str):
        profiles = {
            'conservateur': {
                'risk_aversion': 4.0, 'target_vol': 0.05,
                'max_leverage': 1.0, 'max_drawdown': 0.08,
                'horizon_weights': {'short': 0.15, 'medium': 0.30, 'long': 0.55},
                'ct_pipeline': 'conservateur', 'mt_pipeline': 'conservateur',
                'lt_pipeline': 'conservateur', 'ep_pipeline': 'conservateur',
            },
            'equilibre': {
                'risk_aversion': 2.5, 'target_vol': 0.10,
                'max_leverage': 1.5, 'max_drawdown': 0.15,
                'horizon_weights': {'short': 0.30, 'medium': 0.40, 'long': 0.30},
                'ct_pipeline': 'equilibre', 'mt_pipeline': 'equilibre',
                'lt_pipeline': 'equilibre', 'ep_pipeline': 'equilibre',
            },
            'agressif': {
                'risk_aversion': 1.5, 'target_vol': 0.18,
                'max_leverage': 2.0, 'max_drawdown': 0.25,
                'horizon_weights': {'short': 0.45, 'medium': 0.35, 'long': 0.20},
                'ct_pipeline': 'agressif', 'mt_pipeline': 'agressif',
                'lt_pipeline': 'agressif', 'ep_pipeline': 'agressif',
            },
        }
        if profile in profiles:
            for k, v in profiles[profile].items():
                setattr(self, k, v)

    def get_active_pipelines(self) -> dict:
        """Retourne les pipelines actifs pour le profil courant."""
        return {
            'CT':       get_pipeline('CT',       self.ct_pipeline),
            'MT':       get_pipeline('MT',       self.mt_pipeline),
            'LT':       get_pipeline('LT',       self.lt_pipeline),
            'EP':       get_pipeline('EP',       self.ep_pipeline),
            'COMBINED': get_pipeline('COMBINED', self.mt_pipeline),
        }

    def summary(self) -> str:
        hw = self.horizon_weights
        return (
            f"  Aversion au risque : {self.risk_aversion}\n"
            f"  Vol cible          : {self.target_vol:.0%}\n"
            f"  Levier max         : {self.max_leverage:.1f}x\n"
            f"  DD max toléré      : {self.max_drawdown:.0%}\n"
            f"  Coût transaction   : {self.transaction_cost*10000:.0f} bps\n"
            f"  Horizons           : CT={hw['short']:.0%}  MT={hw['medium']:.0%}  LT={hw['long']:.0%}\n"
            f"  Pipelines          : CT={self.ct_pipeline}  MT={self.mt_pipeline}  LT={self.lt_pipeline}\n"
            f"  Walk-forward       : {self.wf_train_window}j train, retrain /{self.wf_retrain_freq}j\n"
            f"  Wavelet            : {self.wavelet_name} L{self.wavelet_level} ({self.wavelet_method})\n"
            f"  HMM régimes        : {'Oui' if self.use_hmm_regimes else 'Non'}\n"
            f"  GARCH vol          : {'Oui' if self.use_garch else 'Non'}"
        )


# Instance globale
USER = UserConfig()


# ─── Features ML ────────────────────────────────────────────────────────────

# Court terme (1-5j) — fenêtres courtes, mean-reversion, microstructure
FEATURE_COLS_CT = [
    "Return_1d", "Return_5d",
    "RSI", "ATR_14", "Bollinger_Pct",
    "RealizedVol_21d",
    "Return_Clean_1d", "Return_Clean_5d",
    "Shannon_Entropy", "Renyi_Entropy",
    "Hurst", "Hurst_Prediction",
    "Momentum_CS",
]

# Moyen terme (1-6 mois) — fenêtres intermédiaires, momentum 3m, carry
FEATURE_COLS_MT = [
    "Return_1d", "Return_5d", "Return_21d",
    "SMA_21", "SMA_63",
    "MACD", "RSI", "ATR_14", "Bollinger_Pct",
    "RealizedVol_21d", "RealizedVol_63d",
    "Return_Clean_1d", "Return_Clean_5d", "MACD_Clean",
    "Shannon_Entropy", "Renyi_Entropy",
    "Hurst", "Fractal_Dimension", "Hurst_Prediction",
    "Momentum_CS", "Momentum_3m",
]

# Long terme (6m-2 ans) — fenêtres longues, carry, momentum 12m, macro
FEATURE_COLS_LT = [
    "Return_21d", "Return_63d",
    "SMA_63", "SMA_126",
    "RealizedVol_63d", "RealizedVol_126d",
    "MACD_Clean",
    "Hurst", "Fractal_Dimension",
    "Shannon_Entropy",
    "Momentum_CS", "Momentum_3m", "Momentum_12m",
    "Carry_Raw",
]

# Rétrocompat
FEATURE_COLS_BASE = FEATURE_COLS_MT
FEATURE_COLS      = FEATURE_COLS_MT

# ─── ML ─────────────────────────────────────────────────────────────────────
LSTM_LOOKBACK = 40
LSTM_EPOCHS   = 25
LSTM_BATCH    = 32

TRANSFORMER_D_MODEL  = 64
TRANSFORMER_NHEAD    = 4
TRANSFORMER_LAYERS   = 2
TRANSFORMER_DIM_FF   = 128
TRANSFORMER_DROPOUT  = 0.1
TRANSFORMER_EPOCHS   = 20
TRANSFORMER_LR       = 0.001
TRANSFORMER_BATCH    = 32

# TCN (Temporal Convolutional Network)
TCN_FILTERS      = 64
TCN_KERNEL_SIZE  = 3
TCN_DILATIONS    = [1, 2, 4, 8]
TCN_EPOCHS       = 20
TCN_BATCH        = 32

# ─── Création des dossiers ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Thème graphique ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',  'axes.facecolor': '#0f0f0f',
    'axes.edgecolor': '#333333',    'axes.labelcolor': '#cccccc',
    'text.color': '#cccccc',        'xtick.color': '#999999',
    'ytick.color': '#999999',       'grid.color': '#1a1a1a',
    'font.family': 'monospace',     'font.size': 9,
})
C = {
    'primary': '#00d4ff',  'secondary': '#00ff88',  'accent': '#ff6b35',
    'warning': '#ffd700',  'danger': '#ff3366',     'purple': '#a855f7',
    'bg': '#0a0a0a',       'panel': '#0f0f0f',      'text': '#cccccc',
    'dim': '#666666',      'bull': '#00ff88',        'bear': '#ff3366',
    'neutral': '#ffd700',
}