"""
Core di stacking: XGBoost stima il margine atteso di partita dalle feature
avanzate; questo si combina con il margine implicito dal rating (Elo qui,
Kalman quando integrato in main_model.py) per dare la media finale della
t-Student usata in simulazione.

Split temporale pulito: train su 2023, test su 2024. Niente cross-validation
mescolata fra stagioni, altrimenti si "vede il futuro".

Confrontiamo tre modelli sullo stesso test set (2024):
  1. Solo Elo QB-adjusted (elo_win_prob)
  2. Solo XGBoost (xgb_margin -> win prob)
  3. Blend Elo + XGBoost (media pesata, peso stimato via ridge sul train)
E li mettiamo a confronto con la probabilita' implicita dalla linea Vegas
(spread_line), che è il benchmark che conta davvero.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import t as student_t

NU = 3
SIGMA_GAME = 12.5  # stessi parametri della distribuzione margine di main_model.py


def margin_to_win_prob(margin, home=True, home_field_adv=2.2):
    """Converte un margine atteso in probabilita' di vittoria home usando la
    stessa t-Student (NU, SIGMA_GAME) del motore principale, per coerenza."""
    return 1 - student_t.cdf(0, df=NU, loc=margin, scale=SIGMA_GAME)


def spread_to_win_prob(spread_line, home_field_adv=2.2):
    """Converte lo spread Vegas (convenzione nflverse: positivo = home
    favorita di quei punti) in probabilita' di vittoria home, stessa
    famiglia t-Student per confronto equo (non e' la formula esatta usata
    dai book, ma è coerente con come valutiamo gli altri due modelli)."""
    return 1 - student_t.cdf(0, df=NU, loc=spread_line, scale=SIGMA_GAME)


def log_loss_np(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_np(p, y):
    return np.mean((p - y) ** 2)


FEATURE_COLS = [
    "off_epa_adj", "def_epa_adj", "opp_off_epa_adj", "opp_def_epa_adj",
    "cpoe_reg", "qb_elo", "opp_qb_elo",
    "rest_diff", "is_dome", "temp", "wind", "wind_effective",
    "off_epa_3rd_down_roll", "off_epa_redzone_roll", "turnover_diff_roll",
    "plays_roll", "pace_roll",
    "travel_distance", "tz_diff",
    "drive_epd_off", "drive_epd_def",
    "injury_burden", "opp_injury_burden", "qb_out", "opp_qb_out",
]


def train_xgb_margin_model(train_df: pd.DataFrame) -> xgb.XGBRegressor:
    X = train_df[FEATURE_COLS]
    y = train_df["margin"]
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,          # profondita' bassa: ~280 partite/anno, serve regolarizzazione forte
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        objective="reg:squarederror",
        missing=np.nan,
    )
    model.fit(X, y)
    return model


def oof_xgb_predictions(train_df: pd.DataFrame, n_folds=5, seed=42) -> np.ndarray:
    """Predizioni XGBoost OUT-OF-FOLD sul train set: per ogni fold, il modello
    e' allenato SENZA quelle partite e le predice. Necessario per stimare il
    peso del blend senza leakage -- le predizioni in-sample di un modello
    allenato sugli stessi dati sono artificialmente ottime e falsano il peso
    (e' esattamente il bug che ha dato alpha=0 nel primo tentativo)."""
    rng = np.random.default_rng(seed)
    n = len(train_df)
    fold_id = rng.integers(0, n_folds, size=n)
    oof_pred = np.zeros(n)
    idx = train_df.index
    for k in range(n_folds):
        test_mask = fold_id == k
        train_mask = ~test_mask
        m = train_xgb_margin_model(train_df.iloc[train_mask])
        oof_pred[test_mask] = m.predict(train_df.iloc[test_mask][FEATURE_COLS])
    return oof_pred


def fit_blend_weight(train_df: pd.DataFrame, oof_xgb_pred) -> float:
    """Stima il peso ottimale alpha per: margin ~ alpha * elo_margin + (1-alpha) * xgb_margin,
    usando predizioni XGBoost out-of-fold (vedi oof_xgb_predictions) per
    evitare che il fit del peso sia drogato dall'overfitting di XGBoost."""
    elo_margin_train = student_t.ppf(1 - train_df["elo_win_prob"].clip(0.01, 0.99), df=NU, scale=SIGMA_GAME) * -1
    y = train_df["margin"].values
    x1 = np.asarray(elo_margin_train)
    x2 = oof_xgb_pred
    diff = x1 - x2
    num = np.sum((y - x2) * diff)
    den = np.sum(diff ** 2)
    alpha = np.clip(num / den if den > 0 else 0.5, 0.0, 1.0)
    return float(alpha)


def run_backtest(feat_path="data/team_week_features.csv"):
    feat = pd.read_csv(feat_path)
    feat = feat.dropna(subset=["margin", "home_win"])  # solo partite gia' giocate

    train = feat[feat["season"] < 2024].copy()
    test = feat[feat["season"] == 2024].copy()
    print(f"Train (2016-2023): {len(train)} partite | Test (2024): {len(test)} partite")

    model = train_xgb_margin_model(train)
    xgb_pred_test = model.predict(test[FEATURE_COLS])

    oof_pred_train = oof_xgb_predictions(train)
    alpha = fit_blend_weight(train, oof_pred_train)
    print(f"Peso blend stimato su train: alpha_elo={alpha:.2f}  alpha_xgb={1-alpha:.2f}")

    elo_margin_test = np.asarray(student_t.ppf(1 - test["elo_win_prob"].clip(0.01, 0.99), df=NU, scale=SIGMA_GAME) * -1)
    blend_margin_test = alpha * elo_margin_test + (1 - alpha) * xgb_pred_test

    results = {}
    results["elo_solo"] = margin_to_win_prob(elo_margin_test)
    results["xgb_solo"] = margin_to_win_prob(xgb_pred_test)
    results["blend"] = margin_to_win_prob(blend_margin_test)
    results["vegas"] = spread_to_win_prob(test["spread_line"].values)

    y = test["home_win"].values
    print(f"\n{'Modello':<15}{'LogLoss':>10}{'Brier':>10}")
    for name, p in results.items():
        print(f"{name:<15}{log_loss_np(p, y):>10.4f}{brier_np(p, y):>10.4f}")

    return model, alpha, results


if __name__ == "__main__":
    run_backtest()
