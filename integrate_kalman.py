"""
Chiude il loop: fa girare il filtro di Kalman (nfl_model.py, con i fix
multi-stagione) su 2016-2024, lo combina con le predizioni XGBoost (allenato
SOLO su 2016-2023, mai su 2024) e confronta Kalman puro / blend / Vegas sulla
stagione 2024 tenuta da parte.

FIX rispetto alla versione precedente: alpha (il peso del Kalman nel blend)
e' ora stimato sui VERI margini impliciti dal filtro di Kalman per ciascuna
partita del train (2016-2023), non su un proxy ricavato dall'Elo. Questi
margini sono automaticamente "out-of-sample" per costruzione: il Kalman a
walk-forward calcola A.mu - B.mu PRIMA di aggiornare i rating con quella
partita, quindi non c'e' leakage da correggere via k-fold come serviva per
XGBoost (che invece e' un modello statico allenato una volta su tutti i dati).
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from data_ingestion import load_games
from stacking_model import train_xgb_margin_model, oof_xgb_predictions, FEATURE_COLS, NU, SIGMA_GAME
import nfl_model as nm

nm.set_seed(42)


def fit_alpha_from_true_kalman_margins(train_feat, kalman_margins_dict):
    """Stima alpha usando i VERI margini Kalman (via game_id) invece del
    proxy Elo. train_feat deve avere 'game_id', 'margin', e le colonne di
    FEATURE_COLS gia' pronte."""
    train_feat = train_feat.copy()
    train_feat["kalman_margin"] = train_feat["game_id"].map(kalman_margins_dict)
    train_feat = train_feat.dropna(subset=["kalman_margin"])

    oof_xgb = oof_xgb_predictions(train_feat)
    y = train_feat["margin"].values
    x1 = train_feat["kalman_margin"].values
    x2 = oof_xgb
    diff = x1 - x2
    num = np.sum((y - x2) * diff)
    den = np.sum(diff ** 2)
    alpha = float(np.clip(num / den if den > 0 else 0.5, 0.0, 1.0))
    return alpha, train_feat


def main():
    feat = pd.read_csv("data/team_week_features.csv")
    games = load_games([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games = games.rename(columns={"home_score": "home_points", "away_score": "away_points"})
    games = games.sort_values(["season", "week"]).reset_index(drop=True)

    teams_names = sorted(set(games["home_team"]) | set(games["away_team"]))
    teams = {t: nm.Team(t, conference="NA", division="NA") for t in teams_names}

    # --- XGBoost allenato SOLO su 2016-2023 ---
    train_feat = feat[feat["season"] < 2024].dropna(subset=["margin"]).copy()
    model = train_xgb_margin_model(train_feat)

    feat_valid = feat.dropna(subset=FEATURE_COLS, how="all").copy()
    feat_valid["xgb_margin_pred"] = model.predict(feat_valid[FEATURE_COLS])
    xgb_lookup = {(row.game_id, row.team): row.xgb_margin_pred for row in feat_valid.itertuples()}

    def xgb_predict_fn(game_id, home_team):
        return xgb_lookup.get((game_id, home_team))

    # --- PASSO 1: run preliminare con alpha placeholder, solo per raccogliere
    # i veri margini Kalman per-partita (alpha non influenza l'update, quindi
    # questi margini sono corretti indipendentemente dal valore placeholder) ---
    _, _, _, kalman_margins = nm.evaluate_multiseason_stacked(
        games, teams, xgb_predict_fn=xgb_predict_fn, alpha=0.5
    )

    # --- PASSO 2: stima alpha sui veri margini Kalman out-of-sample (train) ---
    alpha, _ = fit_alpha_from_true_kalman_margins(train_feat, kalman_margins)
    print(f"alpha (peso Kalman, stimato su veri margini Kalman): {alpha:.2f}  |  peso XGBoost: {1 - alpha:.2f}")

    # --- PASSO 3: reset dei team e run finale col vero alpha, per la
    # valutazione honest sul 2024 ---
    teams = {t: nm.Team(t, conference="NA", division="NA") for t in teams_names}
    preds_kalman, preds_blend, _, _ = nm.evaluate_multiseason_stacked(
        games, teams, xgb_predict_fn=xgb_predict_fn, alpha=alpha
    )

    games_list = list(games.itertuples(index=False))
    test_idx = [i for i, g in enumerate(games_list) if g.season == 2024]
    pk_test = [preds_kalman[i] for i in test_idx]
    pb_test = [preds_blend[i] for i in test_idx]

    vegas_test = games[games["season"] == 2024].copy()
    vegas_p = 1 - student_t.cdf(0, df=NU, loc=vegas_test["spread_line"].values, scale=SIGMA_GAME)
    y_test = (vegas_test["home_points"] > vegas_test["away_points"]).astype(int).values

    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def br(p, y):
        return np.mean((np.array(p) - y) ** 2)

    print(f"\n{'Modello':<20}{'LogLoss':>10}{'Brier':>10}")
    print(f"{'Kalman puro':<20}{nm.log_loss(pk_test):>10.4f}{nm.brier_score(pk_test):>10.4f}")
    print(f"{'Kalman + XGBoost':<20}{nm.log_loss(pb_test):>10.4f}{nm.brier_score(pb_test):>10.4f}")
    print(f"{'Vegas':<20}{ll(vegas_p, y_test):>10.4f}{br(vegas_p, y_test):>10.4f}")

    return alpha


if __name__ == "__main__":
    main()