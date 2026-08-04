"""
Walk-forward multi-fold per il blend Kalman + XGBoost, analogo a
walk_forward_backtest.py ma per la pipeline di integrate_kalman.py invece
che per il vecchio blend Elo + XGBoost.

PERCHE'
-------
integrate_kalman.py valuta un solo fold (train 2016-2023, test 2024). Come
gia' discusso per walk_forward_backtest.py, un solo anno di test ha varianza
alta -- qui ripetiamo lo stesso schema expanding-window (test 2020, 2021,
2022, 2023, 2024) ma per Kalman puro / Kalman+XGBoost / Vegas, cosi' il
confronto con il vecchio blend Elo+XGBoost e' fatto a parita' di
metodologia di valutazione.

NOTA SUL COSTO: per ogni fold serve rigirare l'intero Kalman multi-stagione
da years[0] fino a test_season incluso (i rating mu/sigma sono persistenti
fra stagioni, quindi non si puo' "saltare" al fold giusto senza aver prima
fatto girare tutte le stagioni precedenti). Il Kalman in se' e' economico
(aggiornamenti numpy su dict), il costo vero e' ri-allenare XGBoost per ogni
fold -- stesso ordine di grandezza di walk_forward_backtest.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from data_ingestion import load_games
from stacking_model import train_xgb_margin_model, FEATURE_COLS, NU, SIGMA_GAME
from integrate_kalman import fit_alpha_from_true_kalman_margins
import nfl_model as nm

MIN_TRAIN_SEASONS = 4


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _br(p, y):
    return np.mean((np.array(p) - y) ** 2)


def walk_forward_kalman(feat: pd.DataFrame, games: pd.DataFrame, seasons=range(2016, 2025)):
    """feat: team_week_features.csv gia' caricato. games: da load_games,
    con home_score/away_score gia' presenti (NON ancora rinominati
    home_points/away_points -- lo facciamo qui dentro, come in
    integrate_kalman.main())."""
    seasons = sorted(seasons)
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games = games.rename(columns={"home_score": "home_points", "away_score": "away_points"})
    games = games.sort_values(["season", "week"]).reset_index(drop=True)

    rows = []
    for cut_idx in range(MIN_TRAIN_SEASONS, len(seasons)):
        test_season = seasons[cut_idx]
        years_so_far = seasons[: cut_idx + 1]

        train_feat = feat[feat["season"] < test_season].dropna(subset=["margin"]).copy()
        if len(train_feat) < 50:
            continue

        # --- XGBoost allenato SOLO su stagioni < test_season ---
        model = train_xgb_margin_model(train_feat)
        feat_valid = feat[feat["season"].isin(years_so_far)].dropna(subset=FEATURE_COLS, how="all").copy()
        feat_valid["xgb_margin_pred"] = model.predict(feat_valid[FEATURE_COLS])
        xgb_lookup = {(row.game_id, row.team): row.xgb_margin_pred for row in feat_valid.itertuples()}

        def xgb_predict_fn(game_id, home_team, _lookup=xgb_lookup):
            return _lookup.get((game_id, home_team))

        sub_games = games[games["season"].isin(years_so_far)].reset_index(drop=True)
        teams_names = sorted(set(sub_games["home_team"]) | set(sub_games["away_team"]))

        # --- PASSO 1: run preliminare per i veri margini Kalman out-of-sample ---
        teams = {t: nm.Team(t, conference="NA", division="NA") for t in teams_names}
        _, _, _, kalman_margins = nm.evaluate_multiseason_stacked(
            sub_games, teams, xgb_predict_fn=xgb_predict_fn, alpha=0.5
        )
        alpha, _ = fit_alpha_from_true_kalman_margins(train_feat, kalman_margins)

        # --- PASSO 2: reset team, run finale col vero alpha ---
        teams = {t: nm.Team(t, conference="NA", division="NA") for t in teams_names}
        preds_kalman, preds_blend, _, _ = nm.evaluate_multiseason_stacked(
            sub_games, teams, xgb_predict_fn=xgb_predict_fn, alpha=alpha
        )

        games_list = list(sub_games.itertuples(index=False))
        test_idx = [i for i, g in enumerate(games_list) if g.season == test_season]
        if len(test_idx) < 10:
            continue
        pk_test = [preds_kalman[i] for i in test_idx]
        pb_test = [preds_blend[i] for i in test_idx]

        vegas_test = sub_games[sub_games["season"] == test_season]
        vegas_p = 1 - student_t.cdf(0, df=NU, loc=vegas_test["spread_line"].values, scale=SIGMA_GAME)
        y_test = (vegas_test["home_points"] > vegas_test["away_points"]).astype(int).values

        rows.append(dict(
            test_season=test_season,
            train_seasons=f"{seasons[0]}-{test_season - 1}",
            alpha=alpha,
            n_test=len(test_idx),
            kalman_logloss=nm.log_loss(pk_test), kalman_brier=nm.brier_score(pk_test),
            blend_logloss=nm.log_loss(pb_test), blend_brier=nm.brier_score(pb_test),
            vegas_logloss=_ll(vegas_p, y_test), vegas_brier=_br(vegas_p, y_test),
        ))

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    cols = [c for c in results.columns if c.endswith("_logloss") or c.endswith("_brier")]
    summary = results[cols].agg(["mean", "std"]).T
    summary.index.name = "metric"
    return summary


if __name__ == "__main__":
    nm.set_seed(42)
    seasons = list(range(2016, 2025))
    feat = pd.read_csv("data/team_week_features.csv")
    games = load_games(seasons)

    results = walk_forward_kalman(feat, games, seasons=seasons)
    print(results.to_string(index=False))
    print()
    print(summarize(results).to_string())