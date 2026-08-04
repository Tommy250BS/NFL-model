"""
Backtest walk-forward multi-stagione (invece del singolo split train<2024 /
test=2024 usato finora) + piccola grid search sugli iperparametri principali.

PERCHE' QUESTO SCRIPT
----------------------
Un solo anno di test (2024) da' una stima ad alta varianza delle metriche:
puo' darsi che uno dei modelli abbia semplicemente "azzeccato" o "sbagliato"
quell'anno specifico per rumore, non per qualita' strutturale. Con 9 stagioni
disponibili (2016-2024) possiamo invece fare expanding-window walk-forward:

    fold 1: train 2016-2019, test 2020
    fold 2: train 2016-2020, test 2021
    fold 3: train 2016-2021, test 2022
    fold 4: train 2016-2022, test 2023
    fold 5: train 2016-2023, test 2024

Ogni fold e' un vero held-out test (XGBoost mai allenato su quell'anno, alpha
del blend mai stimato su quell'anno). Riportiamo media e deviazione standard
delle metriche sui fold, che e' molto piu' informativo di un singolo numero.

GRID SEARCH
-----------
ridge_lambda (opponent-adjusted EPA) e qb_weight (Elo QB overlay) sono oggi
scelti a mano. Qui li testiamo su una griglia piccola, usando lo stesso
schema walk-forward, e scegliamo la combinazione con il miglior log loss
medio. ATTENZIONE AL COSTO: ogni combinazione di ridge_lambda richiede di
ricalcolare build_walk_forward_features da zero (la ridge opponent-adjustment
gira settimana per settimana), quindi la grid search e' O(n_lambda) chiamate
alla parte piu' pesante della pipeline. qb_weight invece e' quasi gratis da
testare perche' non richiede di ricostruire le feature (serve solo rigirare
QBAdjustedElo e ricalcolare elo_win_prob) -- per questo lo separiamo dalla
grid su ridge_lambda invece di fare il prodotto cartesiano completo.
"""

import numpy as np
import pandas as pd

from data_ingestion import load_pbp, load_games
from build_features import build_walk_forward_features
from stacking_model import (
    FEATURE_COLS, train_xgb_margin_model, oof_xgb_predictions,
    fit_blend_weight, margin_to_win_prob, spread_to_win_prob,
    log_loss_np, brier_np,
)

MIN_TRAIN_SEASONS = 4  # servono almeno N stagioni di train prima del primo fold di test


def _eval_fold(train, test):
    """Allena su `train`, valuta su `test` (entrambi con FEATURE_COLS, margin,
    home_win, elo_win_prob, spread_line gia' pronti). Ritorna un dict di
    metriche per elo/xgb/blend/vegas."""
    model = train_xgb_margin_model(train)
    xgb_pred_test = model.predict(test[FEATURE_COLS])

    oof_pred_train = oof_xgb_predictions(train)
    alpha = fit_blend_weight(train, oof_pred_train)

    from scipy.stats import t as student_t
    from stacking_model import NU, SIGMA_GAME
    elo_margin_test = np.asarray(
        student_t.ppf(1 - test["elo_win_prob"].clip(0.01, 0.99), df=NU, scale=SIGMA_GAME) * -1
    )
    blend_margin_test = alpha * elo_margin_test + (1 - alpha) * xgb_pred_test

    preds = {
        "elo_solo": margin_to_win_prob(elo_margin_test),
        "xgb_solo": margin_to_win_prob(xgb_pred_test),
        "blend": margin_to_win_prob(blend_margin_test),
        "vegas": spread_to_win_prob(test["spread_line"].values),
    }
    y = test["home_win"].values
    out = {"alpha": alpha, "n_test": len(test)}
    for name, p in preds.items():
        out[f"{name}_logloss"] = log_loss_np(p, y)
        out[f"{name}_brier"] = brier_np(p, y)
    return out


def walk_forward_backtest(feat: pd.DataFrame, seasons=range(2016, 2025)):
    """feat: output di build_walk_forward_features (o equivalente caricato da
    CSV), con colonna 'season'. Ritorna un DataFrame con una riga per fold."""
    feat = feat.dropna(subset=["margin", "home_win"]).copy()
    seasons = sorted(seasons)
    first_test_season = seasons[MIN_TRAIN_SEASONS]

    rows = []
    for test_season in seasons[MIN_TRAIN_SEASONS:]:
        train = feat[feat["season"] < test_season]
        test = feat[feat["season"] == test_season]
        if len(train) < 50 or len(test) < 10:
            continue
        metrics = _eval_fold(train, test)
        metrics["test_season"] = test_season
        metrics["train_seasons"] = f"{seasons[0]}-{test_season - 1}"
        rows.append(metrics)

    results = pd.DataFrame(rows)
    return results


def summarize(results: pd.DataFrame):
    """Media/std delle metriche sui fold, per modello."""
    cols = [c for c in results.columns if c.endswith("_logloss") or c.endswith("_brier")]
    summary = results[cols].agg(["mean", "std"]).T
    summary.index.name = "metric"
    return summary


def grid_search_ridge_lambda(years, lambdas=(10.0, 25.0, 50.0, 75.0), seasons_for_eval=None):
    """Rifa' build_walk_forward_features per ogni ridge_lambda e valuta con
    walk_forward_backtest, scegliendo il lambda con miglior log loss medio
    del blend. Riusa pbp/games caricati una sola volta (il download e'
    identico per ogni lambda, solo la ridge cambia)."""
    pbp = load_pbp(years)
    games = load_games(years)
    seasons_for_eval = seasons_for_eval or years

    results_by_lambda = {}
    for lam in lambdas:
        feat = build_walk_forward_features(years, ridge_lambda=lam, _pbp=pbp, _games=games)
        wf = walk_forward_backtest(feat, seasons=seasons_for_eval)
        if wf.empty:
            continue
        mean_ll = wf["blend_logloss"].mean()
        results_by_lambda[lam] = mean_ll
        print(f"ridge_lambda={lam:>6.1f}  blend log loss medio (walk-forward)={mean_ll:.4f}")

    best_lambda = min(results_by_lambda, key=results_by_lambda.get)
    return best_lambda, results_by_lambda


def grid_search_qb_weight(base_feat_no_elo_cols, years, weights=(0.3, 0.45, 0.6, 0.75)):
    """qb_weight cambia solo l'Elo (qb_elo, opp_qb_elo, elo_win_prob), non le
    altre feature -- quindi qui ricostruiamo l'intera feature table per
    ciascun peso (l'Elo e' calcolato dentro lo stesso ciclo walk-forward di
    build_walk_forward_features insieme al resto, non e' facilmente
    separabile senza duplicare il ciclo). Piu' costoso di quanto servirebbe
    in teoria, ma piu' semplice e meno rischioso di duplicare la logica
    walk-forward in due posti diversi."""
    pbp = load_pbp(years)
    games = load_games(years)

    results_by_weight = {}
    for w in weights:
        feat = build_walk_forward_features(years, qb_weight=w, _pbp=pbp, _games=games)
        wf = walk_forward_backtest(feat, seasons=years)
        if wf.empty:
            continue
        mean_ll = wf["blend_logloss"].mean()
        results_by_weight[w] = mean_ll
        print(f"qb_weight={w:.2f}  blend log loss medio (walk-forward)={mean_ll:.4f}")

    best_weight = min(results_by_weight, key=results_by_weight.get)
    return best_weight, results_by_weight


if __name__ == "__main__":
    feat = pd.read_csv("data/team_week_features.csv")
    results = walk_forward_backtest(feat, seasons=range(2016, 2025))
    print(results.to_string(index=False))
    print()
    print(summarize(results).to_string())

    print("\nGrid search ridge_lambda (rifa' le feature per ogni valore, puo' richiedere tempo)...")
    best_lambda, _ = grid_search_ridge_lambda([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    print(f"\nMiglior ridge_lambda: {best_lambda}")