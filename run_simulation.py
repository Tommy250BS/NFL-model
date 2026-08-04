"""
Orchestrazione finale, equivalente aggiornato del run_sim.py originale.

Fa girare il Kalman multi-stagione (2016-2024) con blend XGBoost per arrivare
ai rating di fine regular-season 2024, usa i Record della sola stagione 2024
(non cumulati) per il seeding playoff con tiebreaker NFL completi, poi lancia
la Monte Carlo per le probabilita' di vittoria Super Bowl.

Nota: le probabilita' finali qui sono quelle "what actually happened" (usa i
risultati reali della regular season 2024, poi simula SOLO i playoff in
avanti) -- e' il caso d'uso "chi vince il Super Bowl dato quello che sappiamo
oggi", diverso dal Monte Carlo "quanto e' stato fortuna" di tutta la stagione
usato nello script originale (use_actual_results_if_available=False).
"""

import pandas as pd
import numpy as np

from data_ingestion import load_games, ABBR_TO_FULLNAME
from stacking_model import train_xgb_margin_model, oof_xgb_predictions, FEATURE_COLS, NU, SIGMA_GAME
import nfl_model as nm
from scipy.stats import t as student_t

nm.set_seed(42)

FULLNAME_TO_ABBR = {v: k for k, v in ABBR_TO_FULLNAME.items()}


def main():
    teams_df = pd.read_csv("teams.csv")
    teams_df.columns = teams_df.columns.str.strip().str.lower()
    teams_df["name"] = teams_df["name"].str.strip()

    # Le nostre Team hanno come "name" la sigla nflverse (coerente col resto
    # della pipeline), ma conference/division arrivano da teams.csv (nomi
    # completi) -- serve la mappa nome-completo -> sigla.
    teams = {}
    for _, row in teams_df.iterrows():
        abbr = FULLNAME_TO_ABBR.get(row["name"])
        if abbr is None:
            print(f"ATTENZIONE: '{row['name']}' non trovato nella mappa sigle, salto.")
            continue
        teams[abbr] = nm.Team(name=abbr, conference=row["conference"], division=row["division"])

    print(f"Squadre caricate: {len(teams)}/32")

    games = load_games([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    if "game_type" in games.columns:
        games = games[games["game_type"] == "REG"].copy()
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games = games.rename(columns={"home_score": "home_points", "away_score": "away_points"})
    games = games.sort_values(["season", "week"]).reset_index(drop=True)

    # --- XGBoost allenato SOLO su 2016-2023 (2024 e' il "presente" da proiettare) ---
    feat = pd.read_csv("data/team_week_features.csv")
    train_feat = feat[feat["season"] < 2024].dropna(subset=["margin"]).copy()
    model = train_xgb_margin_model(train_feat)

    feat_valid = feat.dropna(subset=FEATURE_COLS, how="all").copy()
    feat_valid["xgb_margin_pred"] = model.predict(feat_valid[FEATURE_COLS])
    xgb_lookup = {(row.game_id, row.team): row.xgb_margin_pred for row in feat_valid.itertuples()}

    def xgb_predict_fn(game_id, home_team):
        return xgb_lookup.get((game_id, home_team))

    # --- PASSO 1: run preliminare per raccogliere i veri margini Kalman
    # per-partita (l'update Kalman usa sempre il margine reale, quindi questi
    # valori sono corretti indipendentemente da alpha) ---
    _, _, _, kalman_margins = nm.evaluate_multiseason_stacked(
        games, teams, xgb_predict_fn=xgb_predict_fn, alpha=0.5
    )
    train_feat_km = train_feat.copy()
    train_feat_km["kalman_margin"] = train_feat_km["game_id"].map(kalman_margins)
    train_feat_km = train_feat_km.dropna(subset=["kalman_margin"])
    oof_pred_train = oof_xgb_predictions(train_feat_km)
    y = train_feat_km["margin"].values
    diff = train_feat_km["kalman_margin"].values - oof_pred_train
    alpha = float(np.clip(np.sum((y - oof_pred_train) * diff) / np.sum(diff ** 2), 0.0, 1.0))
    print(f"alpha (peso Kalman, stimato su veri margini Kalman): {alpha:.2f}")

    # --- PASSO 2: reset team e run finale col vero alpha, per arrivare ai
    # rating (mu, sigma) di fine regular-season 2024 e ai record 2024 ---
    teams = {abbr: nm.Team(name=abbr, conference=t.conference, division=t.division) for abbr, t in teams.items()}
    _, _, records_by_season, _ = nm.evaluate_multiseason_stacked(
        games, teams, xgb_predict_fn=xgb_predict_fn, alpha=alpha
    )
    records_2024 = records_by_season[2024]

    games_2024 = games[games["season"] == 2024]

    print("\nRATING DI FINE REGULAR SEASON 2024 (top 10 per mu)")
    ranked = sorted(teams.values(), key=lambda t: -t.mu)
    for t in ranked[:10]:
        print(f"  {ABBR_TO_FULLNAME[t.name]:<28} mu={t.mu:+.2f}  sigma={t.sigma:.2f}")

    for conf in ["AFC", "NFC"]:
        conf_teams = [t for t in teams.values() if t.conference == conf]
        seeds = nm.seed_conference(conf_teams, records_2024, games_2024)
        print(f"\n{conf} PLAYOFF SEEDS 2024")
        for i, t in enumerate(seeds, 1):
            r = records_2024[t.name]
            record_str = f"{r.wins}-{r.losses}" + (f"-{r.ties}" if r.ties else "")
            print(f"  {i}. {ABBR_TO_FULLNAME[t.name]:<28} {record_str}  PD: {r.point_diff:+.0f}")

    # --- Monte Carlo: playoff a partire dai rating reali di fine regular
    # season 2024 (nessun ricampionamento della regular season: quella e'
    # gia' accaduta) ---
    print("\nMonte Carlo sui playoff 2024 (5000 repliche)...")
    n_sims = 5000
    afc_seeds = nm.seed_conference([t for t in teams.values() if t.conference == "AFC"], records_2024, games_2024)
    nfc_seeds = nm.seed_conference([t for t in teams.values() if t.conference == "NFC"], records_2024, games_2024)
    sb_wins = {t: 0 for t in teams}
    for _ in range(n_sims):
        afc_champ = nm.conference_playoffs(afc_seeds, xgb_margin_fn=None, alpha=alpha)
        nfc_champ = nm.conference_playoffs(nfc_seeds, xgb_margin_fn=None, alpha=alpha)
        sb_winner = nm.simulate_super_bowl(afc_champ, nfc_champ, xgb_margin_fn=None, alpha=alpha)
        sb_wins[sb_winner.name] += 1

    for k in sb_wins:
        sb_wins[k] /= n_sims

    print("\nSUPER BOWL WIN PROBABILITIES (top 10)")
    for abbr, prob in sorted(sb_wins.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ABBR_TO_FULLNAME[abbr]:<28} {prob:.3f}")

    out = pd.DataFrame(
        [(ABBR_TO_FULLNAME[k], v) for k, v in sb_wins.items()], columns=["Team", "SuperBowl_Prob"]
    ).sort_values("SuperBowl_Prob", ascending=False)
    out.to_csv("data/superbowl_probs_2024.csv", index=False)
    print("\nSalvato in data/superbowl_probs_2024.csv")


if __name__ == "__main__":
    main()