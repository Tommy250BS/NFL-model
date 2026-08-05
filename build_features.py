"""
Assembla la feature table finale, settimana per settimana, in modo
walk-forward: per predire la settimana W usiamo solo dati di partite gia'
giocate (< W). Questo evita leakage e rende la tabella direttamente
utilizzabile come training set per XGBoost con split temporale.

Output: una riga per squadra-partita, con:
- off_epa_adj / def_epa_adj   (opponent-adjusted, aggiornato a fine settimana W-1)
- cpoe_reg                    (regolarizzato, aggiornato a fine settimana W-1)
- qb_elo                      (rating Elo QB-adjusted pre-partita)
- rest_diff, is_dome, temp, wind   (da games.csv)
- spread_line                 (linea Vegas, utile come benchmark/feature di calibrazione)
- target: home_win, margin    (risultato reale, per training/backtest)
"""

import numpy as np
import pandas as pd
from data_ingestion import load_pbp, load_games, build_team_game_table, load_injury_burden
from feature_engineering import (
    build_opponent_table,
    opponent_adjusted_epa,
    regularize_cpoe,
    QBAdjustedElo,
)
from drive_markov import build_walk_forward_drive_features


def _trailing_situational_features(tg: pd.DataFrame, games: pd.DataFrame, window=4) -> pd.DataFrame:
    """Media mobile (ultime `window` partite gia' giocate) di EPA su 3rd down,
    EPA in red zone e turnover differential, per squadra. Shiftata di una
    partita: il valore associato alla settimana W riflette solo partite < W
    (nessun leakage)."""
    tg = tg.merge(games[["game_id", "season", "week"]].drop_duplicates(), on=["game_id", "week"], how="left")
    tg = tg.sort_values(["team", "season", "week"])

    def roll(s):
        return s.shift(1).rolling(window, min_periods=1).mean()

    tg = tg.sort_values(["team", "season", "week"])
    tg["off_epa_3rd_down_roll"] = tg.groupby("team")["off_epa_3rd_down"].transform(roll)
    tg["off_epa_redzone_roll"] = tg.groupby("team")["off_epa_redzone"].transform(roll)
    tg["turnover_diff"] = tg["turnovers_forced"] - tg["turnovers_lost"]
    tg["turnover_diff_roll"] = tg.groupby("team")["turnover_diff"].transform(roll)

    # Pace: rolling trailing di plays/partita e secondi/play, stessa logica
    # walk-forward (shift(1) prima del rolling: la settimana W vede solo
    # partite < W). Serve come feature di stile di gioco, non catturata
    # dall'EPA medio (una squadra hurry-up e una ball-control possono avere
    # lo stesso EPA/play ma un numero di possessi molto diverso).
    tg["plays_roll"] = tg.groupby("team")["off_plays"].transform(roll)
    tg["pace_roll"] = tg.groupby("team")["seconds_per_play"].transform(roll)

    return tg[[
        "game_id", "week", "team", "off_epa_3rd_down_roll", "off_epa_redzone_roll",
        "turnover_diff_roll", "plays_roll", "pace_roll",
    ]]



def build_walk_forward_features(years, ridge_lambda=25.0, min_week_for_epa=3,
                                 qb_weight=None, k_team=None, k_qb=None,
                                 include_drive_features=False, drive_prior_strength=50.0, drive_n_value_iters=20,
                                 drive_ridge_lambda=25.0, drive_ridge_weight=0.5, drive_min_drives_for_ridge=20,
                                 _pbp=None, _games=None):
    """qb_weight/k_team/k_qb: passati a QBAdjustedElo, default None = usa i
    default di classe. Esposti per grid search (walk_forward_backtest.py).
    _pbp/_games: permettono di riusare pbp/games gia' caricati fra chiamate
    ripetute della grid search invece di ri-scaricare/ri-parsare ogni volta.

    include_drive_features: se True, aggiunge drive_epd_off/drive_epd_def
    (expected points per drive, walk-forward, vedi drive_markov.py) alla
    tabella finale. DEFAULT FALSE perche' e' l'operazione piu' costosa della
    pipeline (value iteration per squadra per settimana, atteso qualche
    minuto su 9 stagioni) -- attivalo solo quando vuoi effettivamente
    valutare se il modello Markov drive-by-drive aiuta XGBoost, non ad ogni
    run di routine.

    drive_ridge_lambda/drive_ridge_weight/drive_min_drives_for_ridge: fix 3
    in drive_markov.py (opponent-adjustment via ridge invece del solo blend
    lineare per-stato) -- vedi build_walk_forward_drive_features per il
    dettaglio. drive_ridge_weight=0.0 disattiva il rating ridge (torna al
    comportamento solo-Markov)."""
    pbp = _pbp if _pbp is not None else load_pbp(years)
    games = _games if _games is not None else load_games(years)
    tg = build_team_game_table(pbp)

    games = games.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    elo = QBAdjustedElo(teams, k_team=k_team, k_qb=k_qb, qb_weight=qb_weight)

    league_epa_mean_running = tg["off_epa_play"].mean()  # fallback iniziale
    situational = _trailing_situational_features(tg, games)  # precalcolata una volta, walk-forward al suo interno
    injuries = load_injury_burden(years)  # dato pubblico pre-partita, nessun leakage: uso diretto della settimana corrente
    injury_lookup = {
        (row.season, row.week, row.team): (row.injury_burden, row.qb_out)
        for row in injuries.itertuples()
    }

    rows = []
    for (season, week), week_games in games.groupby(["season", "week"]):
        # --- opponent-adjusted EPA usando SOLO partite precedenti (di questa
        # e delle stagioni precedenti caricate) ---
        past_tg = tg[(tg["week"] < week) | (tg["game_id"].str.startswith(f"{season}_") == False)]
        past_tg = past_tg[
            past_tg["game_id"].isin(
                games[(games["season"] < season) | ((games["season"] == season) & (games["week"] < week))]["game_id"]
            )
        ]
        if len(past_tg) >= 20:  # servono abbastanza dati per una ridge sensata
            past_opp = build_opponent_table(past_tg)
            off_r, def_r = opponent_adjusted_epa(past_opp, ridge_lambda=ridge_lambda)
            cpoe_series = regularize_cpoe(past_tg)
            cpoe_by_team = past_tg.assign(cpoe_reg=cpoe_series).groupby("team")["cpoe_reg"].mean()
        else:
            off_r = {t: 0.0 for t in teams}
            def_r = {t: 0.0 for t in teams}
            cpoe_by_team = pd.Series(dtype=float)

        for _, g in week_games.iterrows():
            home, away = g["home_team"], g["away_team"]
            home_qb, away_qb = g["home_qb_id"], g["away_qb_id"]

            # rating Elo pre-partita (prima dell'update)
            elo_home = elo.effective_rating(home, home_qb)
            elo_away = elo.effective_rating(away, away_qb)
            win_prob_home = elo.expected_win_prob(home, home_qb, away, away_qb)

            injury_burden_home, qb_out_home = injury_lookup.get((season, week, home), (0.0, 0))
            injury_burden_away, qb_out_away = injury_lookup.get((season, week, away), (0.0, 0))

            rows.append(dict(
                game_id=g["game_id"], season=season, week=week,
                team=home, opponent=away, is_home=1,
                off_epa_adj=off_r.get(home, 0.0), def_epa_adj=def_r.get(home, 0.0),
                opp_off_epa_adj=off_r.get(away, 0.0), opp_def_epa_adj=def_r.get(away, 0.0),
                cpoe_reg=cpoe_by_team.get(home, np.nan),
                qb_elo=elo_home, opp_qb_elo=elo_away, elo_win_prob=win_prob_home,
                rest_diff=g["rest_diff"], is_dome=g["is_dome"],
                temp=g["temp"], wind=g["wind"],
                # wind_effective: azzera il vento per le squadre in cupola
                # (indoor/closed roof). Il vento raw resta comunque disponibile
                # come feature separata, ma senza questa interazione XGBoost
                # deve "scoprire" da solo la regola is_dome=1 -> wind irrilevante,
                # cosa che con profondita' massima 3 e pochi dati per cella
                # (partite in cupola con vento riportato = rumore residuo del
                # dato meteo stadio, non vento reale) puo' non riuscire a imparare bene.
                wind_effective=0.0 if pd.notna(g["is_dome"]) and g["is_dome"] == 1 else g["wind"],
                travel_distance=g["travel_distance"], tz_diff=g["tz_diff"],
                spread_line=g["spread_line"],
                injury_burden=injury_burden_home, opp_injury_burden=injury_burden_away,
                qb_out=qb_out_home, opp_qb_out=qb_out_away,
                home_win=int(g["home_score"] > g["away_score"]) if pd.notna(g["home_score"]) else np.nan,
                margin=(g["home_score"] - g["away_score"]) if pd.notna(g["home_score"]) else np.nan,
            ))

            # --- update Elo QB-adjusted DOPO aver registrato le feature pre-game ---
            if pd.notna(g["home_score"]) and pd.notna(g["away_score"]):
                home_won = 1.0 if g["home_score"] > g["away_score"] else (0.0 if g["home_score"] < g["away_score"] else 0.5)
                home_epa_rel = tg.loc[(tg["game_id"] == g["game_id"]) & (tg["team"] == home), "off_epa_play"]
                away_epa_rel = tg.loc[(tg["game_id"] == g["game_id"]) & (tg["team"] == away), "off_epa_play"]
                home_epa_rel = (home_epa_rel.iloc[0] - league_epa_mean_running) if len(home_epa_rel) else 0.0
                away_epa_rel = (away_epa_rel.iloc[0] - league_epa_mean_running) if len(away_epa_rel) else 0.0
                real_margin = g["home_score"] - g["away_score"]
                elo.update(home, home_qb, away, away_qb, home_won, home_epa_rel, away_epa_rel, margin=real_margin)

    result = pd.DataFrame(rows)
    result = result.merge(situational, on=["game_id", "week", "team"], how="left")

    if include_drive_features:
        drive_feat = build_walk_forward_drive_features(
            pbp, games, prior_strength=drive_prior_strength, n_value_iters=drive_n_value_iters,
            drive_ridge_lambda=drive_ridge_lambda, drive_ridge_weight=drive_ridge_weight,
            min_drives_for_ridge=drive_min_drives_for_ridge,
        )
        result = result.merge(drive_feat, on=["season", "week", "team"], how="left")
    else:
        # Colonne presenti comunque (NaN) anche quando include_drive_features=False,
        # cosi' stacking_model.FEATURE_COLS puo' includerle sempre senza KeyError --
        # XGBoost gestisce i NaN nativamente (missing=np.nan in train_xgb_margin_model).
        result["drive_epd_off"] = np.nan
        result["drive_epd_def"] = np.nan
        # colonne diagnostiche del fix 3 (markov puro / ridge puro): NaN per
        # coerenza di schema, anche se non usate da FEATURE_COLS.
        result["drive_epd_off_markov"] = np.nan
        result["drive_epd_def_markov"] = np.nan
        result["drive_epd_off_ridge"] = np.nan
        result["drive_epd_def_ridge"] = np.nan

    return result


if __name__ == "__main__":
    feat = build_walk_forward_features([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    print(feat.shape)
    print(feat.head(10).to_string(index=False))
    feat.to_csv("data/team_week_features.csv", index=False)
    print("\nSalvato in data/team_week_features.csv")

    # sanity check veloce: quanto e' informativo elo_win_prob da solo?
    valid = feat.dropna(subset=["home_win", "elo_win_prob"])
    p = valid["elo_win_prob"].clip(1e-6, 1 - 1e-6)
    y = valid["home_win"]
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    print(f"\nSolo Elo QB-adjusted (nessun XGBoost ancora): log loss={log_loss:.4f}  brier={brier:.4f}")
    print("(riferimento: un modello che indovina sempre 50% ha log loss=0.693, brier=0.25;")
    print(" le linee Vegas di solito si attestano su log loss ~0.55-0.58 in stagione regolare)")
