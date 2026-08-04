import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import t
from collections import defaultdict
import os

# =========================
# GLOBAL PARAMETERS
# =========================
HOME_FIELD_ADV = 2.2
SIGMA_GAME = 12.5
NU = 3
SIGMA_INIT = 15.0
Q_DRIFT = 0.6
REGRESSION_LAMBDA = 0.09
CHANGEPOINT_PROB = 0.05
CHANGEPOINT_DROP = 5.0
EPS = 1e-15

# --- FIX 1: floor su sigma per run multi-stagione -----------------------
# Senza floor, sigma decresce monotonicamente ad ogni update (tranne il
# drift Q_DRIFT) e dopo diverse settimane K->0: il filtro smette di
# aggiornarsi in modo significativo. Su una singola stagione non si nota,
# ma su multi-stagione (necessario per allenare XGBoost con abbastanza dati)
# il problema si aggrava. SIGMA_FLOOR impedisce a sigma di scendere sotto
# un livello minimo di incertezza residua.
SIGMA_FLOOR = 6.0

# --- FIX 2: regressione extra a inizio stagione --------------------------
# Il roster turnover (free agency, draft, infortuni offseason) e' molto piu'
# forte del semplice drift settimanale. SEASON_REGRESSION_EXTRA si applica
# una tantum a ogni cambio di stagione, oltre alla normale regress() settimanale.
SEASON_REGRESSION_EXTRA = 0.30

# =========================
# REPRODUCIBILITY
# =========================
_rng = np.random.default_rng()


def set_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)
    return _rng


# =========================
# TEAM AND RECORD CLASSES
# =========================
class Team:
    def __init__(self, name, conference, division):
        self.name = name
        self.conference = conference
        self.division = division
        self.mu = 0.0
        self.sigma = SIGMA_INIT
        self.history = []

    def evolve(self):
        self.sigma = np.sqrt(self.sigma ** 2 + Q_DRIFT ** 2)

    def regress(self):
        self.mu *= (1 - REGRESSION_LAMBDA)

    def season_reset(self):
        """Applicata una volta all'inizio di ogni nuova stagione: regressione
        extra verso 0 (roster turnover) e ripristino di un po' di incertezza
        (sigma), visto che il roster non e' piu' lo stesso della stagione
        precedente anche se il nome della squadra sì."""
        self.mu *= (1 - SEASON_REGRESSION_EXTRA)
        self.sigma = np.sqrt(self.sigma ** 2 + (SIGMA_INIT * 0.5) ** 2)

    def maybe_changepoint(self):
        if _rng.random() < CHANGEPOINT_PROB:
            shock = _rng.normal(-CHANGEPOINT_DROP, 2.0)
            self.mu += shock
            self.sigma += abs(shock) * 0.3


class Record:
    def __init__(self, division=None, conference=None):
        self.division = division
        self.conference = conference
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.div_wins = 0
        self.div_losses = 0
        self.div_ties = 0
        self.conf_wins = 0
        self.conf_losses = 0
        self.conf_ties = 0
        self.points_for = 0
        self.points_against = 0
        self.opponent_team_names = []
        self.defeated_team_strengths = []
        self.tied_teams = []

    @property
    def point_diff(self):
        return self.points_for - self.points_against

    @property
    def win_pct(self):
        total = self.wins + self.losses + self.ties
        return (self.wins + 0.5 * self.ties) / total if total > 0 else 0.0


# =========================
# CORE GAME MATH
# =========================
# --- FIX 3 (parziale/interim): sintesi di un punteggio plausibile dato solo
# il margine simulato -------------------------------------------------------
# La versione precedente usava home_pts = 21 + margin/2 (base fissa 21,
# split simmetrico): deterministico dato il margine, e produce punteggi
# come "24.5-17.5" che non esistono in NFL. Qui invece campioniamo un totale
# di punti realistico (media/deviazione stimate su NFL storica moderna, circa
# 44-45 punti/partita totali) indipendente dal margine, poi deriviamo
# home/away e arrotondiamo ai multipli di punteggio piu' comuni (TD=7, FG=3).
# Resta una euristica, non un vero modello di scoring: la sostituzione
# corretta e' un modello drive-by-drive (Markov chain su down/distanza/field
# position) che genera i punteggi da possessi simulati invece che da un
# margine gia' aggregato -- prossimo passo naturale di questa pipeline.
LEAGUE_TOTAL_POINTS_MEAN = 44.5
LEAGUE_TOTAL_POINTS_SD = 10.0
_SCORE_QUANTA = np.array([0, 3, 6, 7, 8, 9, 10, 13, 14, 16, 17, 20, 21, 23, 24, 27, 28, 30, 31, 34, 35, 38, 41])


def _round_to_plausible_score(x):
    return _SCORE_QUANTA[np.argmin(np.abs(_SCORE_QUANTA - x))]


def _synthesize_scores(margin):
    total = max(6.0, _rng.normal(LEAGUE_TOTAL_POINTS_MEAN, LEAGUE_TOTAL_POINTS_SD))
    home_raw = (total + margin) / 2
    away_raw = (total - margin) / 2
    home_pts = float(_round_to_plausible_score(max(0.0, home_raw)))
    away_pts = float(_round_to_plausible_score(max(0.0, away_raw)))
    return home_pts, away_pts


def simulate_margin(A, B, home=True):
    mean = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    return t.rvs(df=NU, loc=mean, scale=SIGMA_GAME, random_state=_rng)


def win_probability(A, B, home=True):
    mean = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    return 1 - t.cdf(0, df=NU, loc=mean, scale=SIGMA_GAME)


def blended_win_probability(kalman_margin, xgb_margin, alpha):
    """Combina il margine implicito dal rating Kalman con quello stimato da
    XGBoost (vedi stacking_model.py) nella stessa famiglia t-Student usata
    per la Monte Carlo. alpha = peso del Kalman (1-alpha = peso XGBoost)."""
    blended_margin = alpha * kalman_margin + (1 - alpha) * xgb_margin
    return 1 - t.cdf(0, df=NU, loc=blended_margin, scale=SIGMA_GAME), blended_margin


def update_ratings(A, B, margin, home=True):
    expected = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    error = margin - expected
    var_A, var_B = A.sigma ** 2, B.sigma ** 2
    K_A = var_A / (var_A + SIGMA_GAME ** 2)
    K_B = var_B / (var_B + SIGMA_GAME ** 2)
    A.mu += K_A * error
    B.mu -= K_B * error
    A.sigma = max(np.sqrt((1 - K_A) * var_A), SIGMA_FLOOR)
    B.sigma = max(np.sqrt((1 - K_B) * var_B), SIGMA_FLOOR)
    A.history.append(A.mu)
    B.history.append(B.mu)


def center_league(teams):
    mean_mu = np.mean([tm.mu for tm in teams.values()])
    for tm in teams.values():
        tm.mu -= mean_mu


# =========================
# TIEBREAKERS (invariati rispetto all'originale)
# =========================
def head_to_head_pct(team_name, tied_names, season_df):
    opponents_played = [t for t in tied_names if t != team_name]
    games = season_df[
        ((season_df.home_team == team_name) & (season_df.away_team.isin(opponents_played))) |
        ((season_df.away_team == team_name) & (season_df.home_team.isin(opponents_played)))
    ]
    if len(games) < 1:
        return None
    wins = sum(
        ((games.home_team == team_name) & (games.home_points > games.away_points)) |
        ((games.away_team == team_name) & (games.away_points > games.home_points))
    )
    return wins / len(games)


def strength_of_victory(team_name, records):
    r = records[team_name]
    return np.mean(r.defeated_team_strengths) if r.defeated_team_strengths else 0.0


def strength_of_schedule(team_name, records):
    r = records[team_name]
    return np.mean([records[opp].wins for opp in r.opponent_team_names]) if r.opponent_team_names else 0.0


def resolve_tie(tied_teams, records, season_df):
    """Risoluzione ricorsiva di tie multi-squadra secondo le regole NFL.

    FIX rispetto alla versione originale: il primo criterio deve SEMPRE
    essere la percentuale di vittorie (win_pct). Le regole NFL (head-to-head,
    division record, SOV, ecc.) si applicano SOLO per rompere una parita' fra
    squadre che hanno gia' la stessa win_pct -- non come criterio primario
    applicato a un intero gruppo di squadre con record diversi. La versione
    precedente chiamava head-to-head direttamente su tutte le squadre di una
    division indipendentemente dal loro record, il che poteva (ed e'
    successo, verificato sui dati 2024 reali) far vincere la division a una
    squadra con un record peggiore.
    """
    if len(tied_teams) == 1:
        return tied_teams

    # --- STEP 0 (fix): raggruppa per win_pct, ordina i gruppi, e applica i
    # tiebreaker NFL solo DENTRO ogni gruppo alla pari ---
    win_pct = {t: records[t].win_pct for t in tied_teams}
    distinct_pcts = sorted(set(win_pct.values()), reverse=True)
    if len(distinct_pcts) > 1:
        ordered = []
        for pct in distinct_pcts:
            group = [t for t in tied_teams if win_pct[t] == pct]
            ordered += resolve_tie(group, records, season_df)
        return ordered

    # Da qui in poi, tutte le squadre in tied_teams hanno la STESSA win_pct:
    # applichiamo la cascata di tiebreaker NFL vera e propria.
    h2h_records = {}
    for team in tied_teams:
        games = season_df[
            ((season_df['home_team'] == team) & (season_df['away_team'].isin(tied_teams))) |
            ((season_df['away_team'] == team) & (season_df['home_team'].isin(tied_teams)))
        ]
        wins = ((games['home_team'] == team) & (games['home_points'] > games['away_points'])).sum() + \
               ((games['away_team'] == team) & (games['away_points'] > games['home_points'])).sum()
        h2h_records[team] = wins

    max_h2h = max(h2h_records.values())
    top_h2h = [t for t in tied_teams if h2h_records[t] == max_h2h]
    if len(top_h2h) < len(tied_teams):
        return resolve_tie(top_h2h, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_h2h], records, season_df
        )

    divisions = defaultdict(list)
    for t in tied_teams:
        divisions[records[t].division].append(t)
    if len(divisions) == 1:
        div_pct = {t: records[t].div_wins / max(1, records[t].div_wins + records[t].div_losses) for t in tied_teams}
        max_div = max(div_pct.values())
        top_div = [t for t in tied_teams if div_pct[t] == max_div]
        if len(top_div) < len(tied_teams):
            return resolve_tie(top_div, records, season_df) + resolve_tie(
                [t for t in tied_teams if t not in top_div], records, season_df
            )

    conf_pct = {t: records[t].conf_wins / max(1, records[t].conf_wins + records[t].conf_losses) for t in tied_teams}
    max_conf = max(conf_pct.values())
    top_conf = [t for t in tied_teams if conf_pct[t] == max_conf]
    if len(top_conf) < len(tied_teams):
        return resolve_tie(top_conf, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_conf], records, season_df
        )

    sov = {t: strength_of_victory(t, records) for t in tied_teams}
    max_sov = max(sov.values())
    top_sov = [t for t in tied_teams if sov[t] == max_sov]
    if len(top_sov) < len(tied_teams):
        return resolve_tie(top_sov, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_sov], records, season_df
        )

    sos = {t: strength_of_schedule(t, records) for t in tied_teams}
    max_sos = max(sos.values())
    top_sos = [t for t in tied_teams if sos[t] == max_sos]
    if len(top_sos) < len(tied_teams):
        return resolve_tie(top_sos, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_sos], records, season_df
        )

    pd_ = {t: records[t].point_diff for t in tied_teams}
    max_pd = max(pd_.values())
    top_pd = [t for t in tied_teams if pd_[t] == max_pd]
    if len(top_pd) < len(tied_teams):
        return resolve_tie(top_pd, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_pd], records, season_df
        )

    pts = {t: records[t].points_for for t in tied_teams}
    max_pts = max(pts.values())
    top_pts = [t for t in tied_teams if pts[t] == max_pts]
    if len(top_pts) < len(tied_teams):
        return resolve_tie(top_pts, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_pts], records, season_df
        )

    return sorted(tied_teams)


def seed_conference(teams, records, season_df):
    """Seeding di conference completo (7 squadre: 4 division winner + 3 wild card)."""
    divisions = defaultdict(list)
    for t in teams:
        divisions[t.division].append(t.name)

    div_winners = []
    for div, tnames in divisions.items():
        winner = resolve_tie(tnames, records, season_df)[0]
        div_winners.append(winner)

    div_winners = resolve_tie(div_winners, records, season_df)

    wild_cards = [t.name for t in teams if t.name not in div_winners]
    wild_cards = resolve_tie(wild_cards, records, season_df)[:3]

    name_to_team = {t.name: t for t in teams}
    return [name_to_team[n] for n in div_winners + wild_cards]


# =========================
# PLAYOFF SIMULATION (con blend XGBoost opzionale)
# =========================
def playoff_game(A, B, neutral_site=False, xgb_margin_fn=None, alpha=0.5):
    """xgb_margin_fn(team_A, team_B) -> margine atteso XGBoost oppure None se
    non disponibile (es. incrocio playoff mai visto in training): in quel
    caso si ricade sul solo margine Kalman, che e' sempre disponibile."""
    home = not neutral_site
    kalman_mean = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    if xgb_margin_fn is not None:
        xgb_m = xgb_margin_fn(A.name, B.name)
        mean = alpha * kalman_mean + (1 - alpha) * xgb_m if xgb_m is not None else kalman_mean
    else:
        mean = kalman_mean
    margin = t.rvs(df=NU, loc=mean, scale=SIGMA_GAME, random_state=_rng)
    return A if margin > 0 else B


def conference_playoffs(seeds, xgb_margin_fn=None, alpha=0.5):
    alive = [
        seeds[0],
        playoff_game(seeds[1], seeds[6], xgb_margin_fn=xgb_margin_fn, alpha=alpha),
        playoff_game(seeds[2], seeds[5], xgb_margin_fn=xgb_margin_fn, alpha=alpha),
        playoff_game(seeds[3], seeds[4], xgb_margin_fn=xgb_margin_fn, alpha=alpha),
    ]
    alive = [
        playoff_game(alive[0], alive[3], xgb_margin_fn=xgb_margin_fn, alpha=alpha),
        playoff_game(alive[1], alive[2], xgb_margin_fn=xgb_margin_fn, alpha=alpha),
    ]
    return playoff_game(alive[0], alive[1], xgb_margin_fn=xgb_margin_fn, alpha=alpha)


def simulate_super_bowl(afc_champ, nfc_champ, xgb_margin_fn=None, alpha=0.5):
    return playoff_game(afc_champ, nfc_champ, neutral_site=True, xgb_margin_fn=xgb_margin_fn, alpha=alpha)


# =========================
# RECORD BOOKKEEPING COMPLETO (div/conf, opponent tracking per tiebreaker)
# =========================
def _apply_game_to_records(records, home_name, away_name, home_pts, away_pts,
                            home_division, away_division, home_conf, away_conf):
    rA, rB = records[home_name], records[away_name]
    rA.points_for += home_pts; rA.points_against += away_pts
    rB.points_for += away_pts; rB.points_against += home_pts
    rA.opponent_team_names.append(away_name)
    rB.opponent_team_names.append(home_name)

    same_div = home_division == away_division
    same_conf = home_conf == away_conf

    if home_pts > away_pts:
        rA.wins += 1; rB.losses += 1
        if same_div: rA.div_wins += 1; rB.div_losses += 1
        if same_conf: rA.conf_wins += 1; rB.conf_losses += 1
        return 1, rA, rB
    elif away_pts > home_pts:
        rB.wins += 1; rA.losses += 1
        if same_div: rB.div_wins += 1; rA.div_losses += 1
        if same_conf: rB.conf_wins += 1; rA.conf_losses += 1
        return 0, rA, rB
    else:
        rA.ties += 1; rB.ties += 1
        if same_div: rA.div_ties += 1; rB.div_ties += 1
        if same_conf: rA.conf_ties += 1; rB.conf_ties += 1
        return 0.5, rA, rB


# =========================
# WALK-FORWARD SINGOLA STAGIONE (compatibile con l'originale main_model.py)
# =========================
def evaluate_historical_season(season_df, teams):
    records = {name: Record(division=tm.division, conference=tm.conference) for name, tm in teams.items()}
    predictions = []
    current_week = None

    for g in season_df.itertuples(index=False):
        week = g.week
        if week != current_week:
            for tm in teams.values():
                tm.evolve()
                tm.regress()
            center_league(teams)
            current_week = week

        A = teams[g.home_team]
        B = teams[g.away_team]

        p = win_probability(A, B, home=True)
        result = int(g.home_points > g.away_points)
        predictions.append((p, result))

        margin = g.home_points - g.away_points
        update_ratings(A, B, margin, home=True)

        outcome, rA, rB = _apply_game_to_records(
            records, g.home_team, g.away_team, g.home_points, g.away_points,
            A.division, B.division, A.conference, B.conference
        )
        if outcome == 1:
            rA.defeated_team_strengths.append(B.mu)
        elif outcome == 0:
            rB.defeated_team_strengths.append(A.mu)

    return predictions, records


# =========================
# MONTE CARLO IN AVANTI (stocastico, con blend XGBoost opzionale)
# =========================
def simulate_season(season_df, teams, resample_initial_ratings=True,
                     use_actual_results_if_available=False,
                     xgb_margin_fn=None, alpha=0.5):
    sim_teams = deepcopy(teams)

    if resample_initial_ratings:
        for tm in sim_teams.values():
            tm.mu = _rng.normal(tm.mu, tm.sigma)

    records = {name: Record(division=tm.division, conference=tm.conference) for name, tm in sim_teams.items()}
    current_week = None
    has_scores = "home_points" in season_df.columns and "away_points" in season_df.columns

    for g in season_df.itertuples(index=False):
        week = g.week
        if week != current_week:
            for tm in sim_teams.values():
                tm.evolve()
                tm.regress()
                tm.maybe_changepoint()
            center_league(sim_teams)
            current_week = week

        A = sim_teams[g.home_team]
        B = sim_teams[g.away_team]

        actual_known = (
            use_actual_results_if_available and has_scores and
            pd.notna(getattr(g, "home_points", np.nan)) and pd.notna(getattr(g, "away_points", np.nan))
        )

        if actual_known:
            home_pts, away_pts = g.home_points, g.away_points
            margin = home_pts - away_pts
        else:
            kalman_mean = A.mu - B.mu + HOME_FIELD_ADV
            if xgb_margin_fn is not None:
                xgb_m = xgb_margin_fn(getattr(g, "game_id", None), g.home_team)
                mean = alpha * kalman_mean + (1 - alpha) * xgb_m if xgb_m is not None else kalman_mean
            else:
                mean = kalman_mean
            margin = t.rvs(df=NU, loc=mean, scale=SIGMA_GAME, random_state=_rng)
            home_pts, away_pts = _synthesize_scores(margin)

        update_ratings(A, B, margin, home=True)

        outcome, rA, rB = _apply_game_to_records(
            records, g.home_team, g.away_team, home_pts, away_pts,
            A.division, B.division, A.conference, B.conference
        )
        if outcome == 1:
            rA.defeated_team_strengths.append(B.mu)
        elif outcome == 0:
            rB.defeated_team_strengths.append(A.mu)

    return sim_teams, records


def monte_carlo_season(teams, schedule_df, n_simulations=1000,
                        output_file="superbowl_probs.csv",
                        use_actual_results_if_available=False,
                        seed=None, xgb_margin_fn=None, alpha=0.5):
    if seed is not None:
        set_seed(seed)

    sb_wins = {t: 0 for t in teams}

    for _ in range(n_simulations):
        sim_teams, records = simulate_season(
            schedule_df, teams,
            resample_initial_ratings=True,
            use_actual_results_if_available=use_actual_results_if_available,
            xgb_margin_fn=xgb_margin_fn, alpha=alpha,
        )

        afc = [tm for tm in sim_teams.values() if tm.conference.upper() == "AFC"]
        nfc = [tm for tm in sim_teams.values() if tm.conference.upper() == "NFC"]

        afc_seeds = seed_conference(afc, records, schedule_df)
        nfc_seeds = seed_conference(nfc, records, schedule_df)

        afc_champ = conference_playoffs(afc_seeds, xgb_margin_fn=None, alpha=alpha)
        nfc_champ = conference_playoffs(nfc_seeds, xgb_margin_fn=None, alpha=alpha)
        sb_winner = simulate_super_bowl(afc_champ, nfc_champ, xgb_margin_fn=None, alpha=alpha)
        sb_wins[sb_winner.name] += 1

    for k in sb_wins:
        sb_wins[k] /= n_simulations

    output_dir = os.path.dirname(output_file)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(list(sb_wins.items()), columns=["Team", "SuperBowl_Prob"])
    df = df.sort_values("SuperBowl_Prob", ascending=False)
    df.to_csv(output_file, index=False)
    print(f"Super Bowl probabilities saved to {os.path.abspath(output_file)}")

    return sb_wins


# =========================
# WALK-FORWARD MULTI-STAGIONE (con blend XGBoost opzionale) -- usato dal backtest
# =========================
def evaluate_multiseason_stacked(season_df, teams, xgb_predict_fn=None, alpha=0.5):
    """Estensione multi-stagione di evaluate_historical_season, con i due fix
    sopra e blend opzionale con un margine stimato da XGBoost (xgb_predict_fn:
    callable(game_id, home_team) -> margine atteso o None, allenato SOLO su
    stagioni precedenti a quelle qui valutate per evitare leakage).

    Ritorna: predictions_kalman, predictions_blend, records
    """
    records = {name: Record(division=tm.division, conference=tm.conference) for name, tm in teams.items()}
    predictions_kalman = []
    predictions_blend = []
    kalman_margins = {}  # game_id -> margine implicito Kalman (pre-update, no leakage)
    current_key = None
    current_season = None
    records_by_season = {}

    for g in season_df.itertuples(index=False):
        season, week = g.season, g.week
        key = (season, week)
        if key != current_key:
            if current_season is not None and season != current_season:
                for tm in teams.values():
                    tm.season_reset()
                # snapshot dei record della stagione appena conclusa, poi reset:
                # i rating (mu/sigma) restano persistenti da una stagione
                # all'altra, i RECORD (vittorie, division/conf record) no --
                # servono al seeding playoff della singola stagione, non
                # cumulati su piu' anni.
                records_by_season[current_season] = records
                records = {name: Record(division=tm.division, conference=tm.conference) for name, tm in teams.items()}
            for tm in teams.values():
                tm.evolve()
                tm.regress()
            center_league(teams)
            current_key = key
            current_season = season

        A = teams[g.home_team]
        B = teams[g.away_team]

        kalman_margin = A.mu - B.mu + HOME_FIELD_ADV
        kalman_margins[g.game_id] = kalman_margin
        p_kalman = win_probability(A, B, home=True)

        if xgb_predict_fn is not None:
            xgb_margin = xgb_predict_fn(g.game_id, g.home_team)
            if xgb_margin is not None:
                p_blend, _ = blended_win_probability(kalman_margin, xgb_margin, alpha)
            else:
                p_blend = p_kalman
        else:
            p_blend = p_kalman

        result = int(g.home_points > g.away_points)
        predictions_kalman.append((p_kalman, result))
        predictions_blend.append((p_blend, result))

        margin = g.home_points - g.away_points
        update_ratings(A, B, margin, home=True)

        outcome, rA, rB = _apply_game_to_records(
            records, g.home_team, g.away_team, g.home_points, g.away_points,
            A.division, B.division, A.conference, B.conference
        )
        if outcome == 1:
            rA.defeated_team_strengths.append(B.mu)
        elif outcome == 0:
            rB.defeated_team_strengths.append(A.mu)

    records_by_season[current_season] = records  # ultima stagione, mai snapshottata nel loop
    return predictions_kalman, predictions_blend, records_by_season, kalman_margins


def log_loss(preds):
    return -np.mean([y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS) for p, y in preds])


def brier_score(preds):
    return np.mean([(p - y) ** 2 for p, y in preds])