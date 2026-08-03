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

# =========================
# REPRODUCIBILITY
# =========================
# Tutta la casualita' del modulo passa da questo generatore, cosi' un singolo
# seed rende riproducibili sia il backtest storico sia le simulazioni Monte Carlo.
_rng = np.random.default_rng()


def set_seed(seed):
    """Fissa il seed globale usato da tutte le funzioni stocastiche del modulo."""
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
        """Passo di predizione del filtro di Kalman: la stima puntuale (mu) non
        deve essere perturbata casualmente (il rumore di processo ha media zero),
        cresce solo l'incertezza (sigma). Questo rende il backtest storico
        deterministico a parita' di dati, invece che rumoroso ad ogni run.
        """
        self.sigma = np.sqrt(self.sigma ** 2 + Q_DRIFT ** 2)

    def regress(self):
        self.mu *= (1 - REGRESSION_LAMBDA)

    def maybe_changepoint(self):
        """Shock stocastico (es. infortunio QB, cambio staff) che simula un
        cambiamento di regime non spiegabile dai soli risultati. Va usato SOLO
        nelle simulazioni Monte Carlo in avanti, mai nel fitting storico, perche'
        non e' un'informazione osservata ma un'ipotesi sul futuro."""
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
        """Percentuale vittorie alla NFL: i pareggi valgono 0.5."""
        total = self.wins + self.losses + self.ties
        return (self.wins + 0.5 * self.ties) / total if total > 0 else 0.0


# =========================
# CORE GAME MATH
# =========================
def simulate_margin(A, B, home=True):
    mean = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    return t.rvs(df=NU, loc=mean, scale=SIGMA_GAME, random_state=_rng)


def win_probability(A, B, home=True):
    mean = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    return 1 - t.cdf(0, df=NU, loc=mean, scale=SIGMA_GAME)


def update_ratings(A, B, margin, home=True):
    expected = A.mu - B.mu + (HOME_FIELD_ADV if home else 0.0)
    error = margin - expected
    var_A, var_B = A.sigma ** 2, B.sigma ** 2
    K_A = var_A / (var_A + SIGMA_GAME ** 2)
    K_B = var_B / (var_B + SIGMA_GAME ** 2)
    A.mu += K_A * error
    B.mu -= K_B * error
    A.sigma = np.sqrt((1 - K_A) * var_A)
    B.sigma = np.sqrt((1 - K_B) * var_B)
    A.history.append(A.mu)
    B.history.append(B.mu)


def center_league(teams):
    mean_mu = np.mean([tm.mu for tm in teams.values()])
    for tm in teams.values():
        tm.mu -= mean_mu


# =========================
# TIEBREAKERS
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
    """Average strength (rating) of teams defeated by this team."""
    r = records[team_name]
    return np.mean(r.defeated_team_strengths) if r.defeated_team_strengths else 0.0


def strength_of_schedule(team_name, records):
    """Average wins of teams this team has played."""
    r = records[team_name]
    return np.mean([records[opp].wins for opp in r.opponent_team_names]) if r.opponent_team_names else 0.0


def resolve_tie(tied_teams, records, season_df):
    """Recursively resolve multi-team ties according to NFL rules."""
    if len(tied_teams) == 1:
        return tied_teams

    # 1) Head-to-head among tied teams
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

    # 2) Division record (if in same division)
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

    # 3) Conference record
    conf_pct = {t: records[t].conf_wins / max(1, records[t].conf_wins + records[t].conf_losses) for t in tied_teams}
    max_conf = max(conf_pct.values())
    top_conf = [t for t in tied_teams if conf_pct[t] == max_conf]
    if len(top_conf) < len(tied_teams):
        return resolve_tie(top_conf, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_conf], records, season_df
        )

    # 4) Strength of victory
    sov = {t: strength_of_victory(t, records) for t in tied_teams}
    max_sov = max(sov.values())
    top_sov = [t for t in tied_teams if sov[t] == max_sov]
    if len(top_sov) < len(tied_teams):
        return resolve_tie(top_sov, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_sov], records, season_df
        )

    # 5) Strength of schedule
    sos = {t: strength_of_schedule(t, records) for t in tied_teams}
    max_sos = max(sos.values())
    top_sos = [t for t in tied_teams if sos[t] == max_sos]
    if len(top_sos) < len(tied_teams):
        return resolve_tie(top_sos, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_sos], records, season_df
        )

    # 6) Point differential
    pd_ = {t: records[t].point_diff for t in tied_teams}
    max_pd = max(pd_.values())
    top_pd = [t for t in tied_teams if pd_[t] == max_pd]
    if len(top_pd) < len(tied_teams):
        return resolve_tie(top_pd, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_pd], records, season_df
        )

    # 7) Points scored
    pts = {t: records[t].points_for for t in tied_teams}
    max_pts = max(pts.values())
    top_pts = [t for t in tied_teams if pts[t] == max_pts]
    if len(top_pts) < len(tied_teams):
        return resolve_tie(top_pts, records, season_df) + resolve_tie(
            [t for t in tied_teams if t not in top_pts], records, season_df
        )

    # if still tied, sort alphabetically
    return sorted(tied_teams)


def seed_conference(teams, records, season_df):
    """Generate correct playoff seeding including multi-team ties."""
    divisions = defaultdict(list)
    for t in teams:
        divisions[t.division].append(t.name)

    # pick division winners
    div_winners = []
    for div, tnames in divisions.items():
        winner = resolve_tie(tnames, records, season_df)[0]
        div_winners.append(winner)

    # order division winners by record for top seeds
    div_winners = resolve_tie(div_winners, records, season_df)

    # pick wild cards
    wild_cards = [t.name for t in teams if t.name not in div_winners]
    wild_cards = resolve_tie(wild_cards, records, season_df)[:3]

    # return Team objects in seeding order
    name_to_team = {t.name: t for t in teams}
    return [name_to_team[n] for n in div_winners + wild_cards]


# =========================
# PLAYOFF SIMULATION
# =========================
def playoff_game(A, B, neutral_site=False):
    """neutral_site=True per il Super Bowl: nessun vantaggio campo per nessuna
    delle due squadre."""
    margin = simulate_margin(A, B, home=not neutral_site)
    return A if margin > 0 else B


def conference_playoffs(seeds):
    alive = [
        seeds[0],
        playoff_game(seeds[1], seeds[6]),
        playoff_game(seeds[2], seeds[5]),
        playoff_game(seeds[3], seeds[4])
    ]
    alive = [
        playoff_game(alive[0], alive[3]),
        playoff_game(alive[1], alive[2])
    ]
    return playoff_game(alive[0], alive[1])


def simulate_super_bowl(afc_champ, nfc_champ):
    """Il Super Bowl si gioca in sede neutra: niente home field advantage."""
    return playoff_game(afc_champ, nfc_champ, neutral_site=True)


# =========================
# RECORD BOOKKEEPING (condiviso da evaluate/simulate)
# =========================
def _apply_game_to_records(records, home_name, away_name, home_pts, away_pts,
                            home_division, away_division, home_conf, away_conf):
    rA, rB = records[home_name], records[away_name]
    rA.points_for += home_pts
    rA.points_against += away_pts
    rB.points_for += away_pts
    rB.points_against += home_pts

    rA.opponent_team_names.append(away_name)
    rB.opponent_team_names.append(home_name)

    same_div = home_division == away_division
    same_conf = home_conf == away_conf

    if home_pts > away_pts:
        rA.wins += 1
        rB.losses += 1
        if same_div:
            rA.div_wins += 1
            rB.div_losses += 1
        if same_conf:
            rA.conf_wins += 1
            rB.conf_losses += 1
        return 1, rA, rB
    elif away_pts > home_pts:
        rB.wins += 1
        rA.losses += 1
        if same_div:
            rB.div_wins += 1
            rA.div_losses += 1
        if same_conf:
            rB.conf_wins += 1
            rA.conf_losses += 1
        return 0, rA, rB
    else:
        # Pareggio: rarissimo ma regolamentare nella NFL.
        rA.ties += 1
        rB.ties += 1
        if same_div:
            rA.div_ties += 1
            rB.div_ties += 1
        if same_conf:
            rA.conf_ties += 1
            rB.conf_ties += 1
        return 0.5, rA, rB


# =========================
# WALK-FORWARD: VALUTAZIONE STORICA (deterministica)
# =========================
def evaluate_historical_season(season_df, teams):
    """Fitting/valutazione su risultati REALI, gia' accaduti.

    Deterministico a parita' di dati (l'unica fonte di casualita' del modulo,
    maybe_changepoint, viene volutamente esclusa qui: non deve inquinare il
    fit dei rating con shock che non sono osservazioni reali). Usalo per
    calcolare log loss / Brier e per ottenere i record di fine stagione reali.
    """
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
# SIMULAZIONE MONTE CARLO IN AVANTI (stocastica)
# =========================
def simulate_season(season_df, teams, resample_initial_ratings=True,
                     use_actual_results_if_available=False):
    """Una singola replica Monte Carlo della stagione.

    A differenza di evaluate_historical_season, qui i risultati delle partite
    NON vengono presi dai punteggi reali (a meno che use_actual_results_if_available=True
    e il punteggio sia effettivamente disponibile: utile per proiezioni "resto
    stagione" a meta' campionato). Ogni partita viene invece campionata dalla
    distribuzione predittiva del modello (simulate_margin), cosi' la varianza
    tra le repliche riflette davvero l'incertezza sul risultato delle partite,
    non solo rumore nei rating.

    Le partite sono considerate "gia' giocate" (risultato noto) quando
    home_points/away_points sono valorizzati e non NaN.
    """
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
            margin = simulate_margin(A, B, home=True)
            # Ricostruiamo un punteggio plausibile solo per tenere coerente il
            # bookkeeping punti-fatti/subiti (non usato per altro).
            home_pts = max(0.0, 21 + margin / 2)
            away_pts = max(0.0, 21 - margin / 2)

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


# =========================
# MONTE CARLO SEASON (usa simulate_season, non piu' i risultati reali)
# =========================
def monte_carlo_season(teams, schedule_df, n_simulations=1000,
                        output_file="superbowl_probs.csv",
                        use_actual_results_if_available=False,
                        seed=None):
    """Stima le probabilita' di vittoria del Super Bowl via Monte Carlo.

    use_actual_results_if_available=False (default): replica stocastica
        dell'intera stagione, ignorando i punteggi reali -> utile per
        quantificare quanta "fortuna" c'e' stata nei risultati effettivi.
    use_actual_results_if_available=True: usa i risultati reali per le
        partite gia' giocate e simula solo quelle future -> proiezione
        "resto della stagione" a campionato in corso (richiede NaN nelle
        colonne home_points/away_points per le partite non ancora giocate).
    """
    if seed is not None:
        set_seed(seed)

    sb_wins = {t: 0 for t in teams}

    for _ in range(n_simulations):
        sim_teams, records = simulate_season(
            schedule_df, teams,
            resample_initial_ratings=True,
            use_actual_results_if_available=use_actual_results_if_available,
        )

        afc = [tm for tm in sim_teams.values() if tm.conference.upper() == "AFC"]
        nfc = [tm for tm in sim_teams.values() if tm.conference.upper() == "NFC"]

        afc_seeds = seed_conference(afc, records, schedule_df)
        nfc_seeds = seed_conference(nfc, records, schedule_df)

        afc_champ = conference_playoffs(afc_seeds)
        nfc_champ = conference_playoffs(nfc_seeds)
        sb_winner = simulate_super_bowl(afc_champ, nfc_champ)
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
# SCORING METRICS
# =========================
def log_loss(preds):
    return -np.mean([y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS) for p, y in preds])


def brier_score(preds):
    return np.mean([(p - y) ** 2 for p, y in preds])
