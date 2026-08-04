"""
Feature engineering avanzato, walk-forward (nessun leakage dal futuro).

Al posto di DVOA (proprietaria FTN, pesi non pubblici) e PRWR (dati NGS non
pubblici), usiamo proxy costruibili da dati pubblici nflverse:

- off_epa_adj / def_epa_adj : EPA/play aggiustato per la forza dell'avversario,
  stimato via ridge regression sulle partite fin qui giocate (equivalente in
  spirito a una "DVOA fatta in casa", senza pretendere di replicarne i pesi
  esatti).
- qb_adjusted_elo : rating Elo di squadra con overlay separato sul quarterback,
  cosi' un cambio titolare sposta immediatamente il rating effettivo.
- cpoe_reg : CPOE regolarizzato via shrinkage bayesiano verso la media di lega,
  pesato sul numero di attempt (un CPOE da 8 lanci non vale come uno da 35).

Tutte le funzioni sono pensate per essere chiamate settimana per settimana
dentro un ciclo walk-forward (vedi build_features.py), usando solo dati
STRETTAMENTE precedenti alla settimana da predire.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


# =========================
# 1) EPA OPPONENT-ADJUSTED (ridge regression)
# =========================
def opponent_adjusted_epa(team_game_df: pd.DataFrame, ridge_lambda: float = 25.0):
    """Stima off_rating[team] e def_rating[team] tali che

        epa_play_osservato(game, team) ~= off_rating[team] - def_rating[opponent] + league_avg

    risolvendo una ridge regression sparsa (un'osservazione per squadra-partita,
    una colonna per ogni rating offensivo/difensivo). ridge_lambda alto tira i
    rating verso 0 quando i dati sono pochi (early season) -- e' esattamente il
    tipo di regressione "opponent adjustment" usata da metriche pubbliche come
    quelle di rbsdm.com o Sumer Sports, qui reimplementata in casa.

    Ritorna due dict {team: rating}.
    """
    df = team_game_df.dropna(subset=["off_epa_play"]).copy()
    teams = sorted(set(df["team"]) | set(df["opponent"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_obs = len(df)

    # Ogni riga: colonna off_rating[team] = +1, colonna def_rating[opponent] = +1
    # (segno + perche' un opponent con def_rating alto = difesa forte = EPA
    # subita dall'attacco piu' bassa; il segno si sistema nella formula finale)
    rows, cols, vals = [], [], []
    for r, (_, row) in enumerate(df.iterrows()):
        rows += [r, r]
        cols += [idx[row["team"]], n_teams + idx[row["opponent"]]]
        vals += [1.0, -1.0]

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_obs, 2 * n_teams))
    y = df["off_epa_play"].values - df["off_epa_play"].mean()

    # Ridge via lsqr con damping = sqrt(lambda)
    sol = lsqr(X, y, damp=np.sqrt(ridge_lambda))[0]
    off_rating = {t: sol[idx[t]] for t in teams}
    def_rating = {t: sol[n_teams + idx[t]] for t in teams}
    return off_rating, def_rating


def build_opponent_table(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna 'opponent' necessaria a opponent_adjusted_epa,
    unendo ogni riga squadra-partita con la controparte nella stessa game_id."""
    df = team_game_df.copy()
    self_away = df.merge(df, on="game_id", suffixes=("", "_opp"))
    self_away = self_away[self_away["team"] != self_away["team_opp"]]
    self_away = self_away.rename(columns={"team_opp": "opponent"})
    return self_away[[c for c in self_away.columns if not c.endswith("_opp")]]


# =========================
# 2) CPOE REGOLARIZZATO (shrinkage bayesiano)
# =========================
def regularize_cpoe(team_game_df: pd.DataFrame, prior_strength: int = 25) -> pd.Series:
    """Shrink del CPOE di ogni squadra-partita verso la media di lega,
    con peso proporzionale al numero di attempt. prior_strength = numero di
    attempt "virtuali" assegnati alla media di lega (piu' alto = piu' shrink).

        cpoe_reg = (n_att * cpoe_raw + prior_strength * league_mean) / (n_att + prior_strength)
    """
    league_mean = np.average(
        team_game_df["cpoe_raw"].dropna(),
        weights=team_game_df.loc[team_game_df["cpoe_raw"].notna(), "cpoe_attempts"],
    )
    n = team_game_df["cpoe_attempts"].fillna(0)
    raw = team_game_df["cpoe_raw"].fillna(league_mean)
    return (n * raw + prior_strength * league_mean) / (n + prior_strength)


# =========================
# 3) ELO QB-ADJUSTED
# =========================
class QBAdjustedElo:
    """Elo di squadra + overlay separato sul quarterback (in stile 538 QB Elo).

    - team_elo: evolve lentamente, cattura la "forza strutturale" (linea,
      difesa, coaching) al netto del QB.
    - qb_value: rating separato per ogni QB, aggiornato in base alla
      performance (EPA/play della squadra quando quel QB gioca, relativo alla
      media di lega), che si somma al team_elo per dare il rating effettivo.

    Quando cambia il titolare, il rating effettivo si sposta subito al valore
    del nuovo QB, invece di dover "riapprendere" tutto tramite i risultati
    delle partite come farebbe un Elo semplice.
    """

    # Default storici, invariati per compatibilita' con le run esistenti.
    # Esposti anche come parametri del costruttore (k_team, k_qb, qb_weight)
    # cosi' da poter fare grid search via walk-forward (vedi
    # walk_forward_backtest.py) senza dover toccare stato di classe condiviso.
    K_TEAM = 20.0
    K_QB = 10.0
    QB_WEIGHT = 0.6  # quanto del rating effettivo viene dal QB overlay
    HOME_ADV_ELO = 48.0

    def __init__(self, teams, k_team=None, k_qb=None, qb_weight=None, home_adv_elo=None):
        self.team_elo = {t: 1500.0 for t in teams}
        self.qb_value = {}  # qb_id -> rating, inizializzato a 1500 alla prima apparizione
        self.K_TEAM = self.K_TEAM if k_team is None else k_team
        self.K_QB = self.K_QB if k_qb is None else k_qb
        self.QB_WEIGHT = self.QB_WEIGHT if qb_weight is None else qb_weight
        self.HOME_ADV_ELO = self.HOME_ADV_ELO if home_adv_elo is None else home_adv_elo

    def _get_qb(self, qb_id):
        if qb_id not in self.qb_value:
            self.qb_value[qb_id] = 1500.0
        return self.qb_value[qb_id]

    def effective_rating(self, team, qb_id):
        base = self.team_elo[team]
        qb = self._get_qb(qb_id)
        # media pesata tra rating di squadra e rating del QB
        return (1 - self.QB_WEIGHT) * base + self.QB_WEIGHT * qb

    def expected_win_prob(self, home_team, home_qb, away_team, away_qb):
        rh = self.effective_rating(home_team, home_qb) + self.HOME_ADV_ELO
        ra = self.effective_rating(away_team, away_qb)
        return 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))

    def update(self, home_team, home_qb, away_team, away_qb, home_won: float, off_epa_home_rel, off_epa_away_rel, margin=None):
        """home_won in {0, 0.5, 1}. off_epa_*_rel = EPA/play della squadra in
        quella partita meno la media di lega (segnale di performance del QB,
        indipendente dal risultato finale, per aggiornare qb_value anche in
        sconfitte "onorevoli"). margin = punti home - punti away, usato per il
        moltiplicatore MOV (margin-of-victory) in stile 538: una vittoria di
        30 punti sposta il rating piu' di una vittoria di 1 punto, e il
        moltiplicatore si smorza quando il favorito vince "come previsto" per
        non punire/premiare due volte lo stesso segnale."""
        p_home = self.expected_win_prob(home_team, home_qb, away_team, away_qb)

        if margin is not None:
            elo_diff_home = (self.effective_rating(home_team, home_qb) + self.HOME_ADV_ELO
                              - self.effective_rating(away_team, away_qb))
            mov_mult = np.log(abs(margin) + 1) * (2.2 / (abs(elo_diff_home) * 0.001 + 2.2))
        else:
            mov_mult = 1.0

        self.team_elo[home_team] += self.K_TEAM * mov_mult * (home_won - p_home)
        self.team_elo[away_team] += self.K_TEAM * mov_mult * ((1 - home_won) - (1 - p_home))

        # aggiornamento qb_value sulla performance EPA relativa, non sul risultato:
        # un QB puo' giocare bene e perdere per colpa della difesa, e va comunque
        # ricompensato (altrimenti l'overlay collassa nel semplice team elo)
        self.qb_value[home_qb] = self._get_qb(home_qb) + self.K_QB * off_epa_home_rel
        self.qb_value[away_qb] = self._get_qb(away_qb) + self.K_QB * off_epa_away_rel