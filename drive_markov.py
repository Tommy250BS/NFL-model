"""
Modello drive-by-drive via Markov chain su stati (down, distanza, field
position), walk-forward come il resto della pipeline.

STATO E TRANSIZIONI
--------------------
Uno stato e' la tripla (down, distance_bucket, fp_bucket):
- down: 1-4
- distance_bucket: 0 = 1-3 yard, 1 = 4-6, 2 = 7-9, 3 = 10+
- fp_bucket: yardline_100 // 10, clippato a [0,9]. yardline_100 e' la
  convenzione nflverse: 0 = end zone avversaria (goal line offensiva),
  100 = propria end zone. fp_bucket=0 quindi vuol dire "vicino a segnare".

Ogni play e' una transizione da uno stato pre-snap a: un altro stato
(prossimo down/distanza/field position nella stessa drive) oppure a un
ESITO assorbente che chiude la drive (touchdown, field goal, punt,
turnover, turnover su downs, safety). Il numero di stati e' contenuto
(4 down x 4 distanze x 10 field position = 160) apposta per restare
stimabile senza sparsita' estrema, stesso spirito di ridge_lambda alto
nell'opponent-adjusted EPA: con ~11 drive/partita a squadra i dati per
cella sono comunque pochi, quindi qui usiamo shrinkage bayesiano verso la
media di lega (stesso principio di regularize_cpoe) invece di una ridge.

LIMITAZIONI NOTE (da affrontare prima di usarlo in produzione)
----------------------------------------------------------------
1. Safety non e' distinguibile con le colonne pbp che teniamo (PBP_COLS in
   data_ingestion.py) -- andrebbe aggiunta una colonna dedicata o dedotta
   dal cambio di score_differential di -2 per l'attacco. Per ora il
   classificatore la ignora (non emette mai SAFETY).
2. Field position dopo punt/turnover e' approssimata (vedi
   simulate_game_score): non usiamo punt_net_yards ne' la posizione reale
   di recupero, solo touchback fisso. E' una semplificazione grossa,
   probabilmente la prima cosa da migliorare.
3. Nessun aggiustamento per garbage time / gestione clock nel quarto
   periodo: il modello tratta ogni drive allo stesso modo indipendentemente
   dal punteggio o dal tempo rimasto, quindi sovrastima l'aggressivita' nel
   quarto periodo con partite gia' decise.
4. goal-to-go (distanza > yardline_100, es. 3rd & 8 dalla 5 yard line) non
   e' gestito separatamente: il distance_bucket usa ydstogo raw, che vicino
   alla end zone puo' essere fuorviante (non puoi guadagnare piu' yard di
   quante te ne separano dalla end zone).
5. L'adjustment per la difesa avversaria e' un blend lineare semplice
   (vedi DriveMarkovModel.transition_probs), non un vero opponent-adjustment
   via regressione come opponent_adjusted_epa in feature_engineering.py.
   Puo' essere il prossimo miglioramento naturale.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# =========================
# DISCRETIZZAZIONE DELLO STATO
# =========================
N_FP_BUCKETS = 10


def fp_bucket(yardline_100: pd.Series) -> pd.Series:
    b = (yardline_100 // 10).clip(0, N_FP_BUCKETS - 1)
    return b.astype("Int64")


def distance_bucket(ydstogo: pd.Series) -> pd.Series:
    return pd.cut(
        ydstogo, bins=[-0.1, 3, 6, 9, 100], labels=[0, 1, 2, 3]
    ).astype("Int64")


OUTCOMES = [
    "TOUCHDOWN", "FIELD_GOAL_MADE", "FIELD_GOAL_MISS",
    "PUNT", "TURNOVER", "TURNOVER_ON_DOWNS", "SAFETY",
]

# Punti attribuiti all'attacco per ciascun esito (extra point assunto
# automatico per semplicita' -- TODO: modellare 2pt conversion e XP miss
# separatamente, oggi sono una minoranza di casi trascurata).
POINTS_FOR_OUTCOME = {
    "TOUCHDOWN": 7.0, "FIELD_GOAL_MADE": 3.0, "FIELD_GOAL_MISS": 0.0,
    "PUNT": 0.0, "TURNOVER": 0.0, "TURNOVER_ON_DOWNS": 0.0, "SAFETY": -2.0,
}


def _state_key(down, dist_b, fp_b):
    return (int(down), int(dist_b), int(fp_b))


# =========================
# ESTRAZIONE TRANSIZIONI DA PBP
# =========================
def build_play_transitions(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per ogni play, determina lo stato pre-snap e la transizione: verso un
    altro stato (prossimo play nella stessa drive) o verso un OUTCOME
    assorbente se il play chiude la drive. Richiede che pbp contenga
    'drive' (nflverse la fornisce gia', vedi PBP_COLS in data_ingestion.py)
    e sia ordinato per tempo di gioco discendente all'interno di ogni game."""
    df = pbp.copy()
    df["dist_b"] = distance_bucket(df["ydstogo"])
    df["fp_b"] = fp_bucket(df["yardline_100"])
    df = df.dropna(subset=["down", "dist_b", "fp_b", "drive"])
    df["down"] = df["down"].astype(int)

    # ordine di gioco decrescente per game_seconds_remaining dentro ogni drive
    df = df.sort_values(["game_id", "drive", "game_seconds_remaining"], ascending=[True, True, False])

    rows = []
    for (game_id, drive_id), drive_df in df.groupby(["game_id", "drive"], sort=False):
        drive_df = drive_df.reset_index(drop=True)
        n = len(drive_df)
        for i in range(n):
            play = drive_df.iloc[i]
            from_state = _state_key(play["down"], play["dist_b"], play["fp_b"])
            is_last = (i == n - 1)

            if not is_last:
                nxt = drive_df.iloc[i + 1]
                to = _state_key(nxt["down"], nxt["dist_b"], nxt["fp_b"])
            else:
                to = _classify_terminal_outcome(play)

            rows.append(dict(
                game_id=game_id, week=play["week"], season=play.get("season", np.nan),
                team=play["posteam"], opponent=play["defteam"],
                from_state=from_state, to=to,
            ))

    return pd.DataFrame(rows)


def _classify_terminal_outcome(play) -> str:
    """Classifica l'ultimo play di una drive in uno degli OUTCOMES. Ordine
    dei controlli non arbitrario: touchdown/turnover hanno precedenza su
    down=4 generico, altrimenti un 4th-down-TD verrebbe scambiato per
    turnover on downs."""
    if play.get("touchdown", 0) == 1:
        return "TOUCHDOWN"
    if pd.notna(play.get("field_goal_result")):
        return "FIELD_GOAL_MADE" if play["field_goal_result"] == "made" else "FIELD_GOAL_MISS"
    if play.get("interception", 0) == 1 or play.get("fumble_lost", 0) == 1:
        return "TURNOVER"
    if pd.notna(play.get("punt_net_yards")):
        return "PUNT"
    if int(play["down"]) == 4:
        return "TURNOVER_ON_DOWNS"
    # Drive finita per fine tempo/quarto senza un esito netto: non e' un
    # vero outcome del gioco, la scartiamo (None -> il chiamante la filtra).
    return None


# =========================
# MODELLO WALK-FORWARD CON SHRINKAGE
# =========================
class DriveMarkovModel:
    """Tiene contatori cumulativi (offense per squadra, league-wide) delle
    transizioni osservate. update() va chiamato settimana per settimana con
    SOLO le transizioni della settimana appena conclusa (walk-forward:
    transition_probs() a inizio settimana W riflette quindi solo drive di
    partite < W, nessun leakage -- stessa disciplina di
    build_walk_forward_features in build_features.py)."""

    def __init__(self, prior_strength: float = 50.0):
        self.prior_strength = prior_strength
        self.off_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # team -> state -> to -> count
        self.def_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # opponent -> state -> to -> count
        self.league_counts = defaultdict(lambda: defaultdict(float))  # state -> to -> count

    def update(self, transitions: pd.DataFrame):
        transitions = transitions.dropna(subset=["to"])
        for row in transitions.itertuples(index=False):
            self.off_counts[row.team][row.from_state][row.to] += 1
            self.def_counts[row.opponent][row.from_state][row.to] += 1
            self.league_counts[row.from_state][row.to] += 1

    def _down_level_fallback(self, down: int) -> dict:
        """Fallback quando lo stato esatto (down, distanza, field position)
        non e' mai stato osservato in lega: aggrega tutte le celle con lo
        stesso down, marginalizzando su distanza e field position. Meno
        preciso di una cella esatta, ma molto meno arbitrario di un caso
        speciale hard-coded solo per down=4 (bug della prima versione: uno
        stato mai visto a down=1/2/3 restituiva probs={} e rompeva
        simulate_drive)."""
        agg = defaultdict(float)
        for state, to_counts in self.league_counts.items():
            if state[0] == down:
                for o, c in to_counts.items():
                    agg[o] += c
        total = sum(agg.values())
        return {k: v / total for k, v in agg.items()} if total > 0 else {"PUNT": 1.0}

    def transition_probs(self, offense: str, defense: str, state: tuple) -> dict:
        """Blend a tre vie: frequenze offensive della squadra, frequenze
        difensive dell'avversario, media di lega -- pesate per numerosita'
        (shrinkage bayesiano, stesso principio di regularize_cpoe in
        feature_engineering.py, qui applicato per cella di stato invece che
        per squadra-partita). E' un blend lineare, non un vero
        opponent-adjustment via regressione: piu' semplice ma meno
        principled dell'equivalente per EPA -- primo candidato per un
        miglioramento futuro."""
        league_cell = self.league_counts.get(state, {})
        league_total = sum(league_cell.values())
        if league_total > 0:
            league_probs = {k: v / league_total for k, v in league_cell.items()}
        else:
            # stato esatto mai visto in lega: fallback aggregato per down
            # invece di lasciare una cella vuota (vedi _down_level_fallback)
            league_probs = self._down_level_fallback(state[0])

        off_cell = self.off_counts.get(offense, {}).get(state, {})
        off_total = sum(off_cell.values())

        def_cell = self.def_counts.get(defense, {}).get(state, {})
        def_total = sum(def_cell.values())

        outcomes = set(league_probs) | set(off_cell) | set(def_cell)

        probs = {}
        w_prior = self.prior_strength
        for o in outcomes:
            p_league = league_probs.get(o, 0.0)
            p_off = (off_cell.get(o, 0.0) / off_total) if off_total > 0 else p_league
            p_def = (def_cell.get(o, 0.0) / def_total) if def_total > 0 else p_league
            # media offesa/difesa, poi shrink verso la lega pesato sulla
            # numerosita' combinata vista dalla squadra in attacco
            team_signal = 0.5 * p_off + 0.5 * p_def
            n_eff = 0.5 * off_total + 0.5 * def_total
            probs[o] = (n_eff * team_signal + w_prior * p_league) / (n_eff + w_prior)

        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 0 else probs

    def simulate_drive(self, offense: str, defense: str, start_state: tuple,
                        rng: np.random.Generator, max_plays: int = 25):
        """Simula una drive play-by-play campionando dalla catena finche' non
        si raggiunge un OUTCOME assorbente. Ritorna (outcome, punti,
        yardline_100 finale -- rilevante solo per TURNOVER, dove serve a
        determinare da dove riparte l'avversario)."""
        state = start_state
        for _ in range(max_plays):
            probs = self.transition_probs(offense, defense, state)
            if not probs:
                return "TURNOVER_ON_DOWNS", 0.0, state[2] * 10 + 5  # fallback prudente
            choices = list(probs.keys())
            p = np.array([probs[c] for c in choices])
            p = p / p.sum()
            idx = rng.choice(len(choices), p=p)
            nxt = choices[idx]
            if nxt in OUTCOMES:
                return nxt, POINTS_FOR_OUTCOME[nxt], state[2] * 10 + 5
            state = nxt
        # non ha raggiunto un outcome entro max_plays (dovrebbe essere
        # rarissimo): tratta come turnover on downs, prudente.
        return "TURNOVER_ON_DOWNS", 0.0, state[2] * 10 + 5


# =========================
# STIMA WALK-FORWARD (settimana per settimana, no leakage)
# =========================
def build_walk_forward_drive_model(pbp: pd.DataFrame, games: pd.DataFrame, prior_strength: float = 50.0):
    """Ricalcola le transizioni settimana per settimana e le accumula nel
    modello via update(), nello stesso schema walk-forward del resto della
    pipeline. Ritorna il DriveMarkovModel finale (contenente TUTTE le
    settimane) -- per un uso walk-forward vero (es. come feature per
    XGBoost), va richiamato dentro un ciclo esterno analogo a quello di
    build_walk_forward_features in build_features.py, aggiornando il
    modello una settimana alla volta e leggendo transition_probs() PRIMA di
    fare update() con quella stessa settimana."""
    model = DriveMarkovModel(prior_strength=prior_strength)
    transitions = build_play_transitions(pbp)
    transitions = transitions.merge(
        games[["game_id", "season"]].drop_duplicates(), on="game_id", how="left", suffixes=("", "_g")
    )

    for (season, week), week_transitions in transitions.sort_values(["season", "week"]).groupby(["season", "week"]):
        model.update(week_transitions)

    return model


# =========================
# SIMULAZIONE PUNTEGGIO PARTITA (da usare al posto del proxy in nfl_model.py)
# =========================
TOUCHBACK_FP = 75  # yardline_100 dopo touchback (circa la propria 25 yard line)


def simulate_game_score(model: DriveMarkovModel, home: str, away: str,
                         rng: np.random.Generator, n_possessions_each: int = 11):
    """Alterna possessi home/away a partire da un touchback, sommando i
    punti di ogni drive simulata. SEMPLIFICAZIONE (vedi limitazioni in cima
    al file): dopo ogni drive la squadra successiva riparte SEMPRE da
    touchback, indipendentemente da field position reale post-punt/turnover
    -- quindi il modello oggi sottostima il vantaggio di field position
    corto dopo un turnover o un punt corto. n_possessions_each e' fisso
    invece di derivare dal ritmo di gioco reale (pace_roll/plays_roll
    calcolati in data_ingestion.py potrebbero dare un numero di possessi
    atteso migliore del valore fisso 11)."""
    home_pts, away_pts = 0.0, 0.0
    for _ in range(n_possessions_each):
        outcome, pts, _ = model.simulate_drive(home, away, (1, 3, TOUCHBACK_FP), rng)
        home_pts += pts
        outcome, pts, _ = model.simulate_drive(away, home, (1, 3, TOUCHBACK_FP), rng)
        away_pts += pts
    return home_pts, away_pts


# =========================
# EXPECTED POINTS PER DRIVE (value iteration sulla catena) -- feature per
# XGBoost, analoga a off_epa_adj/def_epa_adj ma calcolata dalla catena di
# Markov invece che da una ridge regression su EPA/play.
# =========================
LEAGUE_SENTINEL = "__LEAGUE_AVG__"  # chiave mai popolata in off_counts/def_counts:
                                     # usarla come offense o defense fa collassare
                                     # quel lato del blend sulla sola media di lega
                                     # (vedi transition_probs), dando un rating
                                     # "squadra vs avversario medio" invece che
                                     # "squadra vs questo avversario specifico".


def _all_states():
    return [(d, db, fb) for d in range(1, 5) for db in range(4) for fb in range(N_FP_BUCKETS)]


def compute_expected_points(model: DriveMarkovModel, offense: str, defense: str,
                             all_states=None, n_iter: int = 20) -> dict:
    """Value iteration sulla catena assorbente: V(stato) = valore atteso in
    punti da quello stato in poi, dato come giocano offense/defense secondo
    transition_probs(). Non e' una vera simulazione Monte Carlo (quella la
    fa simulate_drive) -- e' il valore atteso esatto (a convergenza), piu'
    adatto come feature perche' deterministico dato il modello, non
    rumoroso come una singola simulazione.

    n_iter=20 e' una scelta pratica, non una prova di convergenza: le drive
    NFL raramente superano 15-18 play, quindi 20 iterazioni di Bellman
    backup dovrebbero essere piu' che sufficienti perche' il valore si
    stabilizzi, ma non e' stato verificato formalmente (TODO: controllare
    |V_k - V_{k-1}| e iterare fino a tolleranza invece di un numero fisso)."""
    all_states = all_states or _all_states()
    V = {s: 0.0 for s in all_states}
    for _ in range(n_iter):
        newV = {}
        for s in all_states:
            probs = model.transition_probs(offense, defense, s)
            val = 0.0
            for to, p in probs.items():
                if to in OUTCOMES:
                    val += p * POINTS_FOR_OUTCOME[to]
                else:
                    val += p * V.get(to, 0.0)
            newV[s] = val
        V = newV
    return V


def build_walk_forward_drive_features(pbp: pd.DataFrame, games: pd.DataFrame,
                                       prior_strength: float = 50.0, n_value_iters: int = 20) -> pd.DataFrame:
    """Walk-forward vero: per ogni settimana, calcola drive_epd_off/def per
    ogni squadra usando SOLO il modello aggiornato con le settimane < W (la
    lettura di compute_expected_points avviene PRIMA di model.update() con i
    dati della settimana corrente, stesso ordine di operazioni del ciclo
    settimanale in build_features.py). Nessun leakage.

    COSTO: per ogni settimana si fa value iteration completa (n_value_iters
    x 160 stati) per ogni squadra x 2 (offense e defense) -- con 32 squadre
    e ~150 settimane in 9 stagioni sono ~9600 chiamate a
    compute_expected_points, ciascuna con ~20*160=3200 lookup di
    transition_probs. E' l'operazione piu' costosa aggiunta finora alla
    pipeline: aspettati diversi minuti su 2016-2024, non secondi."""
    transitions = build_play_transitions(pbp).dropna(subset=["to"])
    model = DriveMarkovModel(prior_strength=prior_strength)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    states = _all_states()
    start_state = (1, 3, fp_bucket(pd.Series([TOUCHBACK_FP])).iloc[0])

    rows = []
    for (season, week), _ in games.sort_values(["season", "week"]).groupby(["season", "week"]):
        for team in teams:
            off_V = compute_expected_points(model, team, LEAGUE_SENTINEL, states, n_iter=n_value_iters)
            def_V = compute_expected_points(model, LEAGUE_SENTINEL, team, states, n_iter=n_value_iters)
            rows.append(dict(
                season=season, week=week, team=team,
                drive_epd_off=off_V.get(start_state, 0.0),
                drive_epd_def=def_V.get(start_state, 0.0),
            ))

        week_transitions = transitions[(transitions["season"] == season) & (transitions["week"] == week)]
        model.update(week_transitions)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_ingestion import load_pbp, load_games

    years = [2023, 2024]
    pbp = load_pbp(years)
    games = load_games(years)

    print("Costruzione transizioni play-by-play...")
    transitions = build_play_transitions(pbp)
    print(f"{len(transitions)} transizioni estratte, {transitions['to'].isna().sum()} scartate (fine quarto senza esito netto)")

    print("\nStima modello (non walk-forward, su tutte le settimane insieme -- solo per ispezione)...")
    model = DriveMarkovModel(prior_strength=50.0)
    model.update(transitions.merge(games[["game_id", "season"]].drop_duplicates(), on="game_id", how="left"))

    # sanity check: probabilita' di touchdown dalla goal-to-go (1st & goal da 3 yard)
    example_state = (1, 0, 0)
    probs = model.transition_probs("KC", "SF", example_state)
    print(f"\nDa 1st & goal ~3yd (KC in attacco vs SF difesa): {probs}")

    rng = np.random.default_rng(42)
    h, a = simulate_game_score(model, "KC", "SF", rng)
    print(f"\nPartita simulata KC vs SF: {h:.0f} - {a:.0f}")
