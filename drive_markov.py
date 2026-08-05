"""
Modello drive-by-drive via Markov chain su stati (down, distanza, field
position), walk-forward come il resto della pipeline.

STATO E TRANSIZIONI
--------------------
Uno stato e' la tripla (down, distance_bucket, fp_bucket):
- down: 1-4
- distance_bucket: 0 = 1-3 yard, 1 = 4-6, 2 = 7-9, 3 = 10+, 4 = goal-to-go
  dedicato (quando la colonna 'goal_to_go' di nflverse segnala che la
  distanza dalla end zone e' il vincolo reale, non i canonici bucket a
  fasce -- vedi fix goal-to-go piu' sotto)
- fp_bucket: yardline_100 // 10, clippato a [0,9]. yardline_100 e' la
  convenzione nflverse: 0 = end zone avversaria (goal line offensiva),
  100 = propria end zone. fp_bucket=0 quindi vuol dire "vicino a segnare".

Ogni play e' una transizione da uno stato pre-snap a: un altro stato
(prossimo down/distanza/field position nella stessa drive) oppure a un
ESITO assorbente che chiude la drive (touchdown, field goal, punt,
turnover, turnover su downs, safety). Il numero di stati e' 4 down x 5
distanze (incluso goal-to-go dedicato) x 10 field position = 200, ancora
contenuto apposta per restare stimabile senza sparsita' estrema, stesso
spirito di ridge_lambda alto nell'opponent-adjusted EPA: con ~11 drive/partita
a squadra i dati per cella sono comunque pochi, quindi qui usiamo shrinkage
bayesiano verso la media di lega (stesso principio di regularize_cpoe)
invece di una ridge.

BUG STORICO (risolto): transition_probs faceva team_signal = 0.5*p_off +
0.5*p_def con n_eff dimezzato. compute_expected_points isola sempre un lato
con LEAGUE_SENTINEL (nessun conteggio -> quel lato ricade su p_league per
costruzione), quindi meta' del segnale era SEMPRE p_league a prescindere
dalla numerosita' reale: un tetto strutturale di ~50% di peso massimo sul
segnale squadra, che rendeva drive_prior_strength quasi ininfluente. Ora il
blend pesa p_off/p_def per la loro numerosita' reale e n_eff = off_total +
def_total (vedi commento inline in transition_probs).

LIMITAZIONI NOTE (da affrontare prima di usarlo in produzione)
----------------------------------------------------------------
1. Nessun aggiustamento per garbage time / gestione clock nel quarto
   periodo: il modello tratta ogni drive allo stesso modo indipendentemente
   dal punteggio o dal tempo rimasto, quindi sovrastima l'aggressivita' nel
   quarto periodo con partite gia' decise.

RISOLTE:
- Safety ora classificata esplicitamente (richiede la colonna 'safety' in
  PBP_COLS, vedi data_ingestion.py) invece di essere scartata silenziosamente.
- goal-to-go gestito con un distance_bucket dedicato (richiede la colonna
  'goal_to_go' in PBP_COLS) invece di lasciare ydstogo grezzo vicino alla
  end zone, dove puo' essere fuorviante.
- Value iteration ora itera fino a convergenza (tolleranza) invece di un
  numero fisso di passi non verificato.
- Field position dopo punt/turnover non e' piu' sempre un touchback fisso:
  simulate_game_score ora usa _next_start_yardline per derivare la field
  position del possesso successivo dall'esito reale della drive precedente
  (approssimato via net punt medio di lega, non simulazione play-by-play del
  return -- vedi _next_start_yardline per il dettaglio). Come bonus, ha
  anche corretto un piccolo bug preesistente in simulate_game_score che
  passava TOUCHBACK_FP (75, uno yardline) come se fosse gia' un fp_bucket
  (0-9) nello stato iniziale.
- L'opponent-adjustment per le drive feature non e' piu' solo il blend
  lineare per-stato: build_walk_forward_drive_features ora calcola anche un
  rating ridge opponent-adjusted (riusando opponent_adjusted_epa gia'
  testato in feature_engineering.py) sui punti-per-drive, e lo blenda col
  valore Markov (drive_ridge_weight, default 0.5). Il blend per-stato dentro
  transition_probs resta comunque lineare (non e' stato riscritto l'intero
  motore a catena di Markov) -- il fix aggiunge un secondo segnale
  principled invece di sostituire l'architettura esistente.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from feature_engineering import opponent_adjusted_epa

# =========================
# DISCRETIZZAZIONE DELLO STATO
# =========================
N_FP_BUCKETS = 10
# 5 bucket di distanza: 0-3 = distanza normale a fasce (come prima), 4 =
# goal-to-go dedicato. Prima del fix, un 3rd & 8 dalla 5 yard line (che e'
# goal-to-go: non puoi guadagnare piu' delle 5 yard che ti separano dalla end
# zone anche se "servirebbero" 8 yard per il primo down) finiva nel bucket
# "7-9" insieme a normali 3rd & 8 a centrocampo -- due situazioni con
# probabilita' di touchdown molto diverse mischiate nella stessa cella.
N_DISTANCE_BUCKETS = 5
GOAL_TO_GO_BUCKET = 4


def fp_bucket(yardline_100: pd.Series) -> pd.Series:
    b = (yardline_100 // 10).clip(0, N_FP_BUCKETS - 1)
    return b.astype("Int64")


def distance_bucket(ydstogo: pd.Series, goal_to_go: pd.Series = None) -> pd.Series:
    """goal_to_go: flag booleano/0-1 da nflverse (colonna 'goal_to_go' in
    PBP_COLS). Se assente (None), si comporta come prima del fix (nessuna
    distinzione goal-to-go) -- serve per compatibilita' con eventuali
    chiamate che non hanno ancora la colonna disponibile."""
    base = pd.cut(
        ydstogo, bins=[-0.1, 3, 6, 9, 100], labels=[0, 1, 2, 3]
    ).astype("Int64")
    if goal_to_go is None:
        return base
    is_gtg = goal_to_go.fillna(0).astype(int) == 1
    return base.mask(is_gtg, GOAL_TO_GO_BUCKET)


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
    df["dist_b"] = distance_bucket(df["ydstogo"], df.get("goal_to_go"))
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
    if play.get("safety", 0) == 1:
        return "SAFETY"
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

        # FIX BUG: la versione precedente faceva team_signal = 0.5*p_off +
        # 0.5*p_def con n_eff = 0.5*off_total + 0.5*def_total. Quando uno dei
        # due lati e' LEAGUE_SENTINEL (come in compute_expected_points, dove
        # si isola offense o defense per costruzione), quel lato ha SEMPRE
        # off_total=0 o def_total=0 -> p_off o p_def ricade su p_league per
        # definizione. Risultato: meta' del team_signal era sempre p_league
        # a prescindere da quanti dati avesse la squadra sull'altro lato, e
        # n_eff era dimezzato inutilmente -> il peso massimo raggiungibile
        # dal segnale reale era ~50% * (n/(n+prior)), un tetto strutturale
        # che rendeva drive_prior_strength quasi ininfluente (da qui la bassa
        # sensibilita' osservata abbassando lo shrinkage da 50 a 15).
        #
        # Fix: pesa p_off/p_def per la loro numerosita' reale (nessun
        # contributo fittizio di p_league quando un lato manca del tutto:
        # con LEAGUE_SENTINEL il lato mancante e' semplicemente escluso dal
        # blend, non sostituito da p_league) e usa n_eff = off_total +
        # def_total (non dimezzato), cosi' con dati sufficienti il segnale
        # squadra puo' davvero dominare lo shrinkage.
        probs = {}
        w_prior = self.prior_strength
        n_eff = off_total + def_total
        for o in outcomes:
            p_league = league_probs.get(o, 0.0)
            p_off = (off_cell.get(o, 0.0) / off_total) if off_total > 0 else None
            p_def = (def_cell.get(o, 0.0) / def_total) if def_total > 0 else None

            if p_off is None and p_def is None:
                team_signal = p_league
            elif p_off is None:
                team_signal = p_def
            elif p_def is None:
                team_signal = p_off
            else:
                team_signal = (off_total * p_off + def_total * p_def) / (off_total + def_total)

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


def _drive_points_team_game_table(terminal_transitions: pd.DataFrame) -> pd.DataFrame:
    """FIX 3 (limitazione nota #3): aggrega le transizioni terminali (una per
    drive: to in OUTCOMES) a punti-per-drive per squadra-partita, nel formato
    richiesto da opponent_adjusted_epa (feature_engineering.py) -- che si
    aspetta le colonne 'team', 'opponent', 'off_epa_play'. Riusiamo quel nome
    di colonna solo per compatibilita' diretta con la funzione gia' testata:
    qui il contenuto e' punti medi per drive, non EPA/play, ma il principio
    di ridge opponent-adjustment (off_rating - def_rating verso la media di
    lega) e' esattamente lo stesso identico meccanismo, solo su una metrica
    diversa. 'opponent' e' gia' presente per riga nelle transizioni (defteam)
    ed e' costante entro una game_id-team (una sola avversaria per partita),
    quindi basta un 'first' nel groupby, senza bisogno di ricostruire la
    tabella con build_opponent_table come si fa per l'EPA."""
    term = terminal_transitions.copy()
    term["points"] = term["to"].map(POINTS_FOR_OUTCOME)
    agg = (
        term.groupby(["game_id", "team"])
        .agg(off_epa_play=("points", "mean"), opponent=("opponent", "first"))
        .reset_index()
    )
    return agg


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
AVG_NET_PUNT_YARDS = 40.0  # media NFL storica di punt netti (dopo eventuale
                           # return), usata come approssimazione: non simuliamo
                           # il return play per play, ma almeno la field
                           # position del possesso successivo dipende ora da
                           # dove la drive precedente si e' effettivamente
                           # fermata, invece di essere sempre un touchback.


def _next_start_yardline(outcome: str, offense_yardline_100: float) -> float:
    """FIX 1 (limitazione nota #1): data la squadra in attacco a
    offense_yardline_100 (yardline_100, sua prospettiva) quando la drive
    finisce con `outcome`, ritorna lo yardline_100 di partenza della squadra
    che ricevera' il prossimo possesso (dalla LORO prospettiva).

    Resta un'approssimazione (non simuliamo il singolo play di punt/return
    ne' usiamo punt_net_yards reale per squadra, solo una media di lega
    fissa), ma sostituisce il touchback fisso sempre uguale con una field
    position che dipende da dove la drive precedente si e' davvero conclusa
    -- il modello ora "sente" il vantaggio di una difesa che forza turnover
    in territorio avversario o una drive che si spinge in field goal range
    prima di fallire, invece di azzerare sempre quel segnale.

    Casi:
    - TOUCHDOWN / FIELD_GOAL_MADE: segue un kickoff -> touchback standard
      per l'altra squadra (semplificazione ragionevole: i kickoff return
      oltre il touchback sono una minoranza).
    - PUNT: nuova posizione = flip di (offense_yardline_100 - net punt medio),
      clippata a un touchback se il punt "esce" dal campo (rete negativa).
    - TURNOVER / TURNOVER_ON_DOWNS / FIELD_GOAL_MISS: cambio possesso sullo
      stesso punto del campo (nessun guadagno/perdita aggiuntivo), solo flip
      di prospettiva -- questo e' il caso che prima veniva interamente perso
      (turnover in territorio avversario dava comunque touchback all'avversario).
    - SAFETY: free kick dopo safety, approssimato anch'esso a touchback
      (semplificazione nota, il free kick da linea 20 raramente sposta molto
      la field position media).
    """
    if outcome in ("TOUCHDOWN", "FIELD_GOAL_MADE", "SAFETY"):
        return float(TOUCHBACK_FP)
    if outcome == "PUNT":
        landing = offense_yardline_100 - AVG_NET_PUNT_YARDS
        if landing <= 0:
            # punt in touchback (rete che finisce oltre la end zone avversaria)
            return float(TOUCHBACK_FP)
        return float(np.clip(100 - landing, 1, 99))
    if outcome in ("TURNOVER", "TURNOVER_ON_DOWNS", "FIELD_GOAL_MISS"):
        return float(np.clip(100 - offense_yardline_100, 1, 99))
    return float(TOUCHBACK_FP)


def simulate_game_score(model: DriveMarkovModel, home: str, away: str,
                         rng: np.random.Generator, n_possessions_each: int = 11):
    """Alterna possessi home/away, sommando i punti di ogni drive simulata.
    FIX 1: la field position di partenza di ogni possesso ora dipende
    dall'esito REALE della drive precedente (via _next_start_yardline),
    invece di ripartire sempre da un touchback fisso indipendentemente da
    dove/come la drive precedente si e' conclusa -- vedi _next_start_yardline
    per il dettaglio di come ogni esito viene tradotto in field position.
    Resta comunque un'approssimazione (net punt medio di lega fisso, nessun
    return play reale simulato -- vedi docstring di _next_start_yardline).

    n_possessions_each e' ancora fisso invece di derivare dal ritmo di gioco
    reale (pace_roll/plays_roll calcolati in data_ingestion.py potrebbero
    dare un numero di possessi atteso migliore del valore fisso 11)."""
    home_pts, away_pts = 0.0, 0.0
    home_start_fp = float(TOUCHBACK_FP)  # apertura partita: kickoff -> touchback
    away_start_fp = float(TOUCHBACK_FP)
    for _ in range(n_possessions_each):
        home_start_state = (1, 3, int(min(home_start_fp // 10, N_FP_BUCKETS - 1)))
        outcome, pts, end_fp = model.simulate_drive(home, away, home_start_state, rng)
        home_pts += pts
        away_start_fp = _next_start_yardline(outcome, end_fp)

        away_start_state = (1, 3, int(min(away_start_fp // 10, N_FP_BUCKETS - 1)))
        outcome, pts, end_fp = model.simulate_drive(away, home, away_start_state, rng)
        away_pts += pts
        home_start_fp = _next_start_yardline(outcome, end_fp)
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
    return [(d, db, fb) for d in range(1, 5) for db in range(N_DISTANCE_BUCKETS) for fb in range(N_FP_BUCKETS)]


def compute_expected_points(model: DriveMarkovModel, offense: str, defense: str,
                             all_states=None, n_iter: int = 20, tol: float = 1e-4) -> dict:
    """Value iteration sulla catena assorbente: V(stato) = valore atteso in
    punti da quello stato in poi, dato come giocano offense/defense secondo
    transition_probs(). Non e' una vera simulazione Monte Carlo (quella la
    fa simulate_drive) -- e' il valore atteso esatto (a convergenza), piu'
    adatto come feature perche' deterministico dato il modello, non
    rumoroso come una singola simulazione.

    n_iter e' ora un CAP massimo di sicurezza, non il numero di iterazioni
    effettivo: si itera finche' il delta massimo fra V_k e V_{k-1} scende
    sotto `tol`, e ci si ferma prima se la convergenza arriva prima (le
    drive NFL raramente superano 15-18 play, quindi in pratica converge in
    poche iterazioni). Se non converge entro n_iter (dovrebbe essere raro
    dato che la catena e' sub-stocastica con esiti assorbenti), si esce
    comunque con l'ultimo V calcolato invece di girare all'infinito."""
    all_states = all_states or _all_states()
    V = {s: 0.0 for s in all_states}
    for _ in range(n_iter):
        newV = {}
        max_delta = 0.0
        for s in all_states:
            probs = model.transition_probs(offense, defense, s)
            val = 0.0
            for to, p in probs.items():
                if to in OUTCOMES:
                    val += p * POINTS_FOR_OUTCOME[to]
                else:
                    val += p * V.get(to, 0.0)
            newV[s] = val
            max_delta = max(max_delta, abs(val - V[s]))
        V = newV
        if max_delta < tol:
            break
    return V


def build_walk_forward_drive_features(pbp: pd.DataFrame, games: pd.DataFrame,
                                       prior_strength: float = 50.0, n_value_iters: int = 20,
                                       drive_ridge_lambda: float = 25.0,
                                       drive_ridge_weight: float = 0.5,
                                       min_drives_for_ridge: int = 20) -> pd.DataFrame:
    """Walk-forward vero: per ogni settimana, calcola drive_epd_off/def per
    ogni squadra usando SOLO il modello aggiornato con le settimane < W (la
    lettura di compute_expected_points avviene PRIMA di model.update() con i
    dati della settimana corrente, stesso ordine di operazioni del ciclo
    settimanale in build_features.py). Nessun leakage.

    FIX 3 (limitazione nota #3): oltre al valore atteso dalla catena di
    Markov (shrinkage bayesiano per-stato, gia' fixato dal bug del blend
    50/50), calcola ANCHE un rating opponent-adjusted via ridge regression
    (off_rating/def_rating, riusando opponent_adjusted_epa gia' testato in
    feature_engineering.py) sui punti-per-drive osservati fin qui, stesso
    principio dell'opponent-adjustment che gia' usiamo per l'EPA in
    build_features.py. Il valore finale e' un blend pesato
    (drive_ridge_weight) fra i due segnali:
      - il valore Markov cattura la dinamica per-stato (down/distanza/field
        position), ma resta un blend lineare semplice per l'opponent
        adjustment (vedi transition_probs);
      - il rating ridge e' un opponent-adjustment principled (stesso
        meccanismo della ridge EPA) ma aggregato a livello di drive intera,
        senza dettaglio per-stato.
    drive_ridge_weight=0.0 disattiva il rating ridge (comportamento
    equivalente alla versione pre-fix-3, solo Markov). drive_ridge_weight=1.0
    usa solo il rating ridge, ignorando la value iteration per-stato.
    min_drives_for_ridge: se le drive concluse fin qui sono meno di questa
    soglia (early season / prime settimane 2016), il rating ridge non e'
    affidabile -> peso ridge forzato a 0 per quella settimana, si ricade sul
    solo Markov (che ha comunque il proprio shrinkage verso la lega).

    COSTO: per ogni settimana si fa value iteration completa (n_value_iters
    x 200 stati, incluso il bucket goal-to-go dedicato) per ogni squadra x 2
    (offense e defense) -- con 32 squadre e ~150 settimane in 9 stagioni sono
    ~9600 chiamate a compute_expected_points. La ridge aggiuntiva (fix 3) e'
    invece economica (lsqr sparsa su 2*32 colonne, poche migliaia di
    osservazioni al massimo) e non cambia sensibilmente il tempo totale
    rispetto alla sola value iteration, che resta il collo di bottiglia:
    aspettati diversi minuti su 2016-2024, non secondi."""
    transitions = build_play_transitions(pbp).dropna(subset=["to"])
    terminal = transitions[transitions["to"].isin(OUTCOMES)].copy()

    model = DriveMarkovModel(prior_strength=prior_strength)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    states = _all_states()
    start_state = (1, 3, fp_bucket(pd.Series([TOUCHBACK_FP])).iloc[0])

    rows = []
    for (season, week), _ in games.sort_values(["season", "week"]).groupby(["season", "week"]):
        # --- FIX 3: rating ridge opponent-adjusted su tutte le drive
        # concluse STRETTAMENTE prima di questa settimana (stessa disciplina
        # walk-forward di off_epa_adj/def_epa_adj in build_features.py) ---
        past_terminal = terminal[(terminal["season"] < season) | ((terminal["season"] == season) & (terminal["week"] < week))]
        agg = _drive_points_team_game_table(past_terminal) if len(past_terminal) > 0 else pd.DataFrame(columns=["game_id", "team", "off_epa_play", "opponent"])

        if len(agg) >= min_drives_for_ridge:
            off_rating, def_rating = opponent_adjusted_epa(agg, ridge_lambda=drive_ridge_lambda)
            league_avg_ppd = agg["off_epa_play"].mean()
            w = drive_ridge_weight
        else:
            off_rating = {t: 0.0 for t in teams}
            def_rating = {t: 0.0 for t in teams}
            league_avg_ppd = 0.0
            w = 0.0  # dati insufficienti: nessuna fiducia nel rating ridge, solo Markov

        for team in teams:
            off_V = compute_expected_points(model, team, LEAGUE_SENTINEL, states, n_iter=n_value_iters)
            def_V = compute_expected_points(model, LEAGUE_SENTINEL, team, states, n_iter=n_value_iters)
            markov_off = off_V.get(start_state, 0.0)
            markov_def = def_V.get(start_state, 0.0)

            # off_rating[team] ~= quanto l'attacco della squadra segna sopra/
            # sotto la media di lega per drive, aggiustato per gli avversari
            # affrontati (stesso segno/convenzione di off_epa_adj). def_rating
            # analogo ma per la difesa; punti attesi CONCEDUTI = media lega -
            # def_rating[team] (vedi opponent_adjusted_epa: epa ~= off[team] -
            # def[opponent] + media, quindi il termine che riduce i punti
            # dell'attacco avversario e' -def_rating[team]).
            ridge_off = league_avg_ppd + off_rating.get(team, 0.0)
            ridge_def = league_avg_ppd - def_rating.get(team, 0.0)

            drive_epd_off = w * ridge_off + (1 - w) * markov_off
            drive_epd_def = w * ridge_def + (1 - w) * markov_def

            rows.append(dict(
                season=season, week=week, team=team,
                drive_epd_off=drive_epd_off, drive_epd_def=drive_epd_def,
                # colonne diagnostiche: utili per confrontare i due segnali
                # separatamente (es. correlazione con margin) senza dover
                # rigenerare tutto -- non fanno parte di FEATURE_COLS quindi
                # non influenzano XGBoost a meno di aggiungerle esplicitamente.
                drive_epd_off_markov=markov_off, drive_epd_def_markov=markov_def,
                drive_epd_off_ridge=ridge_off, drive_epd_def_ridge=ridge_def,
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
