"""
Ingestione dati nflverse: play-by-play e schedule/games.

Fonti (pubbliche, gratuite, licenza CC):
- play_by_play_{year}.csv.gz  -> github.com/nflverse/nflverse-data (release "pbp")
- games.csv                   -> github.com/nflverse/nfldata (rest days, spread Vegas,
                                  meteo, roof/surface, QB starter per partita)

Le due fonti si uniscono su game_id (stesso formato: "SEASON_WEEK_AWAY_HOME").
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ssl
import urllib.request

# Fix esplicito del contesto SSL di default per urllib: su macOS (e in
# alcuni venv) l'installazione di certifi da sola non basta, perche' il
# contesto SSL di default di Python non lo usa automaticamente -- va
# collegato a mano. Senza questo fix, download_pbp/download_games falliscono
# con CERTIFICATE_VERIFY_FAILED anche dopo "Install Certificates.command" o
# "pip install certifi", a seconda di come e' configurato l'interprete.
# Se certifi non e' installato, ricade silenziosamente sul comportamento di
# default (nessun fix applicato, stesso comportamento di prima).
try:
    import certifi
    _ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl._create_default_https_context = lambda: _ssl_context
except ImportError:
    pass

DATA_DIR = Path(__file__).parent / "data"

PBP_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv.gz"
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# Alias storici -> sigla corrente della stessa franchigia (nel periodo 2016-2024
# coperto da questo progetto sono gli unici due casi: i Raiders si sono
# trasferiti da Oakland a Las Vegas nel 2020, i Chargers da San Diego a Los
# Angeles nel 2017. Senza questa standardizzazione, l'Elo e il Kalman
# "vedono" due squadre diverse e perdono tutta la storia pre-trasloco.
TEAM_ALIAS = {"OAK": "LV", "SD": "LAC"}


def standardize_team_codes(df: pd.DataFrame, cols=("home_team", "away_team", "posteam", "defteam")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(TEAM_ALIAS)
    return df


# Coordinate approssimative dello stadio (lat, lon) e offset UTC standard
# (senza DST, e' una feature euristica: conta la direzione/ampiezza dello
# spostamento, non serve precisione al minuto) di ogni franchigia. Usate per
# calcolare travel_distance e tz_diff -- viaggi lunghi e cambi di fuso sono
# un fattore di affaticamento noto in letteratura NFL (es. squadre della West
# Coast che giocano 1pm ET perdono sistematicamente di piu').
TEAM_LOCATION = {
    "ARI": (33.5276, -112.2626, -7), "ATL": (33.7554, -84.4008, -5),
    "BAL": (39.2780, -76.6227, -5), "BUF": (42.7738, -78.7870, -5),
    "CAR": (35.2258, -80.8528, -5), "CHI": (41.8623, -87.6167, -6),
    "CIN": (39.0955, -84.5160, -5), "CLE": (41.5061, -81.6995, -5),
    "DAL": (32.7473, -97.0945, -6), "DEN": (39.7439, -105.0201, -7),
    "DET": (42.3400, -83.0456, -5), "GB": (44.5013, -88.0622, -6),
    "HOU": (29.6847, -95.4107, -6), "IND": (39.7601, -86.1639, -5),
    "JAX": (30.3239, -81.6373, -5), "KC": (39.0489, -94.4839, -6),
    "LA": (33.9535, -118.3392, -8), "LAC": (33.9535, -118.3392, -8),
    "LV": (36.0909, -115.1833, -8), "MIA": (25.9580, -80.2389, -5),
    "MIN": (44.9736, -93.2575, -6), "NE": (42.0909, -71.2643, -5),
    "NO": (29.9511, -90.0812, -6), "NYG": (40.8135, -74.0745, -5),
    "NYJ": (40.8135, -74.0745, -5), "PHI": (39.9008, -75.1675, -5),
    "PIT": (40.4468, -80.0158, -5), "SEA": (47.5952, -122.3316, -8),
    "SF": (37.4030, -121.9700, -8), "TB": (27.9759, -82.5033, -5),
    "TEN": (36.1665, -86.7713, -6), "WAS": (38.9078, -76.8645, -5),
}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distanza great-circle in km fra due coordinate."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge travel_distance (km percorsi dalla squadra ospite per
    raggiungere lo stadio della squadra di casa) e tz_diff (differenza di
    fuso orario home - away, positiva se l'ospite viaggia verso est).
    Entrambe note pre-partita, nessun leakage."""
    games = games.copy()
    home_loc = games["home_team"].map(TEAM_LOCATION)
    away_loc = games["away_team"].map(TEAM_LOCATION)

    def _unpack(col, i):
        return col.apply(lambda v: v[i] if isinstance(v, tuple) else np.nan)

    home_lat, home_lon, home_tz = _unpack(home_loc, 0), _unpack(home_loc, 1), _unpack(home_loc, 2)
    away_lat, away_lon, away_tz = _unpack(away_loc, 0), _unpack(away_loc, 1), _unpack(away_loc, 2)

    games["travel_distance"] = _haversine_km(away_lat, away_lon, home_lat, home_lon)
    games["tz_diff"] = home_tz - away_tz
    return games


ABBR_TO_FULLNAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

# Colonne pbp effettivamente necessarie a valle: teniamole minime per
# non trascinarsi 372 colonne in memoria per stagioni multiple.
PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "home_team", "away_team",
    "play_type", "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
    "epa", "success", "cpoe", "pass_attempt", "complete_pass", "rush_attempt",
    "wp", "posteam_score", "defteam_score", "score_differential",
    "fumble_lost", "interception", "touchdown", "field_goal_result",
    "punt_net_yards", "drive", "qb_dropback", "sack", "penalty",
]


def download_pbp(year: int, force: bool = False) -> Path:
    out = DATA_DIR / f"pbp_{year}.csv.gz"
    if out.exists() and not force:
        return out
    import urllib.request
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(PBP_URL_TMPL.format(year=year), out)
    return out


def download_games(force: bool = False) -> Path:
    out = DATA_DIR / "games.csv"
    if out.exists() and not force:
        return out
    import urllib.request
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(GAMES_URL, out)
    return out


def load_pbp(years) -> pd.DataFrame:
    """Carica e concatena il play-by-play per una lista di stagioni."""
    frames = []
    for yr in years:
        path = download_pbp(yr)
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        keep = [c for c in PBP_COLS if c in df.columns]
        frames.append(df[keep])
    pbp = pd.concat(frames, ignore_index=True)
    pbp = standardize_team_codes(pbp)

    # Solo play "reali" da scrimmage: escludiamo timeout, kneel, spike, no-play
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["epa", "posteam", "defteam"])
    return pbp


def load_games(seasons=None) -> pd.DataFrame:
    """Carica lo schedule con rest days, spread Vegas, meteo, QB starter."""
    path = download_games()
    games = pd.read_csv(path, low_memory=False)
    games = standardize_team_codes(games)
    if seasons is not None:
        games = games[games["season"].isin(seasons)].copy()

    # Rest differential: gia' presente come away_rest/home_rest (giorni dall'ultima partita)
    games["rest_diff"] = games["home_rest"] - games["away_rest"]

    # Flag indoor/outdoor per neutralizzare l'effetto vento su squadre in cupola
    games["is_dome"] = games["roof"].isin(["dome", "closed"]).astype(int)

    # Travel/timezone: pre-partita, note dallo schedule, nessun leakage
    games = add_travel_features(games)

    # Cambio QB rispetto alla partita precedente della stessa squadra (usato
    # dall'Elo QB-adjusted per applicare lo shock di rating quando cambia titolare)
    games = games.sort_values(["season", "week"])
    for side in ["home", "away"]:
        team_col = f"{side}_team"
        qb_col = f"{side}_qb_id"
        games[f"{side}_qb_changed"] = (
            games.groupby(team_col)[qb_col].transform(lambda s: s != s.shift(1))
        )
        # prima partita della squadra nel dataset: non è un "cambio", è baseline
        games.loc[games.groupby(team_col).cumcount() == 0, f"{side}_qb_changed"] = False

    return games


INJURIES_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv"

# Peso per posizione nell'indice di "carico infortuni": un QB o un OL titolare
# fuori pesa piu' di uno special teamer. Pesi euristici, non stimati -- è un
# indice relativo, non una probabilita' calibrata.
INJURY_POSITION_WEIGHT = {
    "QB": 3.0, "OT": 1.5, "OG": 1.3, "C": 1.3, "WR": 1.2, "EDGE": 1.5,
    "DE": 1.3, "CB": 1.2, "RB": 1.0, "TE": 1.0, "LB": 1.0, "S": 1.0,
    "DT": 1.0, "NT": 1.0, "FB": 0.5, "K": 0.3, "P": 0.3, "LS": 0.2,
}
INJURY_STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.35}


def download_injuries(year: int, force: bool = False) -> Path:
    out = DATA_DIR / f"injuries_{year}.csv"
    if out.exists() and not force:
        return out
    import urllib.request
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(INJURIES_URL_TMPL.format(year=year), out)
    return out


def load_injury_burden(years) -> pd.DataFrame:
    """Indice di carico infortuni per squadra-settimana: somma pesata
    (posizione x gravita' del report) dei giocatori Out/Doubtful/Questionable,
    piu' un flag separato 'qb_out' per il titolare (che l'Elo QB-adjusted usa
    gia' tramite qb_id, ma qui e' un secondo segnale indipendente utile a
    XGBoost come feature di controllo)."""
    frames = []
    for yr in years:
        path = download_injuries(yr)
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
    inj = pd.concat(frames, ignore_index=True)
    inj = standardize_team_codes(inj, cols=("team",))

    inj["status_w"] = inj["report_status"].map(INJURY_STATUS_WEIGHT).fillna(0.0)
    inj["pos_w"] = inj["position"].map(INJURY_POSITION_WEIGHT).fillna(0.8)
    inj["burden"] = inj["status_w"] * inj["pos_w"]
    inj["is_qb_out"] = (inj["position"] == "QB") & (inj["report_status"].isin(["Out", "Doubtful"]))

    agg = (
        inj.groupby(["season", "week", "team"])
        .agg(injury_burden=("burden", "sum"), qb_out=("is_qb_out", "any"))
        .reset_index()
    )
    agg["qb_out"] = agg["qb_out"].astype(int)
    return agg



def build_team_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggrega il pbp a livello squadra-partita: EPA/play offensivo e
    difensivo, success rate, CPOE medio (grezzo, non ancora regolarizzato),
    piu' feature situazionali (3rd down, red zone, turnover) che non sono
    catturate dall'EPA medio complessivo -- una squadra puo' avere EPA/play
    mediocre ma essere elite in red zone o sui third down, ed e' segnale
    ortogonale rispetto a quello che l'Elo gia' vede tramite il risultato
    finale."""
    third_down = pbp[pbp["down"] == 3]
    redzone = pbp[pbp["yardline_100"] <= 20]

    off = (
        pbp.groupby(["game_id", "week", "posteam"])
        .agg(
            off_epa_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            off_plays=("epa", "size"),
            cpoe_raw=("cpoe", "mean"),
            cpoe_attempts=("cpoe", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    # Pace: secondi di game clock consumati per play, proxy grezzo (usa il
    # range di game_seconds_remaining fra il primo e l'ultimo play della
    # squadra in quella partita / numero di play). Non e' il vero tempo fra
    # snap e snap (quello richiederebbe i timestamp reali, non disponibili
    # nelle colonne che teniamo), ma cattura comunque se una squadra tende a
    # "consumare" il game clock rapidamente (hurry-up) o lentamente
    # (ball-control) -- segnale ortogonale a EPA/play.
    pace = (
        pbp.groupby(["game_id", "week", "posteam"])["game_seconds_remaining"]
        .agg(lambda s: (s.max() - s.min()))
        .reset_index()
        .rename(columns={"posteam": "team", "game_seconds_remaining": "_clock_span"})
    )
    off = off.merge(pace, on=["game_id", "week", "team"], how="left")
    off["seconds_per_play"] = off["_clock_span"] / off["off_plays"].replace(0, np.nan)
    off = off.drop(columns="_clock_span")
    to_lost = (
        pbp.groupby(["game_id", "week", "posteam"])
        .agg(fumbles_lost=("fumble_lost", "sum"), interceptions_thrown=("interception", "sum"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    to_lost["turnovers_lost"] = to_lost["fumbles_lost"] + to_lost["interceptions_thrown"]
    off = off.merge(to_lost[["game_id", "week", "team", "turnovers_lost"]], on=["game_id", "week", "team"], how="left")

    dfn = (
        pbp.groupby(["game_id", "week", "defteam"])
        .agg(def_epa_play=("epa", "mean"), def_success_rate=("success", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    to_forced = (
        pbp.groupby(["game_id", "week", "defteam"])
        .agg(fumbles_forced=("fumble_lost", "sum"), interceptions_forced=("interception", "sum"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    to_forced["turnovers_forced"] = to_forced["fumbles_forced"] + to_forced["interceptions_forced"]
    dfn = dfn.merge(to_forced[["game_id", "week", "team", "turnovers_forced"]], on=["game_id", "week", "team"], how="left")

    third_agg = (
        third_down.groupby(["game_id", "week", "posteam"])["epa"].mean()
        .reset_index().rename(columns={"posteam": "team", "epa": "off_epa_3rd_down"})
    )
    rz_agg = (
        redzone.groupby(["game_id", "week", "posteam"])["epa"].mean()
        .reset_index().rename(columns={"posteam": "team", "epa": "off_epa_redzone"})
    )

    out = off.merge(dfn, on=["game_id", "week", "team"], how="outer")
    out = out.merge(third_agg, on=["game_id", "week", "team"], how="left")
    out = out.merge(rz_agg, on=["game_id", "week", "team"], how="left")
    return out


if __name__ == "__main__":
    pbp = load_pbp([2023, 2024])
    games = load_games([2023, 2024])
    tg = build_team_game_table(pbp)
    print("pbp plays:", len(pbp))
    print("games:", len(games))
    print("team-game rows:", len(tg))
    print(tg.head())