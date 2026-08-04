"""
Diagrammi di calibrazione (reliability diagram): confronta la probabilita'
predetta con la frequenza osservata, per bin di probabilita'.

PERCHE'
-------
Log loss e Brier score sono medie: un modello puo' avere una media buona ma
essere sistematicamente scalibrato in certe fasce (es. sovrastimare sempre le
vittorie nette 80%+, o essere troppo prudente vicino al 50%). Non lo si vede
guardando solo log loss/Brier aggregati -- serve il breakdown per bin.

Uso: dopo aver girato stacking_model.run_backtest() (o
walk_forward_backtest.py), passare qui i dict {nome_modello: array probabilita'}
insieme all'array di risultati reali (home_win).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin per decili di probabilita' predetta (bin a numerosita' uguale,
    non a larghezza uguale: con poche centinaia di partite un binning a
    larghezza fissa lascia bin quasi vuoti agli estremi, dove serve invece
    vedere se il modello e' overconfident)."""
    p = np.asarray(p)
    y = np.asarray(y)
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    bin_edges = np.array_split(np.arange(len(p)), n_bins)

    rows = []
    for b in bin_edges:
        if len(b) == 0:
            continue
        rows.append(dict(
            n=len(b),
            p_mean=p_sorted[b].mean(),
            p_min=p_sorted[b].min(),
            p_max=p_sorted[b].max(),
            y_freq=y_sorted[b].mean(),
        ))
    return pd.DataFrame(rows)


def plot_reliability(preds: dict, y: np.ndarray, n_bins: int = 10, out_path: str = "calibration.png"):
    """preds: {nome_modello: array probabilita'}. Disegna un reliability
    diagram con tutti i modelli sovrapposti + la diagonale ideale."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibrazione perfetta")

    tables = {}
    for name, p in preds.items():
        tbl = reliability_table(p, y, n_bins=n_bins)
        tables[name] = tbl
        ax.plot(tbl["p_mean"], tbl["y_freq"], marker="o", label=name)

    ax.set_xlabel("Probabilita' predetta (media per bin)")
    ax.set_ylabel("Frequenza osservata di vittoria home")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return tables, out_path


if __name__ == "__main__":
    # Esempio d'uso standalone: rigira il backtest di stacking_model e produce
    # il grafico. Richiede data/team_week_features.csv gia' costruito.
    from stacking_model import run_backtest

    model, alpha, results = run_backtest()
    feat = pd.read_csv("data/team_week_features.csv")
    feat = feat.dropna(subset=["margin", "home_win"])
    test = feat[feat["season"] == 2024]
    y = test["home_win"].values

    tables, out_path = plot_reliability(results, y, out_path="data/calibration_2024.png")
    for name, tbl in tables.items():
        print(f"\n--- {name} ---")
        print(tbl.to_string(index=False))
    print(f"\nGrafico salvato in {out_path}")