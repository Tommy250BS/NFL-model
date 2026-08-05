import pandas as pd

feat = pd.read_csv("data/team_week_features.csv")
valid = feat.dropna(subset=["drive_epd_off_markov", "drive_epd_off_ridge", "drive_epd_off", "off_epa_adj", "def_epa_adj", "drive_epd_def", "margin"])

# confronto correlazioni: EPA-ridge vs drive-Markov
print(valid[["drive_epd_off_markov", "drive_epd_off_ridge", "drive_epd_off", "off_epa_adj", "def_epa_adj", "drive_epd_def", "margin"]].corr()["margin"])

# split per storico accumulato dal modello Markov (cold-start check)
valid = valid.copy()
valid["season_bucket"] = pd.cut(
    valid["season"], bins=[2015, 2018, 2021, 2025],
    labels=["2016-18 (poco storico)", "2019-21", "2022-24 (molto storico)"]
)
print()
print(valid.groupby("season_bucket")[["drive_epd_off_markov", "margin"]].corr().iloc[0::2, -1])