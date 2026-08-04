from build_features import build_walk_forward_features

feat = build_walk_forward_features(
    [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    include_drive_features=True,
    drive_n_value_iters=10,
    drive_prior_strength=15.0,   # invece di 50 default: meno shrinkage verso la lega
)
print(feat[["drive_epd_off", "drive_epd_def"]].describe())
feat.to_csv("data/team_week_features_lowshrink.csv", index=False)