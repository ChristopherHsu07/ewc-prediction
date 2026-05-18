from pathlib import Path
from src.preprocess import build_team_df
from src.preprocess import build_team_profiles
from src.features import build_matchups
from src.model import train_model
from src.model import evaluate_model

filename = "data/2025_LoL_esports_match_data_from_OraclesElixir.csv"
team_df = build_team_df(filename)
team_profiles = build_team_profiles(team_df)
matchups = build_matchups(team_df)

model, scaler, feature_cols, x_test_scaled, y_test = train_model(matchups)
evaluate_model(model, x_test_scaled, y_test, feature_cols)