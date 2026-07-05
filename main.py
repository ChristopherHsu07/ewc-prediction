import warnings
from sklearn.model_selection import train_test_split

from src.preprocess import build_team_df, build_team_profiles, build_region_weights
from src.features import build_matchups
from src.model import fit_model, evaluate_predict_matchup, predict_matchup
from src.control import (
    compute_pregame_win_rates,
    attach_pregame_win_rates,
    evaluate_winrate_control,
    build_cumulative_win_rates,
    predict_winrate_matchup,
)

warnings.filterwarnings('ignore')
filenames = [
    "data/2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "data/2026_LoL_esports_match_data_from_OraclesElixir.csv",
]
team_df = build_team_df(filenames)

INTERNATIONAL_MODE = True  # set to False for domestic sample matchup predictions

matchups = build_matchups(team_df)
pregame = compute_pregame_win_rates(team_df)
matchups = attach_pregame_win_rates(matchups, pregame)

region_weights = build_region_weights(team_df, verbose=True)

train_matchups, test_matchups = train_test_split(
    matchups, test_size=0.2, random_state=42
)

train_gameids = train_matchups['gameid']
train_team_df = team_df[team_df['gameid'].isin(train_gameids)]

train_profiles_domestic = build_team_profiles(train_team_df, intl_boost=1.0)
train_profiles_intl = build_team_profiles(train_team_df, intl_boost=3.0)
train_weights = build_region_weights(train_team_df)

model, scaler, feature_cols = fit_model(train_matchups)

control_acc = evaluate_winrate_control(test_matchups)
predict_acc = evaluate_predict_matchup(
    test_matchups, model, scaler, feature_cols,
    train_profiles_domestic, train_profiles_intl, train_weights,
)

print("\n--- Side-by-side summary ---")
print(f"Win-rate control:  {control_acc:.2%}")
print(f"predict_matchup:   {predict_acc:.2%}")
print(f"Delta:             {predict_acc - control_acc:+.2%}")

sample_a, sample_b = "T1", "Cloud9"
intl_boost = 3.0 if INTERNATIONAL_MODE else 1.0
live_profiles = build_team_profiles(team_df, intl_boost=intl_boost)
live_weights = region_weights if INTERNATIONAL_MODE else None
win_rates = build_cumulative_win_rates(team_df)
control_prob = predict_winrate_matchup(sample_a, sample_b, win_rates)

print(f"\n--- Sample matchup: {sample_a} vs {sample_b} ---")
print(f"Win-rate control: {sample_a} {control_prob:.2%} | {sample_b} {1 - control_prob:.2%}")
predict_matchup(
    sample_a, sample_b, live_profiles, model, scaler, feature_cols,
    live_weights, verbose=True,
)
