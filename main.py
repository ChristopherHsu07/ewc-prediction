import warnings
from sklearn.metrics import accuracy_score

from src.preprocess import build_team_df, build_team_profiles, build_region_weights
from src.features import build_matchups
from src.model import train_model, evaluate_model, predict_matchup
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

INTERNATIONAL_MODE = True  # set to False for domestic predictions

team_profiles = build_team_profiles(team_df, intl_boost=3.0 if INTERNATIONAL_MODE else 1.0)
matchups = build_matchups(team_df)

pregame = compute_pregame_win_rates(team_df)
matchups = attach_pregame_win_rates(matchups, pregame)

region_weights = build_region_weights(team_df, verbose=True)

model, scaler, feature_cols, x_test_scaled, y_test = train_model(matchups)

control_acc = evaluate_winrate_control(matchups)

print("\n--- Full model ---")
evaluate_model(model, x_test_scaled, y_test, feature_cols)
main_acc = accuracy_score(y_test, model.predict(x_test_scaled))

print("\n--- Side-by-side summary ---")
print(f"Win-rate control: {control_acc:.2%}")
print(f"Full model:       {main_acc:.2%}")
print(f"Delta:            {main_acc - control_acc:+.2%}")

sample_a, sample_b = "T1", "Cloud9"
win_rates = build_cumulative_win_rates(team_df)
control_prob = predict_winrate_matchup(sample_a, sample_b, win_rates)

print(f"\n--- Sample matchup: {sample_a} vs {sample_b} ---")
print(f"Win-rate control: {sample_a} {control_prob:.2%} | {sample_b} {1 - control_prob:.2%}")
predict_matchup(
    sample_a, sample_b, team_profiles, model, scaler, feature_cols, region_weights, verbose=True
)
