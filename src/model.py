import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import is_international_league

DIFF_COLS = [
    'diff_golddiffat15', 'diff_xpdiffat15', 'diff_csdiffat15',
    'diff_golddiffat25', 'diff_killdiffat25', 'diff_gamelength', 'diff_ckpm',
    'diff_win_rate',
]
OBJECTIVE_COLS = ['firstdragon_blue', 'firstbaron_blue', 'firsttower_blue']


def _feature_cols():
    return DIFF_COLS + OBJECTIVE_COLS


def fit_model(matchups):
    '''
    Train logistic regression on all rows in matchups (no holdout split).
    '''
    feature_cols = _feature_cols()
    x_axis = matchups[feature_cols]
    y_axis = matchups['blue_win']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_axis)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_scaled, y_axis)

    return model, scaler, feature_cols


def train_model(matchups):
    '''
    builds and trains a logistic regression model, 
    classifying games by winner

    Args:
        matchups (df) containing filtered data of game details
    
    Returns:
        model (LogisticRegression) fitted to matchup features
        scaler (standardScalar) to scale train data
        feature_cols (list) of influential variables
        x_test_scaled (list) of scaled partition of feature_cols
        y_test (list) of partition of matchups results for test 
    '''
    feature_cols = _feature_cols()
    x_axis = matchups[feature_cols]
    y_axis = matchups['blue_win']

    x_train, x_test, y_train, y_test = train_test_split(
        x_axis, y_axis, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model, scaler, feature_cols, x_test_scaled, y_test

def evaluate_model(model, x_test_scaled, y_test, feature_cols):
    '''
    evaluates the model and prints accuracy and 
    classification report

    Inputs:
        model (LogisticRegression) fitted to matchup features
        x_test_scaled (list) of scaled partition of feature_cols
        y_test (list) of partition of matchups results for test  
        feature_cols (list) of influential variables
    '''
    y_pred = model.predict(x_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))

    weights = pd.Series(model.coef_[0], index=feature_cols).sort_values()
    print("\nFeature weights:")
    print(weights)
    return

def _simulate_side(blue, red, blue_weight, red_weight, model, scaler, feature_cols, stat_map, objective_stats, n_simulations):
    '''
    Monte Carlo simulation treating `blue` as the blue-side team.
    Returns the average predicted win probability for the blue team.
    '''
    win_count = 0
    for _ in range(n_simulations):
        matchup_features = []

        for col in feature_cols:
            if col in stat_map:
                stat = stat_map[col]
                blue_val = np.random.normal(blue[stat], blue[f'{stat}_std']) * blue_weight
                red_val  = np.random.normal(red[stat],  red[f'{stat}_std'])  * red_weight
                matchup_features.append(blue_val - red_val)

            elif col in [f'{s}_blue' for s in objective_stats]:
                stat = col.replace('_blue', '')
                blue_rate = blue[stat] * blue_weight
                red_rate  = red[stat]  * red_weight
                total = blue_rate + red_rate
                prob_blue_gets_it = blue_rate / total if total > 0 else 0.5
                matchup_features.append(prob_blue_gets_it)

            elif col == 'diff_win_rate':
                matchup_features.append(blue['result'] - red['result'])

        features_array  = np.nan_to_num(
            np.array(matchup_features, dtype=float), nan=0.0
        ).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        win_prob = model.predict_proba(features_scaled)[0][1]
        win_count += win_prob

    return win_count / n_simulations


def predict_matchup(team_a, team_b, team_profiles, model, scaler, feature_cols, region_weights=None, n_simulations=10000, verbose=False):

    # get each team's profile
    a = team_profiles[team_profiles['teamname'] == team_a].iloc[0]
    b = team_profiles[team_profiles['teamname'] == team_b].iloc[0]

    stat_map = {
        'diff_golddiffat15': 'golddiffat15',
        'diff_xpdiffat15':   'xpdiffat15',
        'diff_csdiffat15':   'csdiffat15',
        'diff_golddiffat25': 'golddiffat25',
        'diff_killdiffat25': 'killdiffat25',
        'diff_gamelength':   'gamelength',
        'diff_ckpm':         'ckpm',
    }
    objective_stats = ['firstdragon', 'firstbaron', 'firsttower']

    # get region weights, default to 1.0 if not found
    a_weight = 1.0
    b_weight = 1.0
    if region_weights:
        a_weight = region_weights.get(a['region'], 1.0)
        b_weight = region_weights.get(b['region'], 1.0)

    # run simulations from both side assignments and average to remove blue-side bias
    prob_a_as_blue = _simulate_side(a, b, a_weight, b_weight, model, scaler, feature_cols, stat_map, objective_stats, n_simulations)
    prob_a_as_red  = 1 - _simulate_side(b, a, b_weight, a_weight, model, scaler, feature_cols, stat_map, objective_stats, n_simulations)

    avg_win_prob = (prob_a_as_blue + prob_a_as_red) / 2

    if verbose:
        print(f"\n{team_a} vs {team_b}")
        print(f"  {team_a} ({a['region']}, weight: {a_weight:.3f}): {avg_win_prob:.2%}")
        print(f"  {team_b} ({b['region']}, weight: {b_weight:.3f}): {1 - avg_win_prob:.2%}")

    return avg_win_prob


def evaluate_predict_matchup(
    test_matchups,
    model,
    scaler,
    feature_cols,
    profiles_domestic,
    profiles_intl,
    region_weights,
    n_simulations=500,
):
    '''
    Backtest predict_matchup on holdout games using train-only profiles.
    Applies intl_boost profiles and region_weights only for international leagues.
    '''
    known_teams = set(profiles_domestic['teamname'])
    y_true = []
    y_pred = []
    intl_true, intl_pred = [], []
    dom_true, dom_pred = [], []
    missing = 0

    for _, row in test_matchups.iterrows():
        blue = row['teamname_blue']
        red = row['teamname_red']
        actual = int(row['blue_win'])

        if blue not in known_teams or red not in known_teams:
            missing += 1
            prob_blue = 0.5
        elif is_international_league(row['league']):
            prob_blue = predict_matchup(
                blue, red, profiles_intl, model, scaler, feature_cols,
                region_weights, n_simulations=n_simulations,
            )
        else:
            prob_blue = predict_matchup(
                blue, red, profiles_domestic, model, scaler, feature_cols,
                None, n_simulations=n_simulations,
            )

        predicted = int(prob_blue >= 0.5)
        y_true.append(actual)
        y_pred.append(predicted)

        if is_international_league(row['league']):
            intl_true.append(actual)
            intl_pred.append(predicted)
        else:
            dom_true.append(actual)
            dom_pred.append(predicted)

    accuracy = accuracy_score(y_true, y_pred)

    print("\n--- predict_matchup ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_true, y_pred))

    if intl_true:
        print(f"International games: {accuracy_score(intl_true, intl_pred):.2%} ({len(intl_true)} games)")
    if dom_true:
        print(f"Domestic games:      {accuracy_score(dom_true, dom_pred):.2%} ({len(dom_true)} games)")
    if missing:
        print(f"Teams missing from train profiles: {missing} games (defaulted to 0.5)")

    return accuracy