import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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
    
    # choose featuers
    diff_cols = [
        'diff_golddiffat15', 'diff_xpdiffat15', 'diff_csdiffat15',
        'diff_golddiffat25', 'diff_killdiffat25', 'diff_gamelength', 'diff_ckpm'
    ]
    objective_cols = ['firstdragon_blue', 'firstbaron_blue', 'firsttower_blue']

    feature_cols = diff_cols + objective_cols

    # model learns how to use x axis to predict y axis
    x_axis = matchups[feature_cols]
    y_axis = matchups['blue_win']

    # split data into train 80% train and 20% test data
    x_train, x_test, y_train, y_test = train_test_split(
        x_axis, y_axis, test_size = 0.2, random_state = 42
    )

    # scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    # build model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

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

def predict_matchup(team_a, team_b, team_profiles, model, scaler, feature_cols, region_weights = None, n_simulations=10000):
    
    # get each team's profile
    a = team_profiles[team_profiles['teamname'] == team_a].iloc[0]
    b = team_profiles[team_profiles['teamname'] == team_b].iloc[0]

    # map from feature col name to the underlying stat name
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

    # simulate game n times to get win probability
    win_count = 0

    for _ in range(n_simulations):
        matchup_features = []

        # simulate differential features
        for col in feature_cols:
            if col in stat_map:
                stat = stat_map[col]
                a_val = np.random.normal(a[stat], a[f'{stat}_std'])
                b_val = np.random.normal(b[stat], b[f'{stat}_std'])
                matchup_features.append(a_val - b_val)

            # predict objectives with Bradley-Terry normalization
            elif col in [f'{s}_blue' for s in objective_stats]:
                stat = col.replace('_blue', '')
                a_rate = a[stat] * a_weight
                b_rate = b[stat] * b_weight
                prob_a_gets_it = a_rate / (a_rate + b_rate)
                matchup_features.append(prob_a_gets_it)

        features_array = np.array(matchup_features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        win_prob = model.predict_proba(features_scaled)[0][1]
        win_count += win_prob

    avg_win_prob = win_count / n_simulations

    print(f"\n{team_a} vs {team_b}")
    print(f"  {team_a} ({a['region']}, weight: {a_weight:.3f}): {avg_win_prob:.2%}")
    print(f"  {team_b} ({b['region']}, weight: {b_weight:.3f}): {1 - avg_win_prob:.2%}")

    return avg_win_prob