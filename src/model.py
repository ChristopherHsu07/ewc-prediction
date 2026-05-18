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