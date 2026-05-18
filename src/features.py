import pandas as pd

def build_matchups(team_df):
    '''
    train model on matchups: start by blue/red matchup for 
    each matchup and merge to differentiate
    
    Args: 
        team_df, filtered team data with desired features
    
    Returns: 
        matchups, differentials in features in each matchup
    '''
    team_df = team_df.copy()
    team_df['killdiffat25'] = team_df['killsat25'] - team_df['opp_killsat25']

    blue = team_df[team_df['side'] == 'Blue'].copy()
    red  = team_df[team_df['side'] == 'Red'].copy()

    matchups = blue.merge(red, on='gameid', suffixes=('_blue', '_red'))

    # differentiate features to compare opponents
    diff_features = [
    'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'golddiffat25', 'killdiffat25', 'gamelength', 'ckpm'
    ]
    for f in diff_features:
        matchups[f'diff_{f}'] = matchups[f'{f}_blue'] - matchups[f'{f}_red']
    
    matchups['blue_win'] = matchups['result_blue']
    matchups = matchups.dropna()

    return matchups