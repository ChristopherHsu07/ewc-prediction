import pandas as pd

def build_team_df(filename):
    '''
    filters and builds team_df from OraclesElixer csv

    Args:
        filename (str): name of csv file
    
    Returns:
        (dataframe) filtered team stats from data
    '''
    df = pd.read_csv(filename)

    #filter to only team performance
    team_df = df[df['position'] == 'team'].copy()

    # predictors
    features = [
        'gameid', 'teamname', 'result',
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'firstdragon', 'firstbaron', 'firsttower',
        'golddiffat25', 'killsat25', 'opp_killsat25',  # for kill diff at 25
        'gamelength', 'ckpm', 'side'
    ]
    
    # filter to just desired features
    team_df = team_df[features]

    team_df['killdiffat25'] = team_df['killsat25'] - team_df['opp_killsat25']
    
    return team_df

def build_team_profiles(team_df):
    '''
    builds team profiles from team_df
    Args:
        team_df (dataframe): filtered team stats from data
    
    Returns:
        (dataframe) of each team's average stats
    '''

    #aggregate features to go from row = game --> row = team
    agg_features = [
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'firstdragon', 'firstbaron', 'firsttower',
        'golddiffat25', 'killdiffat25',
        'gamelength', 'ckpm'
    ]

    mean_stats = team_df.groupby('teamname')[agg_features].mean()
    std_stats = team_df.groupby('teamname')[agg_features].std()
    win_rate = team_df.groupby('teamname')['result'].mean()

    # rename columns
    std_stats.columns = [f'{c}_std' for c in std_stats.columns]

    team_profiles = mean_stats.join(std_stats).join(win_rate)
    team_profiles.reset_index(inplace=True)

    # fill NaNs with averages
    team_profiles.fillna(team_profiles.mean(numeric_only=True), inplace=True)

    # print(team_profiles.shape)       # (num_teams, ~21 columns)
    # print(team_profiles.head())
    # print(team_profiles.isnull().sum())  # check for NaNs
    
    return team_profiles