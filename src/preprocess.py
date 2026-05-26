import pandas as pd

regional_league_map = {
    'LCK':   'LCK',
    'LCKC':  'LCK',
    'LPL':   'LPL',
    'LEC':   'LEC',
    'NLC':   'LEC',
    'LCS':   'LCS',
    'LTA N': 'LTA',
    'LTA S': 'LTA',
    'LTA':   'LTA',
    'PCS':   'PCS',
    'VCS':   'VCS',
    'LJL':   'LJL',
    'CBLOL': 'CBLOL',
    'TCL':   'TCL',
}

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
        'gameid', 'teamname', 'result', 'league',
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

    # get home region not just league
    regional_df  = team_df[team_df['league'].isin(regional_league_map.keys())]
    team_region  = (
        regional_df
        .groupby('teamname')['league']
        .agg(lambda x: x.mode()[0])
        .map(regional_league_map)
    )

    # rename columns
    std_stats.columns = [f'{c}_std' for c in std_stats.columns]

    team_profiles = mean_stats.join(std_stats).join(win_rate).join(team_region.rename('region'))
    team_profiles.reset_index(inplace=True)

    # fill NaNs with averages
    team_profiles['region'] = team_profiles['region'].fillna('Unknown')
    team_profiles.fillna(team_profiles.mean(numeric_only=True), inplace=True)

    # print(team_profiles.shape)       # (num_teams, ~21 columns)
    # print(team_profiles.head())
    # print(team_profiles.isnull().sum())  # check for NaNs
    
    return team_profiles

def build_region_weights(team_df):
    # map regional leagues to their region

    intl_leagues = ['WLDs', 'MSI', 'EWC', 'FST', 'Asia Master', 'IC']

    # get each team's home region from regional games only
    regional_df = team_df[team_df['league'].isin(regional_league_map.keys())]
    team_region = (
        regional_df
        .groupby('teamname')['league']
        .agg(lambda x: x.mode()[0])  # most common league for that team
        .map(regional_league_map)     # convert league to region
    )

    # filter to international games only
    intl_df = team_df[team_df['league'].isin(intl_leagues)].copy()

    if intl_df.empty:
        print("warning: no international data found, using equal weights")
        return {}

    # attach home region to each international game
    intl_df['region'] = intl_df['teamname'].map(team_region)

    # drop teams we couldn't map to a region
    unmapped = intl_df[intl_df['region'].isna()]['teamname'].unique()
    if len(unmapped) > 0:
        print(f"warning: could not map these teams to a region: {unmapped}")
    intl_df = intl_df.dropna(subset=['region'])

    print(f"Building region weights from {len(intl_df)} international games")

    # win rate per region at international events
    region_winrates = intl_df.groupby('region')['result'].mean()
    region_counts   = intl_df.groupby('region')['result'].count()

    print("\nRaw international win rates:")
    for region in region_winrates.sort_values(ascending=False).index:
        print(f"  {region}: {region_winrates[region]:.3f} ({region_counts[region]} games)")

    # normalize around 1.0
    mean_wr = region_winrates.mean()
    region_weights = (region_winrates / mean_wr).to_dict()

    print("\nNormalized region weights:")
    for region, weight in sorted(region_weights.items(), key=lambda x: -x[1]):
        print(f"  {region}: {weight:.3f}")

    return region_weights