import pandas as pd
import numpy as np

INTL_LEAGUES = {'WLDs', 'MSI', 'EWC', 'FST', 'Asia Master', 'IC'}


def is_international_league(league):
    return league in INTL_LEAGUES


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

def build_team_df(filenames):
    '''
    filters and builds team_df from OraclesElixer csv

    Args:
        filenames (str | list[str]): path(s) to csv file(s)
    
    Returns:
        (dataframe) filtered team stats from data
    '''
    if isinstance(filenames, str):
        filenames = [filenames]

    frames = []
    for filename in filenames:
        df = pd.read_csv(filename)
        frames.append(df[df['position'] == 'team'].copy())

    team_df = pd.concat(frames, ignore_index=True)
    team_df = team_df.drop_duplicates(subset=['gameid', 'side'], keep='last')

    # predictors
    features = [
        'gameid', 'date', 'teamname', 'result', 'league',
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'firstdragon', 'firstbaron', 'firsttower',
        'golddiffat25', 'killsat25', 'opp_killsat25',  # for kill diff at 25
        'gamelength', 'ckpm', 'side'
    ]
    
    # filter to just desired features
    team_df = team_df[features]

    team_df['killdiffat25'] = team_df['killsat25'] - team_df['opp_killsat25']
    
    return team_df

def _weighted_agg(team_df, agg_features, weight_col):
    '''
    Compute weighted mean and std per team for each feature in agg_features,
    plus weighted win rate. Handles NaN values by excluding them per feature.

    Returns a DataFrame indexed by teamname with columns:
        <feat>, <feat>_std  for each feat in agg_features, plus 'result'.
    '''
    records = []
    for name, group in team_df.groupby('teamname'):
        w_all = group[weight_col]
        row = {'teamname': name}
        for f in agg_features:
            valid = group[f].notna()
            x = group.loc[valid, f]
            w = w_all[valid]
            w_sum = w.sum()
            if w_sum == 0 or len(x) == 0:
                row[f] = np.nan
                row[f'{f}_std'] = np.nan
            else:
                wm = (w * x).sum() / w_sum
                ws = np.sqrt(((w * (x - wm) ** 2).sum() / w_sum))
                row[f] = wm
                row[f'{f}_std'] = ws
        # weighted win rate
        w_sum_all = w_all.sum()
        row['result'] = (w_all * group['result']).sum() / w_sum_all if w_sum_all > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index('teamname')


def build_team_profiles(team_df, half_life_days=180, intl_boost=1.0):
    '''
    builds team profiles from team_df
    Args:
        team_df (dataframe): filtered team stats from data
        half_life_days (int): exponential decay half-life in days;
            games this many days old count half as much as the most recent game
        intl_boost (float): multiplier applied to the weight of international
            games (WLDs, MSI, EWC, etc.). Use >1.0 when predicting international
            matchups; keep at 1.0 for domestic predictions.
    
    Returns:
        (dataframe) of each team's recency-weighted average stats
    '''
    team_df = team_df.copy()

    # compute exponential decay weights so recent games matter more
    team_df['date'] = pd.to_datetime(team_df['date'])
    most_recent = team_df['date'].max()
    team_df['days_ago'] = (most_recent - team_df['date']).dt.days
    team_df['decay_weight'] = np.exp(-np.log(2) * team_df['days_ago'] / half_life_days)

    if intl_boost != 1.0:
        intl_mask = team_df['league'].isin(INTL_LEAGUES)
        team_df.loc[intl_mask, 'decay_weight'] *= intl_boost

    agg_features = [
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'firstdragon', 'firstbaron', 'firsttower',
        'golddiffat25', 'killdiffat25',
        'gamelength', 'ckpm'
    ]

    stats = _weighted_agg(team_df, agg_features, 'decay_weight')

    mean_cols = agg_features
    std_cols  = [f'{f}_std' for f in agg_features]
    mean_stats = stats[mean_cols]
    std_stats  = stats[std_cols]
    win_rate   = stats['result']

    # get home region not just league
    regional_df  = team_df[team_df['league'].isin(regional_league_map.keys())]
    team_region  = (
        regional_df
        .groupby('teamname')['league']
        .agg(lambda x: x.mode()[0])
        .map(regional_league_map)
    )

    team_profiles = mean_stats.join(std_stats).join(win_rate).join(team_region.rename('region'))
    team_profiles.reset_index(inplace=True)

    # fill NaNs with averages
    team_profiles['region'] = team_profiles['region'].fillna('Unknown')
    team_profiles.fillna(team_profiles.mean(numeric_only=True), inplace=True)

    return team_profiles

def build_region_weights(team_df, verbose=False):
    # map regional leagues to their region

    # get each team's home region from regional games only
    regional_df = team_df[team_df['league'].isin(regional_league_map.keys())]
    team_region = (
        regional_df
        .groupby('teamname')['league']
        .agg(lambda x: x.mode()[0])  # most common league for that team
        .map(regional_league_map)     # convert league to region
    )

    # filter to international games only
    intl_df = team_df[team_df['league'].isin(INTL_LEAGUES)].copy()

    if intl_df.empty:
        if verbose:
            print("warning: no international data found, using equal weights")
        return {}

    # attach home region to each international game
    intl_df['region'] = intl_df['teamname'].map(team_region)

    # drop teams we couldn't map to a region
    unmapped = intl_df[intl_df['region'].isna()]['teamname'].unique()
    if verbose and len(unmapped) > 0:
        print(f"warning: could not map these teams to a region: {unmapped}")
    intl_df = intl_df.dropna(subset=['region'])

    if verbose:
        print(f"Building region weights from {len(intl_df)} international games")

    # win rate per region at international events
    region_winrates = intl_df.groupby('region')['result'].mean()
    region_counts   = intl_df.groupby('region')['result'].count()

    if verbose:
        print("\nRaw international win rates:")
        for region in region_winrates.sort_values(ascending=False).index:
            print(f"  {region}: {region_winrates[region]:.3f} ({region_counts[region]} games)")

    # normalize around 1.0
    mean_wr = region_winrates.mean()
    region_weights = (region_winrates / mean_wr).to_dict()

    if verbose:
        print("\nNormalized region weights:")
        for region, weight in sorted(region_weights.items(), key=lambda x: -x[1]):
            print(f"  {region}: {weight:.3f}")

    return region_weights