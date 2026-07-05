import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

DEFAULT_WIN_RATE = 0.5


def compute_pregame_win_rates(team_df):
    '''
    Walk games chronologically and record each team's cumulative W/L
    before each game (no future leakage).

    Returns a DataFrame keyed by (gameid, teamname) with pre_wins,
    pre_games, and pre_win_rate columns.
    '''
    df = team_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'gameid', 'side'])

    records = []
    totals = {}

    for gameid, game_rows in df.groupby('gameid', sort=False):
        for _, row in game_rows.iterrows():
            team = row['teamname']
            stats = totals.get(team, {'wins': 0, 'games': 0})
            pre_games = stats['games']
            pre_wins = stats['wins']
            pre_win_rate = pre_wins / pre_games if pre_games > 0 else DEFAULT_WIN_RATE
            records.append({
                'gameid': gameid,
                'teamname': team,
                'pre_wins': pre_wins,
                'pre_games': pre_games,
                'pre_win_rate': pre_win_rate,
            })

        for _, row in game_rows.iterrows():
            team = row['teamname']
            if team not in totals:
                totals[team] = {'wins': 0, 'games': 0}
            totals[team]['games'] += 1
            totals[team]['wins'] += int(row['result'])

    return pd.DataFrame(records)


def attach_pregame_win_rates(matchups, pregame_df):
    '''
    Merge chronological pre-game win rates onto matchup rows.
    '''
    matchups = matchups.copy()

    blue_cols = pregame_df.rename(columns={
        'teamname': 'teamname_blue',
        'pre_win_rate': 'pre_win_rate_blue',
    })
    red_cols = pregame_df.rename(columns={
        'teamname': 'teamname_red',
        'pre_win_rate': 'pre_win_rate_red',
    })

    matchups = matchups.merge(
        blue_cols[['gameid', 'teamname_blue', 'pre_win_rate_blue']],
        on=['gameid', 'teamname_blue'],
        how='left',
    )
    matchups = matchups.merge(
        red_cols[['gameid', 'teamname_red', 'pre_win_rate_red']],
        on=['gameid', 'teamname_red'],
        how='left',
    )
    matchups['diff_pre_win_rate'] = matchups['pre_win_rate_blue'] - matchups['pre_win_rate_red']

    return matchups


def winrate_win_prob(blue_wr, red_wr):
    '''
    Map cumulative win rates to blue-side win probability.
    '''
    denominator = blue_wr + red_wr
    if denominator == 0:
        return DEFAULT_WIN_RATE
    return blue_wr / denominator


def build_cumulative_win_rates(team_df):
    '''
    Final cumulative W/L across all data for live matchup prediction.
    '''
    totals = team_df.groupby('teamname')['result'].agg(['sum', 'count'])
    win_rates = (totals['sum'] / totals['count']).to_dict()
    return win_rates


def predict_winrate_matchup(team_a, team_b, win_rates):
    '''
    Return P(team_a wins) using cumulative win rates only.
    '''
    wr_a = win_rates.get(team_a, DEFAULT_WIN_RATE)
    wr_b = win_rates.get(team_b, DEFAULT_WIN_RATE)
    return winrate_win_prob(wr_a, wr_b)


def evaluate_winrate_control(test_matchups):
    '''
    Evaluate the win-rate control on a pre-split holdout set.
    '''
    y_test = test_matchups['blue_win']
    y_pred = (test_matchups['pre_win_rate_blue'] > test_matchups['pre_win_rate_red']).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Win-rate control ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, y_pred))

    ties = (test_matchups['pre_win_rate_blue'] == test_matchups['pre_win_rate_red']).sum()
    if ties > 0:
        print(f"Tied pre-game win rates: {ties} games (predicted as blue loss)")

    return accuracy
