import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta,datetime
import glob

def generate_H2H(row):
    h2h = [row['loser_id'],row['winner_id']]
    h2h = sorted(h2h)
    return str(h2h[0])+ "_"+ str(h2h[1])

def calculate_score(row):
    parcials = row['score'].split(" ")

    if 'W/O' in parcials or 'RET' in parcials or 'DEF' in parcials:
        if row['best_of']==3:
            winner = 2
        else:
            winner = 3
        loser = 0
        return winner,loser
    
    winner = 0
    loser = 0
    for p in parcials:
        a,b = p.split("-")
        if "(" in b:
            b = b.split("(")[0]

        if int(a)>int(b):
            winner+=1
        else:
            loser+=1

    return winner,loser

def data_import():
    cutoff_date = pd.Timestamp(2010,1,1)

    matches = pd.DataFrame()
    rankings = pd.DataFrame()

    # read matches
    data_folder = glob.glob("tennis_atp/atp_matches_20*.csv")
    for file in data_folder:
        t = pd.read_csv(file,low_memory=False)
        matches = pd.concat([matches,t])

    columns_to_keep = ['tourney_date','tourney_name','tourney_level','draw_size','surface','best_of','round','score',
                       'winner_id','winner_name','winner_age','winner_rank',
                       'loser_id','loser_name','loser_age','loser_rank']

    # clean matches
    matches_clean =  matches[columns_to_keep]

    matches_clean = matches_clean[~matches_clean['tourney_level'].isin(['J','E','T','S','C','D'])]

    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('Olympics')]
    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('Laver Cup')]
    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('United Cup')]

    matches_clean['tourney_date'] = pd.to_datetime(matches_clean['tourney_date'],format='%Y%m%d')

    series_ranking_dict = {'A':250,'M':1000,'G':2000}
    matches_clean['tourney_points'] = matches_clean['tourney_level'].map(series_ranking_dict)

    rounds_ranking_dict = {'RR':7, 'R128':6, 'R64':5, 'R32':4, 'R16':3, 'QF':2, 'SF':1, 'F':0}
    matches_clean['round_level'] = matches_clean['round'].map(rounds_ranking_dict)

    matches_clean['H2H'] = matches_clean.apply(generate_H2H, axis=1)
    # matches_clean['score'] = matches_clean['score'].str.upper()
    # matches_clean['score'] = matches_clean['score'].str.replace("."," ")
    # matches_clean[['winner_sets','loser_sets']]= matches_clean.apply(calculate_score, axis=1, result_type ='expand')
    # matches_clean['quality_win'] = matches_clean['winner_sets']-matches_clean['loser_sets']
    # matches_clean['quality_win'] = matches_clean['quality_win'].fillna(1)
    matches_clean = matches_clean.dropna()
    matches_clean = matches_clean[matches_clean['tourney_date']>=cutoff_date]

    matches_clean['tourney_year'] = matches_clean['tourney_date'].dt.year
    matches_clean['tourney_date'] = matches_clean['tourney_date'].dt.date

    # read players
    players = pd.read_csv('tennis_atp/atp_players.csv')
    
    # filter by the players appearing on matches cleaned
    winners = set(matches_clean['winner_id'].unique())
    losers = set(matches_clean['loser_id'].unique())
    players_to_keep = list(winners.union(losers))

    players = players[players['player_id'].isin(players_to_keep)]
    players['dob'] = pd.to_datetime(players['dob'],format='%Y%m%d')
    players['name'] = players['name_first'] + " " + players['name_last']

    # read rankings
    data_folder = glob.glob("tennis_atp/atp_rankings_*.csv")
    for file in data_folder:
        t = pd.read_csv(file,low_memory=False)
        rankings = pd.concat([rankings,t])

    rankings['ranking_date'] = pd.to_datetime(rankings['ranking_date'],format='%Y%m%d')

    rankings = rankings[rankings['ranking_date']>=cutoff_date]
    rankings['ranking_date'] = rankings['ranking_date'].dt.date
    rankings = rankings.merge(players[['name','player_id']],left_on='player',right_on='player_id')

    return matches_clean, rankings, players

def get_more_info(matches,rankings,players):
    tournaments = matches.sort_values(by='tourney_date',ascending=False).drop_duplicates(subset=['tourney_name','surface','tourney_points'])

    tournaments_dict= {}
    for i,row in tournaments.iterrows():
        tournaments_dict[row['tourney_name']] = [row['surface'],row['tourney_points'],row['tourney_date']]

    players_dict = {}
    for i, row in players.iterrows():
        ranking_player = rankings[rankings['player'] == row['player_id']]
        last_available_rank = ranking_player['ranking_date'].max()
        last_rank = ranking_player[ranking_player['ranking_date']==last_available_rank]['rank'].iloc[0]
        players_dict[row['name']] = [row['player_id'],last_rank]

    rounds = list(matches['round_level'].unique())

    return players_dict, tournaments_dict, rounds

if __name__=='__main__':
    matches, rankings, players = data_import()
    print(matches.info())
    print(matches.head(5))
    print(rankings.head(5))
    print(players.head(5))
    players_dict, tournaments_dict, rounds = get_more_info(matches,rankings,players)
    print(tournaments_dict)
    print(tournaments_dict.keys())
    tournament_surface = tournaments_dict['Roland Garros'][1]
    print(tournament_surface)
    print(players_dict['Roger Federer'])
    df_aux = matches[matches['winner_id']==players_dict['Roger Federer']]    
    print(df_aux)

