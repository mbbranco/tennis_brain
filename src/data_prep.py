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
    if 'W/O' in parcials or 'RET' in parcials or 'DEF' in parcials or 'DEF.' in parcials:
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

    columns_to_keep = ['tourney_date','tourney_name','tourney_level','draw_size','surface','best_of','round',
                       'score',
                       'winner_id','winner_name','winner_age','winner_rank',
                       'loser_id','loser_name','loser_age','loser_rank']


    # clean matches
    matches_clean =  matches[columns_to_keep]
    matches_clean = matches_clean[~matches_clean['tourney_level'].isin(['C','S','F','D','P','PM','I','E','J','T'])]

    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('Olympics')]
    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('Cup')]
    matches_clean = matches_clean[~matches_clean['tourney_name'].str.contains('Finals')]

    matches_clean['tourney_date'] = pd.to_datetime(matches_clean['tourney_date'],format='%Y%m%d')
    
    # TODO: numero de pontos de cada torneio diferenciar - 500 e 250
     
    series_ranking_dict = {'A':250,'M':1000,'G':2000}
    matches_clean['tourney_points'] = matches_clean['tourney_level'].map(series_ranking_dict)

    rounds_ranking_dict = {'RR':7, 'R128':6, 'R64':5, 'R32':4, 'R16':3, 'QF':2, 'SF':1, 'F':0}
    matches_clean['round_level'] = matches_clean['round'].map(rounds_ranking_dict)

    matches_clean['score'] = matches_clean['score'].str.upper().replace("."," ")
    matches_clean[['winner_sets','loser_sets']]= matches_clean.apply(calculate_score, axis=1, result_type ='expand')
    matches_clean['score_quality'] = matches_clean['winner_sets']-matches_clean['loser_sets']

    matches_clean = matches_clean[matches_clean['tourney_date']>=cutoff_date]

    matches_clean['tourney_year'] = matches_clean['tourney_date'].dt.year
    matches_clean['tourney_date'] = matches_clean['tourney_date'].dt.date

    matches_clean = matches_clean.dropna()
    matches_clean.index = pd.RangeIndex(start=0, stop=len(matches_clean), step=1)
    matches_clean = matches_clean.reset_index().rename(columns={'index':'match_id'})

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

    print('Data Import Complete!')
    return matches_clean, rankings, players

    
# def rolling_averages(matches,players_dict):
#     # min_date = matches['tourney_date'].min()
#     # max_date = matches['tourney_date'].max()
#     # dates = pd.DataFrame(pd.date_range(min_date,max_date),columns=['tourney_date'])
#     # dates['tourney_date'] = dates['tourney_date'].dt.date
    
#     # win_ratio_rolling
#     df_ratios = pd.DataFrame()

#     for player_name,player_id in players_dict.items():
#         df_p1 = matches[(matches['winner_id']==player_id) | (matches['loser_id']==player_id)]
#         df_p1 = df_p1.reset_index().sort_values(by=['tourney_date'])

#         df_p1['win'] = np.where(df_p1['winner_id']==player_id,1,0)
#         df_p1['loss'] = np.where(df_p1['loser_id']==player_id,1,0)
#         df_p1['win_cum'] = df_p1['win'].cumsum()
#         df_p1['loss_cum'] = df_p1['loss'].cumsum()

#         df_p1['win_loss_ratio'] = round(df_p1['win_cum']/(df_p1['win_cum']+df_p1['loss_cum']),2)
#         df_p1['player_id'] = player_id
#         df_p1['player_name'] = player_name

#         df_p1 = df_p1[['tourney_date','player_id','player_name','win_loss_ratio']].copy()

#         # df_p1 = dates.merge(df_p1,on='tourney_date',how='left')
#         # df_p1 = df_p1.ffill()
        
#         df_ratios = pd.concat([df_p1,df_ratios])
        
#     print('Rolling Ratio Complete!')

#     return df_ratios

def get_last_rank(player_id,rankings,date=None):
    ranking_player = rankings[rankings['player'] == player_id]

    if date is None:
        last_available_rank = ranking_player['ranking_date'].max()
    else:
        last_available_rank = ranking_player[ranking_player['ranking_date']<=date]
        last_available_rank = last_available_rank['ranking_date'].max()

    last_rank = ranking_player[ranking_player['ranking_date']==last_available_rank]['rank']
    if last_rank.shape[0]!=0:
        last_rank = last_rank.iloc[0]
    else:
        last_rank = 'N/A'

    return last_rank

def get_player_id_by_name(name,players):
    for k, v in players.items():
        if v['name']==name:
            return k
        
def get_players_info(players,rankings):
    players_dict = {}
    for i,row in players.iterrows():
        player_id = row['player_id']
        players_dict[player_id] = row.to_dict()

    return players_dict

def get_tournaments_info(matches):
    tournaments = matches[['tourney_date','tourney_name','surface','tourney_points','best_of','draw_size']].sort_values(by='tourney_date',ascending=True)
    tournaments = tournaments.drop_duplicates(subset=['tourney_name','surface','tourney_points'])

    tournaments_dict= {}
    for name,row in tournaments.groupby('tourney_name'):
        dates = row['tourney_date'].to_list()
        tournaments_dict[name] = row[['surface','tourney_points','best_of','draw_size','tourney_name']].iloc[0].to_dict()
        tournaments_dict[name].update({'dates':dates})
    
    return tournaments_dict

def get_kpis(matches,players):
    ratios = pd.DataFrame()
    for i,row in players.iterrows():
        player_id = row['player_id']
        # win_ratio_rolling
        df_p1 = matches[(matches['winner_id']==player_id) | (matches['loser_id']==player_id)]
        df_p1 = df_p1.sort_values(by=['tourney_date'])
        df_p1 = df_p1[['match_id','tourney_date','winner_id','loser_id']].copy()

        df_p1['win'] = np.where(df_p1['winner_id']==player_id,1,0)
        df_p1['loss'] = np.where(df_p1['loser_id']==player_id,1,0)
        df_p1['win_cum'] = df_p1['win'].cumsum()
        df_p1['loss_cum'] = df_p1['loss'].cumsum()
        
        df_p1['win_loss_ratio'] = np.where(df_p1['loss_cum']==0,np.nan,round(df_p1['win_cum']/(df_p1['loss_cum']),2))
        df_p1['win_perc'] = np.where((df_p1['win_cum']+df_p1['loss_cum'])==0,np.nan,round(df_p1['win_cum']/(df_p1['win_cum']+df_p1['loss_cum']),2))
        
        # df_p1['win_cum_roll'] = df_p1['win'].rolling(min_periods=1, window=11).sum()
        # df_p1['loss_cum_roll'] = df_p1['loss'].rolling(min_periods=1, window=11).sum()

        # df_p1['win_loss_ratio_roll'] = np.where(df_p1['loss_cum_roll']==0,np.nan,
        #                                         round(df_p1['win_cum_roll']/(df_p1['loss_cum_roll']),2))
        
        # df_p1['win_perc_roll'] = np.where((df_p1['win_cum_roll']+df_p1['loss_cum_roll'])==0,np.nan,
        #                                   round(df_p1['win_cum_roll']/(df_p1['win_cum_roll']+df_p1['loss_cum_roll']),2))

        df_p1['player_id'] = [player_id]*df_p1.shape[0]
        df_p1 = df_p1.drop(columns=['winner_id','loser_id'])
        ratios = pd.concat([df_p1,ratios])

    rounds = list(matches['round_level'].unique())

    print('More Info Complete!')

    return ratios, rounds

if __name__=='__main__':
    matches, rankings, players = data_import()
    tournaments_dict = get_tournaments_info(matches)
    players_dict = get_players_info(players,rankings)
    df_ratios,rounds = get_kpis(matches,players)

    id = get_player_id_by_name('Carlos Alcaraz',players)
    print(players_dict[id])
    print(df_ratios[df_ratios['player_id']==id])
    print(tournaments_dict['Australian Open']) 

    # print(tournaments_dict)
    # print(tournaments_dict.keys())
    # tournament_surface = tournaments_dict['Roland Garros'][1]
    # print(tournament_surface)
    # print(players_dict['Roger Federer'])
    # df_aux = matches[matches['winner_id']==players_dict['Roger Federer']]    
    # print(df_aux)

