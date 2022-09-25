import pandas as pd
import numpy as np
import plotly.express as px

def generate_H2H(row):
    h2h = [row['Loser'],row['Winner']]
    h2h = sorted(h2h)
    return str(h2h[0])+ "_"+ str(h2h[1])

def data_cleaner():

    tennis = pd.read_csv('tennis_data.csv',low_memory=False)
    columns_to_keep = ['ATP','Date','Round','Tournament','Series','Court','Surface','Winner','Loser','WRank','LRank','Wsets','Lsets','Best of']

    tennis_clean = tennis[columns_to_keep]
    tennis_clean = tennis_clean[~tennis_clean['Series'].isin(['International','International Gold','Masters','Masters Cup'])]

    series_ranking_dict = {'ATP250':250,'ATP500':500,'Masters1000':1000,'Grand Slam':2000}
    series_ranking = pd.DataFrame.from_dict(series_ranking_dict,orient='index',columns=['SeriesPoints']).reset_index().rename(columns={'index': 'Series'})
    tennis_clean = tennis_clean.merge(series_ranking,on='Series',how='inner')

    rounds_ranking = {'1st Round':1,'2nd Round':2,'3rd Round':3,'4th Round':4,'Quarterfinals':5,'Semifinals':6,'The Final':7,'Round Robin':3}
    rounds_ranking = pd.DataFrame.from_dict(rounds_ranking,orient='index',columns=['RoundDraw']).reset_index().rename(columns={'index': 'Round'})

    tennis_clean = tennis_clean.merge(rounds_ranking,on='Round',how='inner')
    tennis_clean['Type'] = tennis_clean['Court'] +"_"+ tennis_clean['Surface']
    tennis_clean = tennis_clean.drop(['Series','Round'],axis=1)
    # tennis_clean = tennis_clean.drop(['Series','Round','Surface','Court'],axis=1)

    tennis_clean['Date'] = pd.to_datetime(tennis_clean['Date'],format='%Y-%m-%d')
    tennis_clean['Year'] = tennis_clean['Date'].dt.year

    tennis_clean['H2H'] = tennis_clean.apply(generate_H2H, axis=1)
    tennis_clean[tennis_clean['H2H']=='Djokovic N._Nadal R.']

    worst_rank = max(tennis_clean['LRank'].max(),tennis_clean['WRank'].max())
    tennis_clean['LRank'] = tennis_clean['LRank'].fillna(worst_rank)
    tennis_clean['WRank'] = tennis_clean['WRank'].fillna(worst_rank)

    tennis_clean['QualityWin'] = tennis_clean['Wsets']-tennis['Lsets']
    tennis_clean['QualityWin'] = tennis_clean['QualityWin'].fillna(1)

    winners = set(tennis_clean['Winner'].drop_duplicates())
    losers = set(tennis_clean['Loser'].drop_duplicates())

    players = sorted(list(winners.union(losers)))

    return tennis_clean, players


if __name__=='__main__':
    df,players_list = data_cleaner()