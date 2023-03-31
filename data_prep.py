import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
import glob

def generate_H2H(row):
    h2h = [row['Loser'],row['Winner']]
    h2h = sorted(h2h)
    return str(h2h[0])+ "_"+ str(h2h[1])

def data_cleaner():
    tennis = pd.DataFrame()
    columns_to_keep = ['ATP','Date','Round','Tournament','Series','Court','Surface','Winner','Loser','WRank','LRank','Wsets','Lsets','Best of']

    data_folder = glob.glob("data/*.csv")
    for file in data_folder:
        t = pd.read_csv(file,low_memory=False)
        t = t[columns_to_keep]
        tennis = pd.concat([tennis,t])

    tennis_clean = tennis.dropna()
    tennis_clean = tennis_clean[~tennis_clean['Series'].isin(['International','International Gold','Masters','Masters Cup'])]

    series_ranking_dict = {'ATP250':250,'ATP500':500,'Masters 1000':1000,'Grand Slam':2000}
    tennis_clean['SeriesPoints'] = tennis_clean['Series'].map(series_ranking_dict)

    rounds_ranking_dict = {'1st Round':1,'2nd Round':2,'3rd Round':3,'4th Round':4,'Quarterfinals':5,'Semifinals':6,'The Final':7,'Round Robin':3}
    tennis_clean['RoundDraw'] = tennis_clean['Round'].map(rounds_ranking_dict)

    tennis_clean['Type'] = tennis_clean['Court'] +"_"+ tennis_clean['Surface']
    tennis_clean = tennis_clean.drop(['Series','Round'],axis=1)

    tennis_clean['Date'] = pd.to_datetime(tennis_clean['Date'],format='%Y-%m-%d')
    tennis_clean['Year'] = tennis_clean['Date'].dt.year

    tennis_clean['H2H'] = tennis_clean.apply(generate_H2H, axis=1)

    tennis_clean['QualityWin'] = tennis_clean['Wsets']-tennis_clean['Lsets']
    tennis_clean['QualityWin'] = tennis_clean['QualityWin'].fillna(1)
    tennis_clean['Date'] = tennis_clean['Date'].dt.date

    return tennis_clean

def get_more_info(df):
    winners = set(df['Winner'].drop_duplicates())
    losers = set(df['Loser'].drop_duplicates())

    players = sorted(list(winners.union(losers)))

    tournaments = df[df['Year']>=2020]
    tournaments = tournaments[['Tournament','SeriesPoints','Surface']].drop_duplicates().sort_values(by=['SeriesPoints','Tournament','Surface'],ascending=[False,True,False])
    tournaments_max_date = df.groupby(['Tournament'])['Date'].max().reset_index()
    tournaments_max_date['NextDate'] = tournaments_max_date['Date'] + timedelta(days=365)

    tournaments_full = tournaments.merge(tournaments_max_date,on='Tournament',how='inner')

    rounds = sorted(list(set(df['RoundDraw'].drop_duplicates())))
    tournaments_dict= {}
    for i,row in tournaments_full.iterrows():
        tournaments_dict[row['Tournament']] = [row['Surface'],row['SeriesPoints'],row['NextDate']]

    return players, tournaments_dict, rounds

if __name__=='__main__':
    df = data_cleaner()
    players_list,tournaments_list,rounds = get_more_info(df)
    print(tournaments_list)
    print(tournaments_list.keys())
    tournament_surface = tournaments_list['French Open'][1]
    print(tournament_surface)
    df_aux = df[df['Winner']=='Federer R.']    
    # df_aux = df_aux.sort_values(by='Date',ascending=False)
    print(df_aux.head(3))
    # print(df_aux['WRank'].iloc[0])
    print(df_aux['Date'].max())
    print(df_aux['WRank'].loc[df_aux['Date'] == df_aux['Date'].max()].iloc[0])

