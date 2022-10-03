import plotly.express as px
import numpy as np
import pandas as pd

def win_loss_ratio(df,name_player):
    df_p1 = df[(df['H2H'].str.contains(name_player))].copy()
    df_p1['Win/Loss'] =  np.where(df_p1['Winner']==name_player,'Win','Loss')
    df_p1['Win/Loss_Val'] =  np.where(df_p1['Winner']==name_player,1,-1)
    df_p1 = df_p1.sort_values(by='Surface')

    plot = px.bar(data_frame=df_p1,x='Surface',y='Win/Loss_Val',color='Win/Loss',\
        color_discrete_map={
        'Win': 'green',
        'Loss': 'red'
        },\
        title=f"Win/Loss Ratio for {name_player}",\
        hover_data={'Tournament':True,'Date':True})

    return plot

def tournament_performance(df,name_p1):
    df_p1 = df[(df['H2H'].str.contains(name_p1))].copy()
    df_p1 = df_p1.groupby(['Year','Tournament','SeriesPoints','Surface','Winner']).agg({'RoundDraw':'max','Date':'min'}).reset_index()
    df_p1 = df_p1.sort_values(by='Date')
    df_p1['FinalWinner'] = np.where((df_p1['RoundDraw']==7)&(df_p1['Winner']==name_p1),'Winner','')
    df_p1['FinalWinner'] = np.where((df_p1['RoundDraw']==7)&(df_p1['Winner']!=name_p1),'Runner-Up',df_p1['FinalWinner'])

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}
    plot = px.bar(data_frame=df_p1,x='Date',y='RoundDraw',color='Surface',title=f"Tournament Performance for {name_p1}",text='FinalWinner',\
            hover_data={'Tournament':True,'SeriesPoints':True,'Winner':True},color_discrete_map=color_mapping)

    plot.update_xaxes(type='category')
    plot.update_xaxes(categoryorder='category ascending') 
    plot.add_hline(y=7, line_width=3, line_dash="dash", line_color="gold")
    
    return plot


def rank_evol(df,name_p1,name_p2):
    df_p1 = df[(df['H2H'].str.contains(name_p1))].copy()
    df_p2 = df[(df['H2H'].str.contains(name_p2))].copy()

    df_p1['MainPlayer'] = np.where(df_p1['Winner']==name_p1,df_p1['Winner'],df_p1['Loser'])
    df_p1['Win/Loss'] = np.where(df_p1['Winner']==name_p1,'Win','Loss')
    df_p1['Rank'] = np.where((df_p1['Winner']==name_p1),df_p1['WRank'],df_p1['LRank'])

    df_p2['MainPlayer'] = np.where(df_p2['Winner']==name_p2,df_p2['Winner'],df_p2['Loser'])
    df_p2['Win/Loss'] = np.where(df_p2['Winner']==name_p2,'Win','Loss')
    df_p2['Rank'] = np.where((df_p2['Winner']==name_p2),df_p2['WRank'],df_p2['LRank'])

    df_both = pd.concat([df_p1,df_p2])
    
    p1_last_rank = df_p1['Rank'].loc[df_p1['Date'] == df_p1['Date'].max()].iloc[0]
    p2_last_rank = df_p2['Rank'].loc[df_p2['Date'] == df_p2['Date'].max()].iloc[0]

    df_both = df_both.sort_values(by='Date')
    plot = px.line(data_frame=df_both,x='Date',y='Rank',color='MainPlayer',title='Rank Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")
    return plot,p1_last_rank,p2_last_rank

def historical_h2h(df,name_p1,name_p2):
    p1 = (df['H2H'].str.contains(name_p1))
    p2 = (df['H2H'].str.contains(name_p2))

    df = df[p1 & p2]
    wins_p1 = df[df['Winner']==name_p1].shape[0]
    wins_p2 = df[df['Winner']==name_p2].shape[0]
    nr_matches = df.shape[0]
    text = f'The players have faced each other {nr_matches} times. {name_p1} has won {wins_p1} out of {nr_matches}. {name_p2} has won {wins_p2} out of {nr_matches}'
    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}

    plot = px.bar(df,x='Surface',color='Winner',text='Tournament',hover_data={'SeriesPoints':True,'RoundDraw':True},title='H2H Match Winners',color_discrete_map=color_mapping)

    df = df[['Date','Tournament','Court','Surface','Winner','WRank','Loser','LRank','Wsets','Lsets','SeriesPoints','RoundDraw']]

    return plot, df, text


# if __name__=='__main__':
#     df,players_list = data_cleaner()
#     p1 = 'Federer R.'
#     p2 = 'Nadal R.'

#     p = tournament_performance(df,p1)
#     p.show()