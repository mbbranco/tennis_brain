import plotly.express as px
import numpy as np
import pandas as pd
from data_prep import data_import, get_more_info

def win_loss_ratio(df,player_id,player_name):
    df_p1 = df[(df['winner_id']==player_id) | (df['loser_id']==player_id)].copy()
    df_p1['win_loss'] =  np.where(df_p1['winner_id']==player_id,'Win','Loss')
    df_p1['win_loss_val'] =  np.where(df_p1['winner_id']==player_id,1,-1)
    df_p1 = df_p1.sort_values(by='surface')

    txt = ''
    for surface,group in df_p1.groupby('surface'):
        wins = group[group['win_loss_val']==1]['win_loss_val'].sum()
        losses = -group[group['win_loss_val']==-1]['win_loss_val'].sum()
        txt += f'{surface}: {wins/losses:.2f} | '

    plot = px.bar(data_frame=df_p1,x='surface',y='win_loss_val',color='win_loss',\
        color_discrete_map={
        'Win': 'green',
        'Loss': 'red'
        },\
        title=f"Win/Loss Ratio for {player_name}: {txt}",\
        hover_data={'tourney_name':True,'tourney_date':True})

    return plot

def tournament_performance(df,player_id,player_name):
    df_p1 = df[(df['winner_id']==player_id) | (df['loser_id']==player_id)].copy().reset_index()
    df_p1['result'] = np.where(df_p1['winner_id']==player_id,1,0)

    idx = df_p1.groupby(['tourney_year','tourney_name','tourney_points','surface'])['round_level'].idxmin()
    last_rounds = df_p1.loc[idx].sort_values(by='tourney_date')

    max_round = df['round_level'].max()
    last_rounds['tourney_winner'] = np.where((last_rounds['round_level']==0)&(last_rounds['winner_id']==player_id),'Winner',last_rounds['round'])
    last_rounds['tourney_winner'] = np.where((last_rounds['round_level']==0)&(last_rounds['winner_id']!=player_id),'Runner-Up',last_rounds['tourney_winner'])

    last_rounds['round_level'] = max_round - last_rounds['round_level']

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}
    plot = px.bar(data_frame=last_rounds,x='tourney_date',y='round_level',color='surface',title=f"Tournament Performance for {player_name}",text='tourney_winner',\
            hover_data={'tourney_name':True,'tourney_points':True,'winner_name':True},color_discrete_map=color_mapping)

    plot.update_xaxes(type='category')
    plot.update_xaxes(categoryorder='category ascending') 
    plot.add_hline(y=7, line_width=3, line_dash="dash", line_color="gold")
    
    return plot


def ratio_evol(matches,p1_id,p2_id):
    ratios = matches.copy()

    ratios['p1'] = np.where((matches['winner_id']==p1_id),'W','')
    ratios['p1'] = np.where((matches['loser_id']==p1_id),'L',ratios['p1'])
    ratios['p2'] = np.where((matches['winner_id']==p2_id),'W','')
    ratios['p2'] = np.where((matches['loser_id']==p2_id),'L',ratios['p2'])

    ratios_filtered = ratios[(ratios['p1']!='')|(ratios['p2']!='')]

    ratios_p1 = ratios_filtered[ratios_filtered['p1']!=''].copy()
    ratios_p1['win_loss_ratio'] = np.where(ratios_p1['p1']=='W',ratios_p1['winner_win_loss_ratio'],ratios_p1['loser_win_loss_ratio'])
    ratios_p1['player_name'] = np.where(ratios_p1['p1']=='W',ratios_p1['winner_name'],ratios_p1['loser_name'])

    ratios_p2 = ratios_filtered[ratios_filtered['p2']!=''].copy()
    ratios_p2['win_loss_ratio'] = np.where(ratios_p2['p2']=='W',ratios_p2['winner_win_loss_ratio'],ratios_p2['loser_win_loss_ratio'])
    ratios_p2['player_name'] = np.where(ratios_p2['p2']=='W',ratios_p2['winner_name'],ratios_p2['loser_name'])
    
    ratios_all = pd.concat([ratios_p1,ratios_p2])
    ratios_all = ratios_all[['tourney_date','player_name','win_loss_ratio']]
    start_date = max(ratios_p1['tourney_date'].min(),ratios_p2['tourney_date'].min())
    ratios_all = ratios_all[ratios_all['tourney_date']>=start_date]

    ratios_all = ratios_all.sort_values(by='tourney_date')
    plot = px.line(data_frame=ratios_all,x='tourney_date',y='win_loss_ratio',color='player_name',title='W/L Ratio Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")
    
    return plot

def rank_evol(rankings,p1_id,p2_id):
    df_rank = rankings[(rankings['player']==p1_id)|(rankings['player']==p2_id)].copy()
    df_rank = df_rank.sort_values(by='ranking_date')
    df_aux = df_rank.groupby(['player'])['ranking_date'].min().reset_index()
    start_date = df_aux['ranking_date'].max()
    df_rank = df_rank[df_rank['ranking_date']>=start_date]
    plot = px.line(data_frame=df_rank,x='ranking_date',y='rank',color='name',title='Rank Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")
    plot.update_yaxes(autorange="reversed")
    
    return plot

def historical_h2h(df,p1_id,p1_name,p2_id,p2_name):
    df_h2h = df[((df['winner_id']==p1_id)&(df['loser_id']==p2_id)) | ((df['winner_id']==p2_id)&(df['loser_id']==p1_id))]

    wins_p1 = df_h2h[df_h2h['winner_id']==p1_id].shape[0]
    wins_p2 = df_h2h[df_h2h['winner_id']==p2_id].shape[0]

    nr_matches = df_h2h.shape[0]

    if wins_p1>wins_p2:
        text = f'{p1_name} is currently winning the H2H with {wins_p1} wins out of {nr_matches} matches against {p2_name}'
    elif wins_p1==wins_p2:
        text = f'{p1_name} and {p2_name} are currently tied on the H2H after {nr_matches} matches between the both.'
    else:
        text = f'{p2_name} is currently winning the H2H with {wins_p2} wins out of {nr_matches} matches against {p1_name}'

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}

    plot = px.bar(df_h2h,x='surface',color='winner_name',text='tourney_name',hover_data={'tourney_points':True,'round':True,'tourney_date':True},title=f'H2H - {text} ',color_discrete_map=color_mapping)

    return plot, df_h2h

def tournament_predictor(df):
    df['aux'] = 1
    plot = px.bar(data_frame=df,x='round',y='aux',color='winner_name',title=f"Tournament Predictor",\
            hover_data={'model':True,'precision':True,'recall':True})
    
    return plot

if __name__=='__main__':
    matches,rankings,players = data_import()
    players_dict, tourn_dict, rounds_list, matches = get_more_info(matches,rankings,players)


    p1_name = 'Carlos Alcaraz'
    p1_id = 207989
    p2_name = 'Cameron Norrie'
    p2_id = 111815

    players_dict_selected = {}
    players_dict_selected[p1_name] = p1_id
    players_dict_selected[p2_name] = p2_id

    # p = tournament_performance(matches,p1_id,p1_name)
    # p.show()
    p = win_loss_ratio(matches,p1_id,p1_name)
    p.show()
    # p = rank_evol(rankings,p1_id,p2_id)
    # p.show()

    p = ratio_evol(matches,p1_id,p2_id)
    p.show()
    # p,d,t = historical_h2h(matches,p1_id,p1_name,p2_id,p2_name)
    p.show()
    # print(t)