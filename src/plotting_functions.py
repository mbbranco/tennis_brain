import time
import plotly.express as px
import numpy as np
import pandas as pd
# from data_prep import data_import, get_tournaments_info, get_players_info, get_player_id_by_name, get_kpis
from data_prep import data_import_db, select_by_name,prep_matches

def win_loss_ratio(df):
    txt = ''
    for surface,group in df.groupby('surface'):
        wins = group['win'].sum()
        losses = group['loss'].sum()
        txt += f'{surface}: {wins/losses:.2f} | '

    wl_ratio_ever = df['win_loss_ratio_start'].iloc[-1]
    wl_ratio_last_10 = df['win_loss_ratio_last10'].iloc[-1]
    player_name = df['player_name'].iloc[0]

    txt_wl = f'Win Loss Ratio for {player_name} || Ever: {wl_ratio_ever:.2f} | Last 10 Matches: {wl_ratio_last_10:.2f} | '
    txt_title = txt_wl + txt

    df['win_loss_val'] = np.where(df['win']==1,1,-1)
    df['win_loss'] = np.where(df['win']==1,'Win','Loss')

    plot = px.bar(data_frame=df,x='surface',y='win_loss_val',color='win_loss',\
        color_discrete_map={
        'Win': 'green',
        'Loss': 'red'
        },\
        title=txt_title,\
        hover_data={'tourney_name':True,'tourney_date':True})

    df['player_name'] = player_name
    return plot, df

def tournament_performance(df):
    player_name = df['player_name'].iloc[0]

    idx = df.groupby(['tourney_year','tourney_name','tourney_points','surface'])['round_level'].idxmin()
    last_rounds = df.loc[idx].sort_values(by='tourney_date')

    max_round = df['round_level'].max()
    last_rounds['tourney_winner'] = np.where((last_rounds['round_level']==0)&(last_rounds['win']==1),'Winner',last_rounds['round'])
    last_rounds['tourney_winner'] = np.where((last_rounds['round_level']==0)&(last_rounds['win']!=1),'Runner-Up',last_rounds['tourney_winner'])

    last_rounds['round_level'] = max_round - last_rounds['round_level']

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}

    plot = px.bar(data_frame=last_rounds,x='tourney_date',y='round_level',color='surface',title=f"Tournament Performance for {player_name}",text='tourney_winner',\
            hover_data={'tourney_name':True,'tourney_points':True},color_discrete_map=color_mapping)

    plot.update_xaxes(type='category')
    plot.update_xaxes(categoryorder='category ascending') 
    plot.add_hline(y=6, line_width=3, line_dash="dash", line_color="gold")
    
    return plot

def ratio_evol(ratios):
    start_date = ratios.groupby(['player_id'])['tourney_date'].min().max()
    ratios = ratios[ratios['tourney_date']>=start_date]
    ratios = ratios[['tourney_date','player_id','player_name','win_loss_ratio_start']].sort_values(by='tourney_date')

    plot = px.line(data_frame=ratios,x='tourney_date',y='win_loss_ratio_start',color='player_name',title='W/L Ratio Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")

    return plot

def rank_evol(rankings):
    start_date = rankings.groupby(['player'])['ranking_date'].min().max()
    rankings = rankings[rankings['ranking_date']>=start_date]
    rankings = rankings.sort_values(by='ranking_date')
    
    plot = px.line(data_frame=rankings,x='ranking_date',y='rank',color='player',title='Rank Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")
    plot.update_yaxes(autorange="reversed")
    
    return plot

def historical_h2h(df):
    p1_wins = df['win_p1'].sum()
    p2_wins = df['win_p2'].sum()
    p1_name = df[df['win_p1']==1]['winner_name'].iloc[0]
    p2_name = df[df['win_p2']==1]['winner_name'].iloc[0]

    nr_matches = df.shape[0]

    if p1_wins > p2_wins:
        text = f'{p1_name} is currently winning the H2H with {p1_wins} wins out of {nr_matches} matches against {p2_name}'
    elif p1_wins == p2_wins:
        text = f'{p1_name} and {p2_name} are currently tied on the H2H after {nr_matches} matches between the both.'
    else:
        text = f'{p2_name} is currently winning the H2H with {p2_wins} wins out of {nr_matches} matches against {p1_name}'

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}

    plot = px.bar(df,x='surface',color='winner_name',text='tourney_name',hover_data={'tourney_points':True,'round':True,'tourney_date':True},title=f'H2H - {text} ',color_discrete_map=color_mapping)

    return plot, df

def call_time(st,lt):
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    task_time = et - lt
    print('Execution time:', elapsed_time, 'seconds')
    print('Step time:', task_time, 'seconds')
    print()
    return et

if __name__=='__main__':
    db_loc = r'..\database\tennis_atp.db'
    # # start by reading matches
    # df_temp = data_import_db(db_loc,r'..\database\matches_create.sql')
    # # transform matches and return view
    # df_matches = prep_matches(df_temp,db_loc,r'..\database\matches_view.sql')
    # # read players and create view
    # df_players = data_import_db(db_loc,r'..\database\players.sql')
    # # read rankings and create view
    # df_rankings = data_import_db(db_loc,r'..\database\rankings.sql')

    st = time.time()
    p1_name = 'Carlos Alcaraz'
    p2_name = 'Novak Djokovic'

    # get matches for player by player name and calculate KPIs
    p1_results = select_by_name(db_loc,r'..\database\player_matches_kpis.sql',p1_name)
    p2_results = select_by_name(db_loc,r'..\database\player_matches_kpis.sql',p2_name)
    lt = call_time(st,st)

    p1_results['player_name'] = p1_name
    p2_results['player_name'] = p2_name
    results = pd.concat([p1_results,p2_results])

    lt = call_time(st,lt)

    # get rankings for player by player name and calculate KPIs
    p1_ranks = select_by_name(db_loc,r'..\database\get_rank_evol.sql',p1_name)
    lt = call_time(st,lt)

    p2_ranks = select_by_name(db_loc,r'..\database\get_rank_evol.sql',p2_name)
    lt = call_time(st,lt)

    p1_ranks['player_name'] = p1_name
    p2_ranks['player_name'] = p2_name
    ranks = pd.concat([p1_ranks,p2_ranks])
    lt = call_time(st,lt)

    list_names = (p1_name,p2_name)
    h2h = select_by_name(db_loc,r'..\database\get_h2h.sql',list_names)
    lt = call_time(st,lt)

    p,d = win_loss_ratio(p1_results)
    p.show()
    lt = call_time(st,lt)

    p = tournament_performance(p1_results)
    p.show()
    lt = call_time(st,lt)

    rankings = rank_evol(ranks)
    p.show()
    lt = call_time(st,lt)

    ratios = ratio_evol(results)
    ratios.show()
    lt = call_time(st,lt)

    h2h,p = historical_h2h(h2h)
    h2h.show()
    lt = call_time(st,lt)