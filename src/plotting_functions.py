import datetime
import time
import plotly.express as px
import numpy as np
import pandas as pd
# from data_prep import data_import, get_tournaments_info, get_players_info, get_player_id_by_name, get_kpis
from data_prep_db import select_by_name,select_by_name_fetch

def win_loss_ratio(dict,k,cols):
    vals = dict[k]
    df_results = pd.DataFrame(vals,columns=cols)

    txt = ''
    for surface,group in df_results.groupby('surface'):
        wins = group['win'].sum()
        losses = group['loss'].sum()
        txt += f'{surface}: {wins/losses:.2f} | '

    wl_ratio_ever = df_results['win_loss_ratio_start'].iloc[-1]
    wl_ratio_last_10 = df_results['win_loss_ratio_last10'].iloc[-1]
    player_name = k

    txt_wl = f'Win Loss Ratio for {player_name} || Ever: {wl_ratio_ever:.2f} | Last 10 Matches: {wl_ratio_last_10:.2f} | '
    txt_title = txt_wl + txt

    df_results['win_loss_val'] = np.where(df_results['win']==1,1,-1)
    df_results['win_loss'] = np.where(df_results['win']==1,'Win','Loss')

    plot = px.bar(data_frame=df_results,x='surface',y='win_loss_val',color='win_loss',\
        color_discrete_map={
        'Win': 'green',
        'Loss': 'red'
        },\
        title=txt_title,\
        hover_data={'tourney_name':True,'tourney_date':True})

    return plot, df_results

def tournament_performance(dict,k,cols,n_years=None):
    vals = dict[k]
    df_perf = pd.DataFrame(vals,columns=cols)
    player_name = k

    if n_years != None:
        first_date = pd.to_datetime(df_perf['tourney_date'].max()) - pd.DateOffset(years=n_years)
        first_date = str(first_date)
        df_perf = df_perf[df_perf['tourney_date']>=first_date]

    idx = df_perf.groupby(['tourney_year','tourney_name','tourney_points','surface'])['round_level'].idxmin()
    last_rounds = df_perf.loc[idx].sort_values(by='tourney_date')

    last_rounds['last_round'] = np.where((last_rounds['round_level']==0)&(last_rounds['win']==1),'🏆Winner',last_rounds['round'])
    last_rounds['last_round'] = np.where((last_rounds['round_level']==0)&(last_rounds['win']!=1),'🥈Runner-Up',last_rounds['last_round'])

    last_rounds['round_level'] = 7 - last_rounds['round_level']

    color_mapping = {'Clay':'#FFA15A','Grass':'#00CC96','Hard':'#636EFA','Carpet':'#AB63FA'}

    plot = px.bar(data_frame=last_rounds,x='tourney_date',y='round_level',color='surface',
                    title=f"Tournament Performance for {player_name}",text='last_round',\
                    hover_data={'tourney_name':True,'tourney_points':True},color_discrete_map=color_mapping)

    plot.update_xaxes(type='category')
    plot.update_xaxes(categoryorder='category ascending') 
    plot.add_hline(y=7, line_width=3, line_dash="dash", line_color="gold")
    
    return plot

def ratio_evol(dict,cols,n_years=None):

    df_ratios = pd.DataFrame()
    for k, v in dict.items():
        df_aux = pd.DataFrame(v,columns=cols)
        df_aux['player_name'] = k
        df_ratios = pd.concat([df_ratios,df_aux])

    if n_years == None:
        first_date = df_ratios.groupby(['player_id'])['tourney_date'].min().max()
    else:
        start_date = df_ratios.groupby(['player_name'])['tourney_date'].max().min()
        first_date = pd.to_datetime(start_date) - pd.DateOffset(years=n_years)
        first_date = str(first_date)

    ratios = df_ratios[df_ratios['tourney_date']>=first_date]
    ratios = ratios[['tourney_date','player_id','player_name','win_loss_ratio_start']].sort_values(by='tourney_date')

    plot = px.line(data_frame=ratios,x='tourney_date',y='win_loss_ratio_start',color='player_name',title='W/L Ratio Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")

    return plot

def rank_evol(rankings_dict,col_names,n_years=None):

    df_rankings = pd.DataFrame()
    for k, v in rankings_dict.items():
        df_aux = pd.DataFrame(v,columns=col_names)
        df_aux['player_name'] = k
        df_rankings = pd.concat([df_rankings,df_aux])

    if n_years == None:
        first_date = df_rankings.groupby(['player_name'])['ranking_date'].min().max()
    else:
        start_date = df_rankings.groupby(['player_name'])['ranking_date'].max().min()
        first_date = pd.to_datetime(start_date) - pd.DateOffset(years=n_years)
        first_date = str(first_date)

    rankings = df_rankings[df_rankings['ranking_date']>=first_date]
    rankings = rankings.sort_values(by='ranking_date')

    plot = px.line(data_frame=rankings,x='ranking_date',y='rank',color='player_name',title='Rank Evolution')
    plot.update_traces(mode="markers+lines", hovertemplate=None)
    plot.update_layout(hovermode="x")
    plot.update_yaxes(autorange="reversed")
    
    return plot

def historical_h2h(rows,cols,p1_name,p2_name):

    df = pd.DataFrame(rows,columns=cols)
    p1_wins = sum(df['winner_name']==p1_name)
    p2_wins = sum(df['winner_name']==p2_name)

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
    db_loc = 'tennis_atp.db'
    p1_name = 'Carlos Alcaraz'
    p2_name = 'Stefanos Tsitsipas'
    st = time.time()
    lt = call_time(st,st)

    # print('start')
    # # get matches for player by player name and calculate KPIs
    # p1_results, col_names_res = select_by_name_fetch(db_loc,r'database_sql\player_matches_kpis.sql',p1_name)
    # p2_results, col_names_res = select_by_name_fetch(db_loc,r'database_sql\player_matches_kpis.sql',p2_name)
    # results_dict = {p1_name:p1_results,p2_name:p2_results}

    # print('results')    
    # lt = call_time(lt,st)

    # p1, col_names = select_by_name_fetch(db_loc,r'database_sql\get_player_info.sql',p1_name)
    # p2, col_names = select_by_name_fetch(db_loc,r'database_sql\get_player_info.sql',p2_name)
    # print('player_info')
    # lt = call_time(st,lt)

    # index_col = col_names.index('player_id')

    # p1_id = p1[0][index_col]
    # p2_id = p2[0][index_col]

    # get rankings for player by player name and calculate KPIs
    # p1_ranks,col_names_ranks = select_by_name_fetch(db_loc,r'database_sql\get_rank_evol.sql',p1_id)
    # p2_ranks,col_names_ranks = select_by_name_fetch(db_loc,r'database_sql\get_rank_evol.sql',p2_id)
    # rankings_dict = {p1_name:p1_ranks,p2_name:p2_ranks}
    # print('ranks')
    # lt = call_time(st,lt) 
    
    # p1_wl,df = win_loss_ratio(results_dict,p1_name,col_names_res)
    # p2_wl,df = win_loss_ratio(results_dict,p2_name,col_names_res)
    # print('wl')
    # lt = call_time(st,lt)

    # p1_tp = tournament_performance(results_dict,p1_name,col_names_res,2)
    # p2_tp = tournament_performance(results_dict,p1_name,col_names_res,1)
    # print('tp')
    # lt = call_time(st,lt)

    # rank = rank_evol(rankings_dict,col_names_ranks)
    # print('rank_evol')
    # lt = call_time(st,lt)

    # ratio = ratio_evol(results_dict,col_names_res,1)
    # print('ratio_evol')
    # lt = call_time(st,lt)

    # get h2h matches
    list_names = (p1_name,p2_name)
    h2h,col_names_h2h = select_by_name_fetch(db_loc,r'database_sql\get_h2h.sql',list_names)
    print('h2h')
    lt = call_time(st,lt)

    h2h_plot, h2h_table = historical_h2h(h2h,col_names_h2h,p1_name,p2_name)
    h2h_table = h2h_table[col_names_h2h].to_dict('records')
    print('h2h_tbl')
    lt = call_time(st,lt)

    p1_img,cols_img = select_by_name_fetch(db_loc,r'database_sql\get_img.sql',p1_name)
    p2_img,cols_img = select_by_name_fetch(db_loc,r'database_sql\get_img.sql',p2_name)

    index_col = cols_img.index('photo_url')

    p1_img = p1_img[0][index_col]
    p2_img = p2_img[0][index_col]

    print('photos')
    lt = call_time(st,lt)

    # show plots
    # p1_wl.show()
    # p2_wl.show()
    # p1_tp.show()
    # p2_tp.show()
    # rank.show()
    # ratio.show()
    h2h_plot.show()
    print(h2h_table)
