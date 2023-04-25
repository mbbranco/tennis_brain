# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, dash_table,html,get_asset_url          # pip install dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import plotly.express as px
import plotly.io as pio

from data_prep import data_import, get_more_info
from plotting_functions import historical_h2h, tournament_performance,win_loss_ratio,rank_evol, ratio_evol, tournament_predictor

from model_predict import run_predictor_tournament
from web_scraper import get_current_ranking_photo

import numpy as np
import pandas as pd

pio.templates.default = "plotly_dark"

# incorporate data into app
matches,rankings,players = data_import()
players_dict, tourn_dict, rounds_list, matches = get_more_info(matches,rankings,players)
features_table = ['tourney_date','tourney_name','tourney_points','surface','round','winner_name','winner_rank','loser_name','loser_rank','score']

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app._favicon = 'img/tennis_logo.ico'
app.title = 'Tennis Brain'

logo = html.Div([
    html.Img(src='assets/img/tennis_logo.png',width=50,height=50)
])

p2_img = html.Div([
    html.Img(id='p2_img',height=200)
])

p1_img = html.Div([
    html.Img(id='p1_img',height=200)
])

mytitle = dcc.Markdown(children='# Tennis Brain')
mytitle2 = dcc.Markdown(children='# Head2Head')
mytitle3 = dcc.Markdown(children='# Predictor')

mygraph_p1_wl = dcc.Graph(id='p1_wl',figure={})
mygraph_p2_wl = dcc.Graph(id='p2_wl',figure={})
mygraph_p1_tp = dcc.Graph(id='p1_tp',figure={})
mygraph_p2_tp = dcc.Graph(id='p2_tp',figure={})
mygraph_rank = dcc.Graph(id='rank_evol',figure={})
mygraph_ratio = dcc.Graph(id='ratio_evol',figure={})

mygraph_h2h = dcc.Graph(id='h2h_graph',figure={})
mytext_h2h = dcc.Markdown(id='h2h_info',children='')

mytable_h2h = dash_table.DataTable(
        id='h2h_table',
        columns = [{"name": i, "id": i} for i in matches[features_table]],
        data = matches[features_table].to_dict('records'),
        style_header={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
)

# mygraph_predict = dcc.Graph(id='predict_graph',figure={})
# text_tournament = dcc.Markdown(id='tournament_info',children='')

dropdown_p1 = html.Div(dcc.Dropdown(options=list(players_dict.keys()),
                        value='Carlos Alcaraz',  # initial value displayed when page first loads
                        clearable=False,id='p1_dd',placeholder='Select a Player'))

dropdown_p2 = html.Div(dcc.Dropdown(options=list(players_dict.keys()),
                        value='Rafael Nadal',  # initial value displayed when page first loads
                        clearable=False,id='p2_dd',placeholder='Select a Player'))


# dropdown_tournament = html.Div(dcc.Dropdown(options=list(tourn_dict.keys()),
#                         value = 'Roland Garros',
#                         clearable=False,id='tournament_dd',placeholder='Select a Tournament'))


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([mytitle], width='auto'),
        dbc.Col([logo],width='auto',align='center')
    ], justify='left'),
    dbc.Row([
        dbc.Col([dropdown_p1], width=3),dbc.Col([p1_img], width=3),
        dbc.Col([dropdown_p2], width=3),dbc.Col([p2_img], width=3)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_p1_wl], width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_p1_tp], width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_p2_wl], width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_p2_tp], width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytitle2], width=12)
    ], justify='left'),
    dbc.Row([
        dbc.Col([mygraph_rank], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_ratio], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_h2h], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytable_h2h], width=12)
    ], justify='center'),
    # dbc.Row([
    #     dbc.Col([mytitle3], width=12)
    # ], justify='center',),
    # dbc.Row([
    #     dbc.Col([dropdown_tournament], width=4)
    # ], justify='left'),
    # dbc.Row([
    #     dbc.Col([text_tournament], width=12)
    # ], justify='center'),
    # dbc.Row([
    #     dbc.Col([mygraph_predict], width=12)
    # ], justify='center'),

], fluid=True)


# Callback allows components to interact
@app.callback(
    [Output('p1_img', component_property='src'),
    Output('p2_img', component_property='src'),
    Output('p1_wl', component_property='figure'),
    Output('p2_wl', component_property='figure'),
    Output('p1_tp', component_property='figure'),
    Output('p2_tp', component_property='figure'),
    Output('rank_evol', component_property='figure'),
    Output('ratio_evol', component_property='figure'),
    Output('h2h_graph', component_property='figure'),
    Output('h2h_table', component_property='data'),
    # Output('tournament_info', component_property='children'),
    # Output('predict_graph', component_property='figure')
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    # Input('tournament_dd', component_property='value')
    ]
)

def update_graphs(p1_name,p2_name):

# def update_graphs(p1_name,p2_name,tournament_name):
    p1_id = players_dict[p1_name][0]
    p2_id = players_dict[p2_name][0]

    p1_wl = win_loss_ratio(matches,p1_id,p1_name)
    p2_wl = win_loss_ratio(matches,p2_id,p2_name)

    p1_tp = tournament_performance(matches,p1_id,p1_name)
    p2_tp = tournament_performance(matches,p2_id,p2_name)

    rank = rank_evol(rankings,p1_id,p2_id)
    ratio = ratio_evol(matches,p1_id,p2_id)

    h2h_plot, h2h_table = historical_h2h(matches,p1_id,p1_name,p2_id,p2_name)
    h2h_table = h2h_table[features_table].to_dict('records')

    p1_img,p2_img = get_current_ranking_photo(p1_name,p2_name)
    if p1_img == None:
        p1_img = 'assets/img/tennis_logo.png'
    if p2_img==None:
        p2_img = 'assets/img/tennis_logo.png'


    # tourn_info = tourn_dict[tournament_name]
    # tourn_info.append(tournament_name)
    
    # df_result = run_predictor_tournament(p1_name,p2_name,players_dict,tourn_info,rounds_list)
    # df_result['winner_name'] = np.where(df_result['winner_id']==p1_id, p1_name, p2_name)
    # predict_result = tournament_predictor(df_result)

    # tournament_surface = tourn_dict[tournament_name][0]
    # tournament_points = tourn_dict[tournament_name][1]
    # tournament_info = f'Surface: {tournament_surface} | Points: {tournament_points} \n Rank P1 {p1_name}: {p1_rank} | Rank P2 {p2_name}: {p2_rank}'

    return [p1_img,p2_img,p1_wl,p2_wl,p1_tp,p2_tp,rank,ratio,h2h_plot,h2h_table]

    # return [p1_img,p2_img,p1_wl,p2_wl,p1_tp,p2_tp,rank,ratio,h2h_plot,h2h_table,tournament_info, predict_result]

# Run app
if __name__=='__main__':
    app.run_server(port=8051,debug=True, dev_tools_silence_routes_logging=True)