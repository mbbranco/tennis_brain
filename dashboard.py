# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, dash_table,html,get_asset_url          # pip install dash
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import plotly.express as px
import plotly.io as pio

from data_prep import data_cleaner, get_more_info
from plotting_functions import historical_h2h, tournament_performance,win_loss_ratio,rank_evol
from model_predict import run_predictor

import numpy as np
import pandas as pd

pio.templates.default = "plotly_dark"

# incorporate data into app
df = data_cleaner()
players_list, tourn_dict, rounds_list = get_more_info(df)
# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app._favicon = (r"tennis_logo.ico")
app.title = 'Tennis Brain'

logo = html.Div([
    html.Img(src=get_asset_url('tennis_logo.png'),width=50,height=50)
])
mytitle = dcc.Markdown(children='# Tennis App')
mytitle2 = dcc.Markdown(children='# Head2Head')
mytitle3 = dcc.Markdown(children='# Predictor')

mygraph_p1_wl = dcc.Graph(id='p1_wl',figure={})
mygraph_p2_wl = dcc.Graph(id='p2_wl',figure={})
mygraph_p1_tp = dcc.Graph(id='p1_tp',figure={})
mygraph_p2_tp = dcc.Graph(id='p2_tp',figure={})
mygraph_rank = dcc.Graph(id='rank_evol',figure={})

mygraph_h2h = dcc.Graph(id='h2h_graph',figure={})
mytext_h2h = dcc.Markdown(id='h2h_info',children='')

mytable_h2h = dash_table.DataTable(
        id='h2h_table',
        columns = [{"name": i, "id": i} for i in df],
        data = df.to_dict('records'),
        style_header={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
)

mytext_predict = dcc.Markdown(id='predict_result',children='')
text_tournament = dcc.Markdown(id='tournament_info',children='')

dropdown_p1 = html.Div(dcc.Dropdown(options=players_list,
                        value='Federer R.',  # initial value displayed when page first loads
                        clearable=False,id='p1_dd',placeholder='Select a Player'))

dropdown_p2 = html.Div(dcc.Dropdown(options=players_list,
                        value='Nadal R.',  # initial value displayed when page first loads
                        clearable=False,id='p2_dd',placeholder='Select a Player'))


dropdown_tournament = html.Div(dcc.Dropdown(options=list(tourn_dict.keys()),
                        clearable=False,id='tournament_dd',placeholder='Select a Tournament'))

dropdown_round = html.Div(dcc.Dropdown(options=rounds_list,
                        clearable=False,id='round_dd',placeholder='Select a Round'))



app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([mytitle], width='auto'),
        dbc.Col([logo],width='auto',align='center')
    ], justify='left'),
    dbc.Row([
        dbc.Col([dropdown_p1], width=6),
        dbc.Col([dropdown_p2], width=6)
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
        dbc.Col([mytext_h2h], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mygraph_h2h], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytable_h2h], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytitle3], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([dropdown_tournament], width=6),
        dbc.Col([dropdown_round], width=6)
    ], justify='center'),
    dbc.Row([
        dbc.Col([text_tournament], width=12)
    ], justify='center'),
    dbc.Row([
        dbc.Col([mytext_predict], width=12)
    ], justify='center'),

], fluid=True)


# Callback allows components to interact
@app.callback(
    [Output('p1_wl', component_property='figure'),
    Output('p2_wl', component_property='figure'),
    Output('p1_tp', component_property='figure'),
    Output('p2_tp', component_property='figure'),
    Output('rank_evol', component_property='figure'),
    Output('h2h_graph', component_property='figure'),
    Output('h2h_info', component_property='children'),
    Output('h2h_table', component_property='data'),
    Output('tournament_info', component_property='children'),
    Output('predict_result', component_property='children')
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    Input('tournament_dd', component_property='value'),
    Input('round_dd', component_property='value')
    ]
)

def update_graph(p1,p2,tournament_name,tournament_phase):
    p1_wl = win_loss_ratio(df,p1)
    p2_wl = win_loss_ratio(df,p2)
    p1_tp = tournament_performance(df,p1)
    p2_tp = tournament_performance(df,p2)
    rank,p1_last_rank,p2_last_rank = rank_evol(df,p1,p2)
    h2h_plot, h2h_table, h2h_text = historical_h2h(df,p1,p2)

    h2h_text = "#### " + h2h_text
    h2h_table = h2h_table.to_dict('records')
    if tournament_name != None and tournament_phase!=None:
        tournament_surface = tourn_dict[tournament_name][0]
        tournament_points = tourn_dict[tournament_name][1]
        tournament_date = '2022-09-20'
        #TODO: adicionar um date picker para selecionar a data do torneio
        #TODO: escolher a fase do torneio
        tournament_info = f'Surface: {tournament_surface} | Points: {tournament_points} \n Rank P1 {p1}: {p1_last_rank} | Rank P2 {p2}: {p2_last_rank}'

        winner_name,model_name,preci,recall = run_predictor(p1,p2,p1_last_rank,p2_last_rank,tournament_date,tournament_points,tournament_phase,tournament_surface)
        predict_result = f'### The winner will be {winner_name} \n Model used: {model_name} | Precision {preci:.2%} | Recall {recall:.2%}'
    else:
        predict_result = ''
        tournament_info = ''

    return [p1_wl,p2_wl,p1_tp,p2_tp,rank,h2h_plot,h2h_text,h2h_table, tournament_info, predict_result]

# Run app
if __name__=='__main__':
    app.run_server(port=8051,debug=True, dev_tools_silence_routes_logging=False)