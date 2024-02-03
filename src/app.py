# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, dash_table,html,get_asset_url
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio

from data_prep import data_import, get_players_info,get_tournaments_info,get_player_id_by_name, get_kpis
from plotting_functions import historical_h2h, tournament_performance,win_loss_ratio, rank_evol, ratio_evol

from web_scraper import get_current_ranking_photo

pio.templates.default = "plotly_dark"

# incorporate data into app
matches, rankings, players = data_import()
tourn_dict = get_tournaments_info(matches)
players_dict = get_players_info(players,rankings)
df_ratios,rounds = get_kpis(matches,players)

players_names = sorted([p['name'] for k,p in players_dict.items()])
features_table = ['tourney_date','tourney_name','tourney_points','surface','round','winner_name','winner_rank','loser_name','loser_rank','score']

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app._favicon = '../assets/img/tennis_logo.ico'
app.title = 'Tennis Brain'

logo = html.Div([
    html.Img(src='../assets/img/tennis_logo.png',width=50,height=50)
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
            'color': 'white',
            'textAlign':'center'
        },
        style_data={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'textAlign':'center'
        },
        style_as_list_view=True,
)


dropdown_p1 = html.Div(dcc.Dropdown(options=list(players_names),
                        value='Carlos Alcaraz',  # initial value displayed when page first loads
                        clearable=False, id='p1_dd',placeholder='Select a Player'))

dropdown_p2 = html.Div(dcc.Dropdown(options=list(players_names),
                        value='Rafael Nadal',  # initial value displayed when page first loads
                        clearable=False, id='p2_dd',placeholder='Select a Player'))


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
        dbc.Col([mytable_h2h], width=8)
    ], justify='center'),

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
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    ]
)

def update_graphs(p1_name,p2_name):
    p1_id = get_player_id_by_name(p1_name,players_dict)
    p2_id = get_player_id_by_name(p2_name,players_dict)

    p1 = players_dict[p1_id]
    p2 = players_dict[p2_id]

    p1_wl = win_loss_ratio(matches,p1)
    p2_wl = win_loss_ratio(matches,p2)

    p1_tp = tournament_performance(matches,p1)
    p2_tp = tournament_performance(matches,p2)

    rank = rank_evol(rankings,p1,p2)
    ratio = ratio_evol(df_ratios,p1,p2)

    h2h_plot, h2h_table = historical_h2h(matches,p1,p2)
    h2h_table = h2h_table[features_table].to_dict('records')

    p1_img,p2_img = get_current_ranking_photo(p1,p2)
    if p1_img == None:
        p1_img = '../assets/img/tennis_logo.png'
    if p2_img==None:
        p2_img = '../assets/img/tennis_logo.png'

    return [p1_img,p2_img,p1_wl,p2_wl,p1_tp,p2_tp,rank,ratio,h2h_plot,h2h_table]


# Run app
if __name__=='__main__':
    app.run_server(port=8051,debug=True, dev_tools_silence_routes_logging=True)   