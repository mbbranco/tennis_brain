# If you prefer to run the code online instead of on your computer click:
# https://github.com/Coding-with-Adam/Dash-by-Plotly#execute-code-in-browser

from dash import Dash, dcc, Output, Input, dash_table,html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.io as pio

from data_prep_db import data_import_db, select_by_name_fetch
from plotting_functions import historical_h2h, tournament_performance,win_loss_ratio, rank_evol, ratio_evol
pio.templates.default = "plotly_dark"



### LOADING DATA
db_loc = r'tennis_atp.db'

# read players and create view
df_players = data_import_db(db_loc,r'database_sql\get_players_view.sql')
print('read players')

players_names = list(df_players['player_name'].unique())

features_table = ['tourney_date','tourney_name','tourney_points','surface','round','winner_name','winner_rank','loser_name','loser_rank','score']
df_matches = pd.DataFrame(columns=features_table)


### APP STARTS
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
        columns = [{"name": i, "id": i} for i in df_matches[features_table]],
        data = df_matches[features_table].to_dict('records'),
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
                        value='Novak Djokovic',  # initial value displayed when page first loads
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
        dbc.Col([dbc.Spinner(children=[mygraph_p1_wl], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_p1_tp], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_p2_wl], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_p2_tp], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mytitle2], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='left'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_rank], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_ratio], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mygraph_h2h], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),
    dbc.Row([
        dbc.Col([dbc.Spinner(children=[mytable_h2h], size="lg", color="primary", type="border", fullscreen=False,)],width=12),
    ], justify='center'),

], fluid=True)


# Callback allows components to interact
@app.callback(
    [Output('p1_img', component_property='src'),
    Output('p2_img', component_property='src'),
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    ]
)

def update_pics(p1_name,p2_name):

    p1_img,cols_img = select_by_name_fetch(db_loc,r'database_sql\get_img.sql',p1_name)
    p2_img,cols_img = select_by_name_fetch(db_loc,r'database_sql\get_img.sql',p2_name)

    index_col = cols_img.index('photo_url')

    p1_img = p1_img[0][index_col]
    p2_img = p2_img[0][index_col]

    print('photos')

    return [p1_img,p2_img]

@app.callback(
    [
    Output('p1_wl', component_property='figure'),
    Output('p2_wl', component_property='figure'),
    Output('p1_tp', component_property='figure'),
    Output('p2_tp', component_property='figure'),
    Output('rank_evol', component_property='figure'),
    Output('ratio_evol', component_property='figure'),
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    ],
)

def update_graphs(p1_name,p2_name):
    print('start')
    # get matches for player by player name and calculate KPIs
    p1_results, col_names_res = select_by_name_fetch(db_loc,r'database_sql\player_matches_kpis.sql',p1_name)
    p2_results, col_names_res = select_by_name_fetch(db_loc,r'database_sql\player_matches_kpis.sql',p2_name)
    results_dict = {p1_name:p1_results,p2_name:p2_results}
    print('results')    

    p1, col_names = select_by_name_fetch(db_loc,r'database_sql\get_player_info.sql',p1_name)
    p2, col_names = select_by_name_fetch(db_loc,r'database_sql\get_player_info.sql',p2_name)
    print('player_info')

    index_col = col_names.index('player_id')

    p1_id = p1[0][index_col]
    p2_id = p2[0][index_col]

        # get rankings for player by player name and calculate KPIs
    p1_ranks,col_names_ranks = select_by_name_fetch(db_loc,r'database_sql\get_rank_evol.sql',p1_id)
    p2_ranks,col_names_ranks = select_by_name_fetch(db_loc,r'database_sql\get_rank_evol.sql',p2_id)
    rankings_dict = {p1_name:p1_ranks,p2_name:p2_ranks}
    print('ranks')
    
    p1_wl, df = win_loss_ratio(results_dict,p1_name,col_names_res)
    p2_wl, df = win_loss_ratio(results_dict,p2_name,col_names_res)
    print('wl')

    n_years = 3

    p1_tp = tournament_performance(results_dict,p1_name,col_names_res,n_years)
    p2_tp = tournament_performance(results_dict,p2_name,col_names_res,n_years)
    print('tp')

    rank = rank_evol(rankings_dict,col_names_ranks,n_years)
    print('rank_evol')

    ratio = ratio_evol(results_dict,col_names_res,n_years)
    print('ratio_evol')

    return [p1_wl,p2_wl,p1_tp,p2_tp,rank,ratio]


@app.callback(
    [
    Output('h2h_graph', component_property='figure'),
    Output('h2h_table', component_property='data'),
    ],
    [Input('p1_dd', component_property='value'),
    Input('p2_dd', component_property='value'),
    ],
)

def update_h2h(p1_name,p2_name):
    # get h2h matches
    list_names = (p1_name,p2_name)
    h2h,col_names_h2h = select_by_name_fetch(db_loc,r'database_sql\get_h2h.sql',list_names)
    print('h2h')

    h2h_plot, h2h_table = historical_h2h(h2h,col_names_h2h,p1_name,p2_name)
    h2h_table = h2h_table[col_names_h2h].to_dict('records')
    print('h2h_tbl')

    return [h2h_plot,h2h_table]
# Run app
if __name__=='__main__':
    app.run_server(port=8051,debug=False, dev_tools_silence_routes_logging=True)   