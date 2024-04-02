import pandas as pd
from sqlite3 import Error
import psycopg2
from sqlalchemy import create_engine 

def create_connection():
    conn_string = 'postgresql://mbranco:b6rtc8hyXZ0C3Rsn9BnXa1VuIF001shZ@dpg-cnf2ilda73kc73f0koog-a.frankfurt-postgres.render.com/tennis_atp'
    db = create_engine(conn_string) 

    try:
        conn_engine = db.connect() 
        conn_db = psycopg2.connect(database="postgres", 
                                user='mbranco', 
                                password='b6rtc8hyXZ0C3Rsn9BnXa1VuIF001shZ',
                                host='dpg-cnf2ilda73kc73f0koog-a.frankfurt-postgres.render.com', 
                                port= '5432')
    except Error as e:
        print(e)

    return conn_engine,conn_db

def select_by_name(sql, name):
    conn_engine,conn_db = create_connection()

    cur = conn_db.cursor()
    with open(sql, 'r') as fd:
        command = fd.read().format(name)

    cur.execute(command)  
    
    df = pd.read_sql(command, conn_db)
    conn_db.close()

    return df

def select_by_name_fetch(sql,name):
    conn_engine,conn_db = create_connection()

    cur = conn_db.cursor()
    with open(sql, 'r') as fd:
        command = fd.read().format(name)

    print('start exec')
    cur.execute(command)  
    print('finish exec')
    
    rows = cur.fetchall()
    col_names = [description[0] for description in cur.description]

    conn_db.close()

    return rows,col_names


def data_import_db(sql):
    # create a database connection
    conn_engine,conn_db = create_connection()

    cur = conn_db.cursor()

    with open(sql, 'r') as fd:
        lines = fd.read().strip()

    command = lines.split(";")
    command = [c for c in command if c!=""]

    for c in command:
        cur.execute(c)

    df = pd.read_sql(command[-1], conn_db)

    return df

if __name__=='__main__':
    p1_name = 'Carlos Alcaraz'
    p2_name = 'Novak Djokovic'

    p1 = select_by_name(r'database_sql\get_player_info.sql',p1_name)
    print(p1)
    p2 = select_by_name(r'database_sql\get_player_info.sql',p2_name)
    print(p2)

    print('player_info')

    # p1_id = p1['player_id'].iloc[0]
    # p2_id = p2['player_id'].iloc[0]

    # p1_ranks = select_by_name(db_loc,r'database_sql\get_rank_evol.sql',p1_id)
    # print(p1_ranks)

    # # start by reading matches
    # df_temp = data_import_db(db_loc,r'..\database\matches_create.sql')

    # # transform matches and return view
    # df_matches = prep_matches(df_temp,db_loc,r'..\database\matches_view.sql')

    # # read players and create view
    # df_players = data_import_db(db_loc,r'..\database\players.sql')

    # # read rankings and create view
    # df_rankings = data_import_db(db_loc,r'..\database\rankings.sql')

    # # # get tournament info by tournaments name
    # # tournament = select_by_name(db_loc,r'..\database\get_tournament_info.sql','Wimbledon')
    # # print(tournament)

    # # # get player info by players name
    # # player = select_by_name(db_loc,r'..\database\get_player_info.sql','Carlos Alcaraz')
    # # print(player)

    # # # get last rank available by player name
    # # rank = select_by_name(db_loc,r'..\database\get_last_rank.sql','Carlos Alcaraz')
    # # print(rank)

    # # # get matches for player by player name and calculate KPIs
    # # results = select_by_name(db_loc,r'..\database\player_matches_kpis.sql','Carlos Alcaraz')
    # # print(results)