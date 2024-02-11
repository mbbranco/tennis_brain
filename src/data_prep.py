import time
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta,datetime
import glob
import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def data_import_db(db,sql):
    # create a database connection
    conn = create_connection(db)
    with conn:
        cur = conn.cursor()
        with open(sql, 'r') as fd:
            lines = fd.read().strip()
    
        command = lines.split(";")
        command = [c for c in command if c!=""]

        for c in command:
            cur.execute(c)

        df = pd.read_sql(command[-1], conn) 
    return df


def select_by_name(db, sql, name):
    conn = create_connection(db)

    with conn:
        cur = conn.cursor()
        with open(sql, 'r') as fd:
            command = fd.read().format(name)
        print(command)
        
        cur.execute(command)        
        df = pd.read_sql(command, conn)
    return df


def calculate_score(row):
    parcials = row['score'].split(" ")
    if 'W/O' in parcials or 'RET' in parcials or 'DEF' in parcials or 'DEF.' or 'DEFAULT' in parcials:
        if row['best_of']==3:
            winner = 2
        else:
            winner = 3
        loser = 0
        return winner,loser
    
    winner = 0
    loser = 0

    for p in parcials:
        a,b = p.split("-")
        if "(" in b:
            b = b.split("(")[0]

        if int(a)>int(b):
            winner+=1
        else:
            loser+=1

    return winner,loser

def prep_matches(df,db,sql):

    df['score'] = df['score'].str.upper().replace("."," ")
    df[['winner_sets','loser_sets']]= df.apply(calculate_score, axis=1, result_type ='expand')
    df['score_quality'] = df['winner_sets']-df['loser_sets']

    conn = create_connection(db)
    with conn:
        df.to_sql(name='matches', con=conn, if_exists = 'replace', index=False)

    df_new = data_import_db(db,sql)

    return df_new

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
    st = time.time()
    db_loc = r'..\database\tennis_atp.db'

    # start by reading matches
    df_temp = data_import_db(db_loc,r'..\database\matches_create.sql')

    # transform matches and return view
    df_matches = prep_matches(df_temp,db_loc,r'..\database\matches_view.sql')

    # read players and create view
    df_players = data_import_db(db_loc,r'..\database\players.sql')

    # read rankings and create view
    df_rankings = data_import_db(db_loc,r'..\database\rankings.sql')

    # # get tournament info by tournaments name
    # tournament = select_by_name(db_loc,r'..\database\get_tournament_info.sql','Wimbledon')
    # print(tournament)

    # # get player info by players name
    # player = select_by_name(db_loc,r'..\database\get_player_info.sql','Carlos Alcaraz')
    # print(player)

    # # get last rank available by player name
    # rank = select_by_name(db_loc,r'..\database\get_last_rank.sql','Carlos Alcaraz')
    # print(rank)

    # # get matches for player by player name and calculate KPIs
    # results = select_by_name(db_loc,r'..\database\player_matches_kpis.sql','Carlos Alcaraz')
    # print(results)

