import pandas as pd
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

def select_by_name(db, sql, name):
    conn = create_connection(db)

    cur = conn.cursor()
    with open(sql, 'r') as fd:
        command = fd.read().format(name)

    cur.execute(command)  
    
    df = pd.read_sql(command, conn)
    conn.close()

    return df

def select_by_name_fetch(db,sql,name):
    conn = create_connection(db)

    cur = conn.cursor()
    with open(sql, 'r') as fd:
        command = fd.read().format(name)

    print('start exec')
    cur.execute(command)  
    print('finish exec')
    
    rows = cur.fetchall()
    col_names = [description[0] for description in cur.description]

    conn.close()

    return rows,col_names


def data_import_db(db,sql):
    # create a database connection
    conn = create_connection(db)

    cur = conn.cursor()

    with open(sql, 'r') as fd:
        lines = fd.read().strip()

    command = lines.split(";")
    command = [c for c in command if c!=""]

    for c in command:
        cur.execute(c)

    df = pd.read_sql(command[-1], conn)

    return df


def data_import_db_simple(db,sql):
    # create a database connection
    conn = create_connection(db)

    cur = conn.cursor()


    command = sql.split(";")
    command = [c for c in command if c!=""]

    for c in command:
        cur.execute(c)

    df = pd.read_sql(command[-1], conn)

    return df

if __name__=='__main__':
    db_loc = r'tennis_atp.db'

    p1_name = 'Carlos Alcaraz'
    p2_name = 'Novak Djokovic'

    p1 = select_by_name(db_loc,r'database_sql\get_player_info.sql',p1_name)
    print(p1)
    p2 = select_by_name(db_loc,r'database_sql\get_player_info.sql',p2_name)
    print(p2)

    print('player_info')

    p1_id = p1['player_id'].iloc[0]
    p2_id = p2['player_id'].iloc[0]

    p1_ranks = select_by_name(db_loc,r'database_sql\get_rank_evol.sql',p1_id)
    print(p1_ranks)