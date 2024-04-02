import glob
import sqlite3
from sqlite3 import Error
import psycopg2
from sqlalchemy import create_engine 
from web_scraper import get_photo_url

import pandas as pd

def create_tbl_folder_csv(conn, paths, table_name):
    df = pd.DataFrame()
    for p in paths:
        data_folder = glob.glob(p)

        for file in data_folder:
            t = pd.read_csv(file,low_memory=False)
            df = pd.concat([df,t])

    df['id'] = list(range(0,df.shape[0]))
    print(df.head())

    df.to_sql(table_name, conn, if_exists='replace', index=False)

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def main_internal_db():
    database = r"../src/tennis_atp.db"
    data_folder = r"tennis_atp"

    # create a database connection
    conn = create_connection(database)
    print(conn)
    with conn:
        paths = [data_folder + r"\atp_matches_2*.csv"]
        create_tbl_folder_csv(conn,paths,'matches')

        path_1 = data_folder + r"\atp_rankings_1*.csv"
        path_2 = data_folder + r"\atp_rankings_2*.csv"
        paths = [path_1,path_2]
        create_tbl_folder_csv(conn,paths,'rankings')

        paths = [data_folder + r"\atp_players*.csv"]
        create_tbl_folder_csv(conn,paths,'players')
    
    paths = [r"sql_files\*.sql"]
    execute_sql(conn,paths)

def create_connection_external():
    conn = None
    
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

def execute_sql(conn,paths):

    cur = conn.cursor()

    for p in paths:
        data_folder = glob.glob(p)
        for sql_file in data_folder:
            print(sql_file)

            with open(sql_file, 'r') as fd:
                command = fd.read().format()
            print('start exec')
            cur.executescript(command)  
            print('finish exec')

    conn.close()

def main_external_db():
    # create a database connection
    conn_engine,conn_db = create_connection_external()
    paths = [r"sql_files\*.sql"]
    execute_sql(conn_db,paths)

if __name__ == '__main__':
    main_internal_db()
    # main_external_db()