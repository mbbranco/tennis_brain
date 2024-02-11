import glob
import sqlite3
from sqlite3 import Error

import pandas as pd


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


def create_tbl_folder_csv(conn, paths, table_name):
    df = pd.DataFrame()
    for p in paths:
        data_folder = glob.glob(p)
    
        for file in data_folder:
            t = pd.read_csv(file,low_memory=False)
            df = pd.concat([df,t])
    
    df['id'] = list(range(0,df.shape[0]))
    print(df.head())

    df.to_sql(table_name, conn, if_exists='replace', index=False,dtype={'id': 'INTEGER PRIMARY KEY AUTOINCREMENT'})

def main():
    database = r"tennis_atp.db"
    data_folder = r"..\src\tennis_atp"

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


if __name__ == '__main__':
    main()