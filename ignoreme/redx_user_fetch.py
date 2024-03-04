import os
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm

username = 'yetao'
password = 'RedxTechYt2024@!'  # Original password
encoded_password = urllib.parse.quote_plus(
    password)  # Password is now percent-encoded
host = 'redx-instance-1.cvlqw2cbanmf.ap-southeast-2.rds.amazonaws.com'
database = 'shineserver'

engine = create_engine(
    f'mysql+pymysql://{username}:{encoded_password}@{host}/{database}', pool_size=10, max_overflow=20)


def get_sn_list(engine, date='2023-06-25'):
    sql = f"""SELECT DISTINCT device_sn FROM shineserver.log_d_{date.replace('-', '_')};"""
    df = pd.read_sql(sql, con=engine)
    return df


def fetch_data_for_date(date, engine, device_sn=None):
    """
    Fetches data for a specific date from the database.

    Args:
        date (datetime.date): The date for which to fetch the data.
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine object for connecting to the database.
        device_sn (str, optional): The device serial number. Defaults to None.

    Returns:
        pandas.DataFrame: The fetched data.

    """
    date_str = date.strftime("%Y-%m-%d")
    table_name = f"log_d_{date_str.replace('-', '_')}"

    sql = f"""SELECT 
                AVG(case when sub.loadP>0 then sub.loadP else 0 end)*0.5 AS loadP_30min, 
                sub.truncated_datetime 
              FROM ( 
                    SELECT 
                        loadP, 
                        TIMESTAMP( 
                            DATE(time), 
                            MAKETIME( 
                                HOUR(time), 
                                IF(MINUTE(time) < 30, 0, 30), 
                                0 
                            ) 
                        ) AS truncated_datetime 
                    FROM 
                        {table_name} 
                    WHERE 
                        device_sn = '{device_sn}' 
                ) sub 
              GROUP BY 
                    sub.truncated_datetime;"""

    df = pd.read_sql(sql, con=engine)
    return df


def agg_tables(start_date=None, end_date=None, engine=None, device_sn=None):
    date_range = pd.date_range(start=start_date, end=end_date)

    all_df_futures = []
    with ThreadPoolExecutor() as executor:
        for date in date_range:
            future = executor.submit(
                fetch_data_for_date, date, engine, device_sn=device_sn)
            all_df_futures.append(future)

    # Collecting results
    all_df = [future.result() for future in as_completed(all_df_futures)]

    # Aggregate all DataFrames
    aggregated_df = pd.concat(all_df)
    return aggregated_df

# get all csv file name under current folder


def get_csv_file_names():
    file_names = []
    for file in os.listdir():
        if file.endswith(".csv"):
            file_names.append(file[4:])
    return file_names


def get_sns_of_johnathan():
    return ['RX2505ACA10J0A160016', '011LOKL140058B', 'RX2505ACA10J0A180003']


def get_common_sns(start_date, end_date, engine):
    df_1 = get_sn_list(engine=engine, date=start_date)
    df_2 = get_sn_list(engine=engine, date=end_date)

    common_sns = set(df_1['device_sn']).intersection(set(df_2['device_sn']))
    return common_sns


def get_meter_start_date_johnathan():
    date_search_start = '2024-01-01'
    date_search_end = '2024-02-07'
    date_range = pd.date_range(start=date_search_start, end=date_search_end)
    for date in date_range:
        print(f'Searching {date}')
        df = get_sn_list(engine=engine, date=date.strftime('%Y-%m-%d'))
        if all(sn in df['device_sn'].values for sn in get_sns_of_johnathan()):
            print(f'Found in {date}')
            return date


def get_meter_end_date_johnathan():

    date_search_start = '2023-11-10'
    date_search_end = '2024-02-07'
    date_range = pd.date_range(start=date_search_start, end=date_search_end)[::-1]


    for date in date_range:
        print(f'Searching {date}')
        df = get_sn_list(engine=engine, date=date.strftime('%Y-%m-%d'))
        if all(sn in df['device_sn'].values for sn in get_sns_of_johnathan()):
            return date


if __name__ == '__main__':
    date_end = get_meter_end_date_johnathan()
    date_start = date_end - timedelta(days=30) 
    
    sns = get_sns_of_johnathan()
    for sn in sns:
        print(f'Processing {sn}')
        df = agg_tables(start_date=date_start, end_date=date_end,
                        engine=engine, device_sn=sn)
        if df.empty:
            continue
        os.makedirs('jc', exist_ok=True)
        df.to_csv(f'jc/agg_{sn}.csv', index=False)
