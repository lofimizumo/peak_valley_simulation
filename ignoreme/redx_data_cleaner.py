import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from ignoreme.amber_fetcher import get_prices

def change_csv_headers(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines[0] = "usage,time,price\n"  # Change the header to "usage, time, price"
            with open(file_path, 'w') as file:
                file.writelines(lines)


def select_and_resample_price(date_start, date_end):
    df = pd.read_csv("qld_price_merged.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    date_start = pd.to_datetime(date_start)
    date_end = pd.to_datetime(date_end)
    mask = (df.index >= date_start) & (df.index <= date_end)
    df = df.loc[mask]
    resampled_price = df['price'].resample('30T').mean()

    return resampled_price


def add_price_column(directory_path, date_start, date_end):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            prices = select_and_resample_price(date_start, date_end)
            df['time'] = pd.to_datetime(df['time'])
            df = pd.merge(df, prices, left_on='time', right_on='date')
            df.drop(columns=['price_x'], inplace=True)
            df.rename(columns={'price_y': 'price'}, inplace=True)
            df.to_csv(file_path, index=False)

def add_price_column_jc(directory_path, date_start, date_end):
    date_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
    date_end = (datetime.now()-timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
    prices = get_prices(date_start, date_end,
                        'psk_2d5030fe84a68769b6f48ab73bd48ebf', resolution=30)
    df_prices = pd.DataFrame(prices['general'])
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize('Australia/Brisbane')
            df = pd.merge(df, df_prices, left_on='time', right_on='date')
            df.drop(columns=['price_x','date'], inplace=True)
            df['time'] = df['time'].dt.tz_localize(None)
            df.rename(columns={'price_y': 'price'}, inplace=True)
            filename = filename[:-4] + '_merged.csv' 
            file_path = os.path.join(directory_path, filename)

            df.to_csv(file_path, index=False)

def divide_usage_by_thousand(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith("_merged.csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            df['usage'] = df['usage'] / 1000 
            df.to_csv(file_path, index=False)


def clip_price(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            df['price'] = df['price'].clip(0, 300)
            df.to_csv(file_path, index=False)

def solar_exposure_data_cleaner(filename):
    file_path = os.path.join("", filename)
    df = pd.read_csv(file_path)
    df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    df.rename(columns={'Daily global solar exposure (MJ/m*m)': 'solar_exposure'}, inplace=True)
    df = df[['date', 'solar_exposure']]
    df.to_csv('solar_clean', index=False)

def jc_data_cleaner():
    date_end = datetime.now()
    date_start = date_end - timedelta(days=30)
    change_csv_headers("jc") 
    add_price_column_jc("jc", date_start, date_end)
    divide_usage_by_thousand("jc")

if __name__ == "__main__":
    jc_data_cleaner()
    print("Done!")

