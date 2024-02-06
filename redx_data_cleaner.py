import os
import csv
import pandas as pd


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


def add_price_column(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            prices = select_and_resample_price('2023-01-01', '2024-01-01')
            df['time'] = pd.to_datetime(df['time'])
            df = pd.merge(df, prices, left_on='time', right_on='date')
            df.drop(columns=['price_x'], inplace=True)
            df.rename(columns={'price_y': 'price'}, inplace=True)
            df.to_csv(file_path, index=False)


def divide_usage_by_thousand(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
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
if __name__ == "__main__":
    add_price_column("redx_data")
    print("Done!")

