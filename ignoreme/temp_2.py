import pandas as pd

def add_peak_times(df):
    def calculate_peak_times(group):
        max_price_row = group.loc[group['price'].idxmax()]
        group['Peak time start'] = max_price_row['date'] - pd.Timedelta(hours=1)
        group['Peak time end'] = max_price_row['date'] + pd.Timedelta(hours=1)
        return group
    df['date'] = pd.to_datetime(df['date']) 
    df['date_only'] = df['date'].dt.date  # Extract date part for grouping
    grouped = df.groupby('date_only').apply(calculate_peak_times)
    grouped.drop('date_only', axis=1, inplace=True)
    return grouped
