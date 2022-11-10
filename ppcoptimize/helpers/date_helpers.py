import os
import pandas as pd
from datetime import datetime


# date should be in datetime64
def file_date(path):
    t = os.path.getmtime(path)
    return datetime.fromtimestamp(t).strftime('%Y-%m-%d')


def format_date(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    return df


def df_between_date(df):
    pass


def df_last_x_day(df):
    pass


def date_ymd(d):
    return datetime.date(d.year, d.month, d.day)
