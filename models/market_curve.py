import sys
import array
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def market_curve(df_bid_history):
    # https://scipython.com/book/chapter-8-scipy/examples/weighted-and-non-weighted-least-squares-fitting/
    # to keep things simple we estimate the uplift by adgroup. If we don't have enough data for a keyword/target we take the adgroup default.
    sigma = (1 / df_bid_history['clicks']).to_numpy()  # sampling error
    # popt, pcov = curve_fit(avg_cpc_model, dfbid_history['Max bid'], dfbid_history['avg_cpc'], sigma=sigma, absolute_sigma=True)							#reverse the order of the years
    print(df_bid_history['avg_cpc'])
    popt, pcov = curve_fit(avg_cpc_model, df_bid_history['bid'], df_bid_history['avg_cpc'], sigma=sigma, absolute_sigma=True)
    a = popt[0]
    if a < 1. / 3.:
        a = 1. / 3.

    if a > 3.:
        a = 3.
    return a


def get_slope(df_bid_history, df_keyword_history, profileId):
    # For the slope, if we don't have enough dataset, we use the default value 1
    # if we have more than 5 dataset for a campaign, we use this data for all the target in the campaign
    # if we have more than 5 dataset for an adgroup, we use this data for all the target in the adgroup
    # if we have more than 5 dataset for a target, we use this data for  the target

    df_keyword_history['avg_cpc'] = df_keyword_history['cost'] / df_keyword_history['clicks']
    df_keyword_history['a'] = 1.0
    # merge df_bid_history and df_keyword_history

    # count clicks group by campaign
    # Group the dataframe by campaign and count the number of clicks in each group
    click_counts = df_keyword_history.groupby(['campaignId', 'date'])["clicks"].transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_keyword_history.loc[mask, "a"] = market_curve(df_bid_history)

    # count clicks group by adgroup
    # Group the dataframe by adgroup and count the number of clicks in each group
    click_counts = df_keyword_history.groupby(['adGroupId', 'date'])["clicks"].transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_keyword_history.loc[mask, "a"] = market_curve(df_bid_history)

    # count clicks group by target
    click_counts = df_keyword_history.transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_keyword_history.loc[mask, "a"] = market_curve(df_bid_history)

    return df_keyword_history


def date_ymd(d):
    return datetime.date(d.year, d.month, d.day)


def avg_cpc_model(x, a):
    return a * x  # the factor b make sense because it's not possible to bid 0


if __name__ == "__main__":
    main()
