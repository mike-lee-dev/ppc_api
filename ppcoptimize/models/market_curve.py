import sys
import array
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def market_curve(dfbid_history):
    # https://scipython.com/book/chapter-8-scipy/examples/weighted-and-non-weighted-least-squares-fitting/
    # to keep things simple we estimate the uplift by adgroup. If we don't have enough data for a keyword/target we take the adgroup default.
    sigma = (1 / dfbid_history['Clicks']).to_numpy()  # sampling error
    # popt, pcov = curve_fit(avg_cpc_model, dfbid_history['Max Bid'], dfbid_history['avg_cpc'], sigma=sigma, absolute_sigma=True)							#reverse the order of the years
    print(dfbid_history['avg_cpc'])
    popt, pcov = curve_fit(avg_cpc_model, dfbid_history['Bid'], dfbid_history['avg_cpc'], sigma=sigma,
                           absolute_sigma=True)
    a = popt[0]
    if a < 1. / 3.:
        a = 1. / 3.

    if a > 3.:
        a = 3.
    return a


def date_ymd(d):
    return datetime.date(d.year, d.month, d.day)


def avg_cpc_model(x, a):
    return a * x  # the factor b make sense because it's not possible to bid 0


if __name__ == "__main__":
    main()
