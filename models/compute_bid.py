def compute_bid(ACOStarget, a, conversion_rate, conversion_value, covariance, risk_level):
    # ACOS   = Cost/Revenue= avg_cpc #clicks / (conversion_rate*conversion_value* #clicks)
    #       = avg_cpc / (conversion_rate*conversion_value)
    # =>avg_cpc  = ACOS * (conversion_rate*conversion_value) (1)
    # avg_cpc = a * max_cpc + b (2)
    # (1) + (2) => max_cpc = (avg_cpc -b)/a
    #                     = (ACOS * (conversion_rate*conversion_value) - b)/a

    return (ACOStarget * ((conversion_rate + covariance * risk_level) * conversion_value)) / a


def no_bid_change(new_bid, old_bid):
    if abs(new_bid / old_bid - 1) <= 0.05:
        bid_change = 0
    else:
        bid_change = np.sign(new_bid - old_bid)
    return valid_bid(new_bid, min_bid, max_bid, step_size)


def platform_limit_bid():
    pass


def compute_all_bid(df):
    # https://towardsdatascience.com/apply-function-to-pandas-dataframe-rows-76df74165ee4
    pass
