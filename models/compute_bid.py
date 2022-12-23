def compute_bid(df):
    # ACOS   = Cost/Revenue= avg_cpc #clicks / (conversion_rate*conversion_value* #clicks)
    #       = avg_cpc / (conversion_rate*conversion_value)
    # =>avg_cpc  = ACOS * (conversion_rate*conversion_value) (1)
    # avg_cpc = a * max_cpc + b (2)
    # (1) + (2) => max_cpc = (avg_cpc -b)/a
    #                     = (ACOS * (conversion_rate*conversion_value) - b)/a
    df['bid before limit'] = df['acos target'] * df['CR'] * df['conv_value'] / df['a']
    return df


def no_bid_change(new_bid, old_bid):
    if abs(new_bid / old_bid - 1) <= 0.05:
        bid_change = 0
    else:
        bid_change = np.sign(new_bid - old_bid)
    return valid_bid(new_bid, min_bid, max_bid, step_size)


def limit_bid_change(old_bid, new_bid, max_change):
    upper_limit = old_bid * (1 + max_change)
    lower_limit = old_bid / (1 + max_change)
    if new_bid > upper_limit:
        new_bid = upper_limit
    if new_bid < lower_limit:
        new_bid = lower_limit
    print(f"new bid after limit{new_bid}")
    return new_bid


def valid_bid(new_bid, campaign_type='SP'):
    if campaign_type == 'SP':
        limit = 0.02
    if campaign_type == 'SB':
        limit = 0.1
    if new_bid < limit:
        new_bid = limit
    if new_bid == '':
        new_bid = limit
    return new_bid


def platform_limit_bid():
    pass


def compute_all_bid(df):
    # https://towardsdatascience.com/apply-function-to-pandas-dataframe-rows-76df74165ee4
    pass
