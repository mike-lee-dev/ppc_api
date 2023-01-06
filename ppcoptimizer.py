# from models.conversion_rate_estimation import clustering
import pandas as pd
import numpy as np
from utils import input_output, dataframe, mongo_db
from helpers import date_helpers
import global_var
from models import cluster, conversion_rate_estimation, object_structure, compute_bid, market_curve
from datetime import datetime, date
import math
import glob
import os
import datetime
import requests


def main():  # optimize all accounts
    global_var.db = mongo_db.db_connect()
    accounts = mongo_db.read_collection_as_df('accounts', global_var.db)
    json_dict = {}
    for i in range(len(accounts)):
        profileId = accounts.iloc[i]['profileId']
        print(f"{profileId}  --- Merge")
        df_new_bid = optimize_account(profileId)
        df_campaign, df_adgroup, df_keyword = update_bid_to_db(df_new_bid, profileId)
        df_campaign.to_csv("./data/result_campaign.csv")
        df_adgroup.to_csv("./data/result_adgroup.csv")
        df_keyword.to_csv("./data/result_keyword.csv")
        update_into_db(df_campaign, accounts.iloc[i])
        update_into_db(df_adgroup, accounts.iloc[i])
        update_into_db(df_keyword, accounts.iloc[i])
        json_dict[profileId + '_campaign'] = df_campaign.to_json(default_handler=str)
        json_dict[profileId + '_adgroup'] = df_adgroup.to_json(default_handler=str)
        json_dict[profileId + '_keyword'] = df_keyword.to_json(default_handler=str)
    return json_dict


def optimize_account(profileId):
    df_campaign = input_output.get_campaign(profileId)
    # df_campaign.to_csv('./data/df_campaign.csv')
    df_adgroup = input_output.get_adgroup(profileId)
    # df_adgroup.to_csv('./data/df_adgroup.csv')
    df_keyword = input_output.get_keyword(profileId)
    # df_keyword.to_csv('./data/df_keyword.csv')
    df_kw_history = input_output.read_keyword_history(profileId)
    # df_kw_history.to_csv('./data/df_kw_history.csv')
    df_target = input_output.read_targets(profileId)
    # df_target.to_csv('./data/df_targets.csv')
    df_target_history = input_output.read_target_history(profileId)
    # df_target_history.to_csv('./data/df_target_history.csv')
    # df_price = input_output.get_price(profileId)
    df_history = merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history, df_target, df_target_history)
    # df_history.to_csv('./data/df_history.csv')
    df_clustered, RF_decoding = initiate_clustering(df_history, profileId)
    # df_clustered.to_csv('./data/df_clustered.csv')
    df_forecast = conversion_rate(df_clustered, RF_decoding, profileId)
    # df_forecast.to_csv('./data/df_forecast.csv')
    df_bid_history_merge = merge_forecast_bid(df_campaign, df_adgroup, df_keyword, df_kw_history, df_forecast)
    # df_bid_history_merge.to_csv('./data/df_bid_history_merge.csv')
    df_slope_conv = get_slope_conv_value(df_campaign, df_history, df_kw_history, df_bid_history_merge, profileId)
    # df_slope_conv.to_csv('./data/df_slope_conv.csv')
    df_new_bid = update_new_bid(df_slope_conv, profileId)
    # df_new_bid.to_csv('./data/df_new_bid.csv')
    return df_new_bid


def convert_time(str_time):
    new_time = str(str_time)[0:4] + "-" + str(str_time)[4:6] + "-" + str(str_time)[6:]
    return new_time


def merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history, df_target, df_target_history):
    if len(df_campaign) == 0 or len(df_adgroup) == 0 or len(df_keyword) == 0 or \
            len(df_kw_history) == 0 or len(df_target) == 0 or len(df_target_history) == 0:
        return pd.DataFrame()

    # Append Keyword And target
    # df_keyword['targeting'] = df_keyword['keywordText']
    # df_keyword = df_keyword[['keywordId', 'adGroupId', 'campaignId', 'matchType', 'targeting']]
    df_keyword = df_keyword[['keywordId', 'adGroupId', 'campaignId', 'matchType']]
    # df_target['targeting'] = df_target['resolvedExpression']
    df_target['keywordId'] = df_target['targetId']
    df_target['matchType'] = ['-'] * len(df_target)
    df_target = df_target[['keywordId', 'campaignId', 'matchType']]
    # Add adgroup ID to targets DF
    adgroup_list = []
    for i in range(len(df_target)):
        targetID = df_target.iloc[i]['keywordId']
        df_target_adgroup = df_target_history[df_target_history['targetId'] == targetID]
        if len(df_target_adgroup) > 0:
            adgroup_list.append(df_target_adgroup.iloc[0]['adGroupId'])
        else:
            adgroup_list.append(0)
    df_target['adGroupId'] = adgroup_list
    df_target = df_target[df_target['adGroupId'] != 0]
    df_keyword = pd.concat([df_keyword, df_target])

    # Append Keyword History And target History
    df_kw_history = df_kw_history[['keywordId', 'adGroupId', 'campaignId', 'profileId', 'targeting', 'clicks', 'impressions',
                                   'cost', 'date', 'conversions', 'sales']]
    df_target_history['keywordId'] = df_target_history['targetId']
    df_target_history['targeting'] = df_target_history['targetingExpression']
    df_target_history["date"] = df_target_history["date"].apply(convert_time)
    df_target_history["date"] = pd.to_datetime(df_target_history["date"], format='%Y-%m-%d')
    df_target_history = df_target_history[['keywordId', 'adGroupId', 'campaignId', 'profileId', 'targeting', 'clicks', 'impressions',
                                           'cost', 'date', 'conversions', 'sales']]
    df_kw_history = pd.concat([df_kw_history, df_target_history])
    df_kw_history = df_kw_history[df_kw_history['clicks'] != 0]

    df_history = df_keyword.merge(df_adgroup, how='left', on=['adGroupId', 'campaignId'])
    df_history = df_history.merge(df_campaign, how='left', on='campaignId')
    df_history = df_history.merge(df_kw_history, how='left', on=['campaignId', 'adGroupId', 'keywordId'])
    df_history = df_history[df_history['clicks'] > 0]
    return df_history


def initiate_clustering(df_history, profileId):
    if len(df_history) > 0:
        df_clustered, RF_decoding = cluster.clustering(df_history, profileId)
        return df_clustered, RF_decoding
    else:
        return pd.DataFrame(), pd.DataFrame()


def conversion_rate(df_clustered, RF_decoding, profileId):
    if not len(df_clustered) == 0:
        if not os.path.exists("./data/" + profileId):
            os.mkdir("./data/" + profileId)
            os.mkdir("./data/" + profileId + "/prediction")
        kf, X_t = conversion_rate_estimation.initiate(df_clustered, './data/' + profileId)
        X_t['Leave'] = X_t.index
        X_t.to_csv('./data/x_t.csv')
        df_forecast = X_t.join(RF_decoding.set_index('Leave'), how='outer').drop_duplicates()
        df_forecast.to_csv("./data/" + profileId + "/prediction/df_forecast.csv")
        return df_forecast
    else:
        print("No data")
        return pd.DataFrame()


def merge_forecast_bid(df_campaign, df_adgroup, df_keyword, df_kw_history, df_forecast):
    if len(df_kw_history) == 0:
        return pd.DataFrame()
    df_kw_history = pd.merge(df_kw_history, df_campaign[['campaignId', 'campaignName']], on='campaignId')
    df_kw_history = pd.merge(df_kw_history, df_adgroup[['adGroupId', 'adGroupName']], on="adGroupId")
    df_kw_history = pd.merge(df_kw_history, df_keyword[['keywordId', 'matchType']], on="keywordId")
    df_bid = pd.merge(df_kw_history, df_forecast, how='left',
                      left_on=['campaignName', 'adGroupName', 'targeting', 'matchType'],
                      right_on=['campaignName', 'adGroupName', 'targeting', 'matchType'])
    return df_bid


def filter_avg_cpc(df_merge):
    clicks_sum_list, cost_sum_list = [], []
    for i in range(len(df_merge)):
        cur_date = df_merge.iloc[i]['date']
        past_date = cur_date - datetime.timedelta(days=30)
        campaignId = df_merge.iloc[i]['campaignId']
        df_filter = df_merge.loc[df_merge['campaignId'] == campaignId]
        df_filter = df_filter.loc[(df_filter['date'] <= cur_date) & (df_filter['date'] >= past_date)]
        click_sum = df_filter['clicks'].sum()
        if click_sum <= 0:
            adGroupId = df_merge.iloc[i]['adGroupId']
            df_filter = df_merge.loc[df_merge['adGroupId'] == adGroupId]
            df_filter = df_filter.loc[(df_filter['date'] <= cur_date) & (df_filter['date'] >= past_date)]
            click_sum = df_filter['clicks'].sum()
            if click_sum <= 0:
                keywordId = df_merge.iloc[i]['keywordId']
                df_filter = df_merge.loc[df_merge['keywordId'] == keywordId]
                df_filter = df_filter.loc[(df_filter['date'] <= cur_date) & (df_filter['date'] >= past_date)]
                click_sum = df_filter['clicks'].sum()
        clicks_sum_list.append(click_sum)
        cost_sum = df_filter['cost'].sum()
        cost_sum_list.append(cost_sum)

    df_merge['clicks_filter'] = clicks_sum_list
    df_merge['cost_filter'] = cost_sum_list
    return df_merge


def limit_bid_change(row):
    upper_limit = row.bid * (1 + 0.25)
    lower_limit = row.bid / (1 + 0.25)
    if row.update_bid > upper_limit:
        new_bid = upper_limit
    elif row.update_bid < lower_limit:
        new_bid = lower_limit
    else:
        new_bid = row.update_bid
    return new_bid


def get_price_adgroup_history(row):
    if row['sales30d'] == 0:
        return 0
    return row['sales30d'] / row['purchases30d']


def get_slope_conv_value(df_campaign, df_history, df_kw_history, df_bid_history_merge, profileId):
    # for every campaign, every adgroup, every target in df_bid get a + b
    if len(df_campaign) == 0 or len(df_history) == 0 or len(df_kw_history) == 0 or len(df_bid_history_merge) == 0:
        return pd.DataFrame()
    try:
        default_conv_val = df_history['sales'].sum() / df_history['conversions'].sum()
    except ZeroDivisionError:
        default_conv_val = 20

    ### Calculate Conversion values
    ## Get campaign Type first
    df_bid_history_merge['campaignType'] = df_bid_history_merge['campaignId'].apply(lambda x: df_campaign.loc[df_campaign['campaignId'] == x].iloc[0]['campaignType'])

    ## Get Conversion Value as a List
    conv_val_list = []
    df_adgroup_history = input_output.read_adgroup_history(profileId)
    df_adgroup_history.dropna(axis=0, inplace=True)
    df_adgroup_history['date'] = df_adgroup_history['date'].apply(lambda x: pd.Timestamp(x))
    df_adgroup_history['price'] = df_adgroup_history.apply(get_price_adgroup_history, axis=1)
    df_ads = input_output.get_ads(profileId)
    df_campaign_history = input_output.read_campaign_history(profileId)
    df_campaign_history['date'] = df_campaign_history['date'].apply(lambda x: pd.Timestamp(x))
    for i in range(len(df_bid_history_merge)):
        row = df_bid_history_merge.iloc[i]
        campaign_type = row['campaignType']

        if campaign_type == 'sponsoredBrands':
            check_idx_list = []
            for k in range(len(df_ads)):
                if df_ads.iloc[k]['active']:
                    check_idx_list.append(k)
            df_ads_ch = df_ads.loc[check_idx_list]
            df_adgroup_history_ch = df_adgroup_history.loc[(df_adgroup_history['adGroupId'].isin(list(df_ads_ch['adGroupId']))) & (df_adgroup_history['price'] != 0)]
            conv_value = df_adgroup_history_ch['price'].mean()

        else:
            df_campaign_history['date'] = df_campaign_history['date'].apply(lambda x: pd.Timestamp(x))
            if dataframe.last_n_days(df_campaign_history, 30)['conversions'].sum() >= 1:
                conv_value = (
                        dataframe.last_n_days(df_campaign_history, 30)['sales30d'].sum() /
                        dataframe.last_n_days(df_campaign_history, 30)['conversions'].sum())
            elif dataframe.last_n_days(df_adgroup_history, 30)['conversions'].sum() >= 1:
                conv_value = (
                        dataframe.last_n_days(df_adgroup_history, 30)['sales30d'].sum() /
                        dataframe.last_n_days(df_adgroup_history, 30)['conversions'].sum())
            else:
                conv_value = 0
        conv_val_list.append(conv_value)

    df_bid_history_merge['conv_value'] = conv_val_list
    df_bid_history_merge = filter_avg_cpc(df_bid_history_merge)

    df_bid_history_merge['avg_cpc'] = df_bid_history_merge['cost_filter'] / df_bid_history_merge['clicks_filter']
    df_bid_history = input_output.read_bid_history(profileId)
    df_bid_history_merge = pd.merge(df_bid_history_merge, df_bid_history[['keywordId', 'bid']], on=['keywordId'])

    df_bid_history_merge.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    df_bid_history_merge = df_bid_history_merge.loc[df_bid_history_merge['avg_cpc'] >= 0]

    df_bid_history_merge['slope'] = 1.0
    # count clicks group by campaign
    # Group the dataframe by campaign and count the number of clicks in each group
    click_counts = df_bid_history_merge.groupby(['campaignId', 'date'])["clicks"].transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_bid_history_merge.loc[mask, "slope"] = market_curve.market_curve(df_bid_history_merge)

    # count clicks group by adgroup
    # Group the dataframe by adgroup and count the number of clicks in each group
    click_counts = df_bid_history_merge.groupby(['adGroupId', 'date'])["clicks"].transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_bid_history_merge.loc[mask, "slope"] = market_curve.market_curve(df_bid_history_merge)

    # count clicks group by target
    click_counts = df_bid_history_merge.groupby(['targeting', 'date'])["clicks"].transform("count")

    # Create a mask indicating which rows should have their value in column a changed
    mask = click_counts > 5

    # Replace the values in column A where the mask is True with the slope
    df_bid_history_merge.loc[mask, "slope"] = market_curve.market_curve(df_bid_history_merge)

    return df_bid_history_merge


def update_new_bid(df_slope_merge, profileId):
    if len(df_slope_merge) == 0:
        return pd.DataFrame()
    df_keyword = input_output.get_keyword(profileId)
    df_slope_merge = pd.merge(df_slope_merge, df_keyword[['keywordId', 'target_acos']], on=['keywordId'])
    df_slope_merge['update_bid'] = df_slope_merge['target_acos'] * df_slope_merge['CR'] * df_slope_merge['conv_value'] / df_slope_merge['slope']

    df_slope_merge['new_bid'] = df_slope_merge.apply(limit_bid_change, axis=1)
    return df_slope_merge


def update_bid_to_db(df_new_bid, profileId):
    if len(df_new_bid) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df_keyword_db = mongo_db.read_collection_account_as_df('keywords', profileId, global_var.db)
    df_new_bid.sort_values(by='date', ascending=False, inplace=True)
    df_new_bid = pd.merge(df_new_bid, df_keyword_db[['keywordId', 'state']], on=['keywordId'])

    df_campaign = df_new_bid
    df_campaign_db = mongo_db.read_collection_account_as_df('campaigns', profileId, global_var.db)
    df_campaign.drop_duplicates(subset=['campaignId'], inplace=True)
    df_campaign = pd.merge(df_campaign, df_campaign_db[['campaignId', 'optimizing']], on=['campaignId'])

    df_adgroup = df_new_bid
    df_adgroup_db = mongo_db.read_collection_account_as_df('adgroups', profileId, global_var.db)
    df_adgroup.drop_duplicates(subset=['adGroupId'])
    df_adgroup = pd.merge(df_adgroup, df_adgroup_db[['adGroupId', 'optimizing']], on=['adGroupId'])

    df_keyword = df_new_bid
    df_keyword.drop_duplicates(subset=['keywordId'])
    df_keyword = pd.merge(df_keyword, df_keyword_db[['keywordId', 'optimizing']], on=['keywordId'])

    return df_campaign, df_adgroup, df_keyword


def update_into_db(df_update, account):
    if len(df_update) == 0:
        return
    req_body = []
    header = {
        'Content-Type': 'application/json',
        'Amazon-Advertising-API-ClientId': "amzn1.application-oa2-client.9249028043c04df085aafd96e8e23908",
        'Amazon-Advertising-API-Scope': account['profileId'],
        'Authorization': account['access_token'],
    }
    # url = "https://advertising-api.amazon.com/sp/keywords"
    # Version 2 url for sp update
    url = "https://advertising-api.amazon.com/v2/sp/keywords"
    for i in range(len(df_update)):
        if df_update.iloc[i]['optimizing'] and df_update.iloc[i]['new_bid']:
            req_body.append({
                "keywordId": int(df_update.iloc[i]['keywordId']),
                "state": df_update.iloc[i]['state'],
                "bid": round(float(df_update.iloc[i]['new_bid']) * 100) / 100
            })
            # response = requests.put(url, json=req_body, headers=header)
            # print(df_update.iloc[i]['keywordId'], '___Updated___',response.status_code)

    if len(req_body) > 0:
        # response = requests.put(url, json={"keywords": req_body}, headers=header)
        response = requests.put(url, json=req_body, headers=header)
        print(response.status_code)


if __name__ == "__main__":
    main()
