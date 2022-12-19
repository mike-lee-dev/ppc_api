# from models.conversion_rate_estimation import clustering
import pandas as pd
from utils import input_output, dataframe, mongo_db
from helpers import date_helpers
import global_var
from models import cluster, conversion_rate_estimation, object_structure, compute_bid, market_curve
from datetime import datetime, date
import math
import glob
import os


def main():  # optimize all accounts
    global_var.db = mongo_db.db_connect()
    accounts = mongo_db.read_collection_as_df('accounts', global_var.db)
    json_dict = {}
    for profileId in accounts['profileId']:
        print(f"{profileId}  --- Merge")
        json_dict[profileId] = optimize_account(profileId)
    return json_dict

def optimize_account(profileId):
    df_campaign = input_output.get_campaign(profileId)
    df_adgroup = input_output.get_adgroup(profileId)
    df_keyword = input_output.get_keyword(profileId)
    df_kw_history = input_output.read_keyword_history(profileId)
    df_price = input_output.get_price(profileId)
    df_history = merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history)
    df_clustered, RF_decoding = initiate_clustering(df_history, profileId)
    df_forecast = conversion_rate(df_clustered, RF_decoding, profileId)
    df_bid_history_merge = merge_forecast_bid(df_campaign, df_adgroup, df_keyword, df_kw_history, df_forecast)
    df_bid_conv = get_slope_conv_value(df_campaign, df_history, df_kw_history, df_bid_history_merge, profileId)
    if df_bid_conv is None:
        return {}
    return df_bid_conv.to_json(default_handler=str)
    # df_bid = compute_bid(df_bid)
    # df_bid_SP.to_csv(global_var.account + "/prediction/newbids_SP.csv")
    # update_bid_excel(df_bid_SP, 'SP')


def merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history):
    try:
        df_history = df_keyword.merge(df_adgroup, how='left', on=['adGroupId', 'campaignId'])
        df_history = df_history.merge(df_campaign, how='left', on='campaignId')
        df_history = df_history.merge(df_kw_history, how='left', on=['campaignId', 'adGroupId', 'keywordId'])
        return df_history
    except:
        return pd.DataFrame()


# df_history=df_history.merge(df_campaign, how='left', on='campaignId')


"""
def merge_no_date(df_bid, df_history, campaign_type, account):
	#add target without data
	#Get only the relevant columns from df_bid and drop duplicate rows
	if campaign_type=='SP':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['campaignName', 'adGroupName', 'Targeting', 'matchType']]#.drop_duplicates()
		#df_bid_part['Portfolio name']='Not grouped'
	if campaign_type=='SB':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['campaignName', 'Targeting', 'matchType']]
		#df_bid_part['Portfolio name']='Not grouped'
	if campaign_type=='SD':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['campaignName', 'adGroupName', 'Targeting']]#.drop_duplicates()
		#df_bid_part['Portfolio name']='-'
	df_bid_part['Currency']='USD'
	# append df_bid_no_click to a row of df_history to copy the structure of df_history, then get rid of this row by indexing
	df_bid_no_click = pd.concat([df_history.iloc[0:1,:],df_bid_part]).iloc[1:]
	max_Date = df_history['Date'].max()

	#Create a dictionary that fills na values for each column. NaN dates are replaced with the most recent date, and all
	#numerical columns are replaced with 0.
	dict_fillna = {'Date': max_Date}
	for i in df_bid_no_click.columns[7:]:
	    dict_fillna[i] = 0

	#Perform the replacement
	df_bid_final = df_bid_no_click.fillna(dict_fillna)

	#Append the final version of df_bid to df_history
	df_history = pd.concat([df_history, df_bid_final],ignore_index=True)#.reset_index(drop = True)
	df_history.Date = df_history.Date.dt.date
	df_history=date_helpers.format_date(df_history)
	df_history.to_csv(account+"/prediction/full_history.csv")
	return df_history
"""


def initiate_clustering(df_history, profileId):
    if len(df_history) > 0:
        df_clustered, RF_decoding = cluster.clustering(df_history, profileId)
        return df_clustered, RF_decoding
    else:
        return pd.DataFrame(), pd.DataFrame()
    # input_output.write_clustering(df_clustered, RF_decoding, account)
    # df_clustered=cluster.aggregate_by_node(RF_decoding, df_history)
    # return df_clustered, RF_decoding


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


def get_slope(row):
    df_campaign_history = input_output.read_campaign_history(row.profileId)
    count_clicks = dataframe.last_n_days(df_campaign_history, 5)['clicks'].count()
    df_bid_history = df_campaign_history
    # the default value for a=1
    if count_clicks <= 0:
        # if a campaign has more than 5 days with number of click>0: we take calculate a for the campaign
        # if an adgroup has more than 5 days with number of click>0: we take calculate a for the adgroup
        df_adgroup_history = input_output.read_adgroup_history(row.profileId)
        count_clicks = dataframe.last_n_days(df_adgroup_history, 5)['clicks'].count()
        df_bid_history = df_adgroup_history
        # if a keyword/target has more than 5 days with number of click>0: we take calculate a for the keyword/target
        if count_clicks <= 0:
            df_keyword_history = input_output.read_keyword_history(row.profileId)
            count_clicks = dataframe.last_n_days(df_keyword_history, 5)['clicks'].count()
            df_bid_history = df_keyword_history
            if count_clicks <= 0:
                df_target_history = input_output.read_target_history(row.profileId)
                count_clicks = dataframe.last_n_days(df_target_history, 5)['clicks'].count()
                df_bid_history = df_target_history
                if count_clicks <= 0:
                    return 1

    df_bid_history['avg_cpc'] = row.avg_cpc
    a = market_curve.market_curve(dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history), 'campaignId', row.campaignId).dropna(subset=['avg_cpc']))
    return a


def get_slope_conv_value(df_campaign, df_history, df_kw_history, df_bid_history_merge, profileId):
    # for every campaign, every adgroup, every target in df_bid get a + b
    if len(df_campaign) == 0 or len(df_history) == 0 or len(df_kw_history) == 0 or len(df_bid_history_merge) == 0:
        return pd.DataFrame()
    try:
        default_conv_val = df_history['sales'].sum() / df_history['conversions'].sum()
    except ZeroDivisionError:
        default_conv_val = 20

    ### Calculate AVG_CPC values
    ## Get campaign Type first
    df_bid_history_merge['campaignType'] = df_bid_history_merge['campaignName'].apply(lambda x: df_campaign.loc[df_campaign['campaignName'] == x].iloc[0]['campaignType'])

    ## Get Conversion Value as a List
    conv_val_list = []
    df_adgroup_history = input_output.read_adgroup_history(profileId)
    df_adgroup_history['date'] = df_adgroup_history['date'].apply(lambda x: pd.Timestamp(x))
    df_ads = input_output.get_ads(profileId)
    df_campaign_history = input_output.read_campaign_history(profileId)
    df_campaign_history['date'] = df_campaign_history['date'].apply(lambda x: pd.Timestamp(x))
    for i in range(len(df_bid_history_merge)):
        row = df_bid_history_merge.iloc[i]
        campaign_type = row['campaignType']

        if campaign_type == 'sponsoredBrands':
            df_ads_ch = df_ads.loc[df_ads['active'] == True]
            df_adgroup_history_ch = df_adgroup_history.loc[(df_adgroup_history['adGroupId'] in df_ads_ch['adGroupId']) and (df_adgroup_history['price'] != 0)]
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
    df_bid_history_merge.to_csv('./data/df_bid_history_merge_slope_conv_value.csv')

    ## Get Average CPC Value as a List
    df_keyword = input_output.get_keyword(profileId)
    avg_cpc_list = []
    for i in range(len(df_bid_history_merge)):
        row = df_bid_history_merge.iloc[i]
        df_keyword_ch = df_keyword.loc[df_keyword['keywordId'] == row['keywordId']]
        target_acos = df_keyword_ch.iloc[0]['target_acos']
        # target_acos = # from campaignId, adgroupId, profileId, keywordId but finally keywordId is last one
        # print(target_acos.isnull())
        if not target_acos or not row['CR']:
            avg_cpc = 0
        else:
            avg_cpc = target_acos * (row['CR'] * row['conv_value'])
        avg_cpc_list.append(avg_cpc)

    df_bid_history_merge['avg_cpc'] = avg_cpc_list

    df_bid_history = input_output.read_bid_history(profileId)
    df_bid_history_merge = pd.merge(df_bid_history_merge, df_bid_history[['keywordId', 'bid']], on=['keywordId'])
    df_bid_history_merge.to_csv('./data/df_bid_history_merge_slope_avg_cpc.csv')

    # # df_bid_history_merge['max_cpc'] = df_bid_history_merge.apply(cal_max_cpc, axis=1)

    count_clicks = dataframe.last_n_days(df_campaign_history, 5)['clicks'].count()
    df_bid_history = df_campaign_history
    slope_list = []
    for i in range(len(df_bid_history_merge)):
        row = df_bid_history_merge.iloc[i]
        check_slope = 0
        # the default value for a=1
        if count_clicks <= 0:
            # if a campaign has more than 5 days with number of click>0: we take calculate a for the campaign
            # if an adgroup has more than 5 days with number of click>0: we take calculate a for the adgroup
            count_clicks = dataframe.last_n_days(df_adgroup_history, 5)['clicks'].count()
            df_bid_history = df_adgroup_history
            check_slope = 0
            # if a keyword/target has more than 5 days with number of click>0: we take calculate a for the keyword/target
            if count_clicks <= 0:
                df_keyword_history = input_output.read_keyword_history(profileId)
                df_keyword_history['date'] = df_keyword_history['date'].apply(lambda x: pd.Timestamp(x))
                count_clicks = dataframe.last_n_days(df_keyword_history, 5)['clicks'].count()
                df_bid_history = df_keyword_history
                check_slope = 0
                if count_clicks <= 0:
                    df_target_history = input_output.read_target_history(profileId)
                    df_target_history['date'] = df_target_history['date'].apply(lambda x: pd.Timestamp(x))
                    count_clicks = dataframe.last_n_days(df_target_history, 5)['clicks'].count()
                    df_bid_history = df_target_history
                    check_slope = 0
                    if count_clicks <= 0:
                        check_slope = 1
                        slope_list.append(1)
        if check_slope != 1:
            df_bid_history['avg_cpc'] = df_bid_history_merge['avg_cpc']
            df_bid_history['bid'] = df_bid_history_merge['bid']
            a = market_curve.market_curve(dataframe.select_row_by_val(df_bid_history, 'campaignId', row['campaignId']).dropna(subset=['avg_cpc']))
            slope_list.append(a)
    df_bid_history_merge['slope'] = slope_list
    return df_bid_history_merge

    # loop over campaign, then adgroup, then target and add output (CV and slope) to the prediction file
    # for i, cam in df_campaign.iterrows():  # we could use itertuples to speed up the performance but we would need to have the same format for all campaigns
    #     # if cam['Product'] == 'Sponsored Display':
    #     #     if cam['Cost Type'] == 'vcpm':
    #     #         continue
    #     df_campaign_history = dataframe.select_row_by_val(df_history, 'campaignName', cam['campaignName'])
    #     df_campaign_bid_history = dataframe.select_row_by_val(df_bid_history, 'campaignId', str(cam['campaignId']))
    #     c = object_structure.Campaign(cam['campaignId'], cam['campaignName'], cam['state'], df_campaign_history,
    #                                   df_campaign_bid_history, 1., default_conv_val)
    #     dataframe.change_val_if_col_contains(df_bid, 'a', c.a, 'campaignId', str(c.campaign_id))
    #     dataframe.change_val_if_col_contains(df_bid, 'conv_value', c.conv_value, 'campaignId', str(c.campaign_id))
    #     dfadgr = dataframe.select_row_by_val(df_bid, 'Entity', "adGroupId", 'campaignId', str(c.campaign_id))
    #
    #     for j, adgr in dfadgr.iterrows():
    #         df_adgroup_history = dataframe.select_row_by_val(df_history, 'campaignName', c.campaign_name, 'adGroupName',
    #                                                          adgr['adGroupName'])
    #         df_adgroup_bid_history = dataframe.select_row_by_val(df_bid_history, 'campaignId', str(c.campaign_id),
    #                                                              'adGroupId', str(adgr['adGroupId']))
    #         a = object_structure.Adgroup(
    #             adgroup_id=adgr['adGroupId'],
    #             adgroup_name=adgr['adGroupName'],
    #             adgroup_status=adgr['state'],
    #             adgroup_bid=adgr['defaultBid'],
    #             df_adgroup_history=df_adgroup_history,
    #             df_adgroup_bid_history=df_adgroup_bid_history,
    #             a=c.a,
    #             conv_value=c.conv_value
    #         )
    #
    #         dataframe.change_val_if_col_contains(df_bid, 'a', a.a, 'adGroupId', str(a.adgroup_id))
    #         dataframe.change_val_if_col_contains(df_bid, 'conv_value', a.conv_value, 'adGroupId', str(a.adgroup_id))
    #         dfkw = dataframe.select_row_by_val(df_bid, 'Entity', "Keyword")
    #         dfpat = dataframe.select_row_by_val(df_bid, 'Entity', "Product Targeting")
    #         dfct = dataframe.select_row_by_val(df_bid, 'Entity', "Contextual Targeting")
    #         dfaud = dataframe.select_row_by_val(df_bid, 'Entity', "Audience Targeting")
    #         dftarget = pd.concat([dfkw, dfpat, dfct, dfaud])
    #         # dftarget=dftarget.concat()
    #         dftarget = dataframe.select_row_by_val(dftarget, 'campaignId', str(c.campaign_id), 'adGroupId',
    #                                                str(a.adgroup_id))
    #
    #         for k, target in dftarget.iterrows():
    #             target_name = target['targeting']
    #             df_target_history = dataframe.select_row_by_val(
    #                 df_history,
    #                 'campaignName', c.campaign_name,
    #                 'adGroupName', a.adgroup_name,
    #                 'targeting', target_name,
    #                 'matchType', target['matchType']
    #             )
    #             df_target_bid_history = dataframe.select_row_by_val(
    #                 df_bid_history,
    #                 'Target Id', str(target['Target Id']),
    #             )
    #             if target['Entity'] in ['Keyword', 'Product Targeting', 'Contextual Targeting', 'Audience Targeting']:
    #                 t = object_structure.Target(
    #                     match_type=target['matchType'],
    #                     target_bid=target['bid'],
    #                     target_name=target_name,
    #                     target_id=target['Target Id'],
    #                     target_status=target['State'],
    #                     df_target_history=df_target_history,
    #                     df_target_bid_history=df_target_bid_history,
    #                     a=a.a,
    #                     conv_value=a.conv_value
    #                 )
    #
    #                 dataframe.change_val_if_col_contains(df_bid, 'a', t.a, 'Target Id', t.target_id)
    #                 dataframe.change_val_if_col_contains(df_bid, 'conv_value', t.conv_value, 'Target Id', t.target_id)
    #
    # dftarget = input_output.read_target(path)
    # df_bid = pd.merge(df_bid, dftarget, how='left', left_on='Campaign Id', right_on='Campaign Id').drop_duplicates(ignore_index=True)
    #
    # df_bid['new_bid'] = df_bid['target_acos'] * df_bid['CR'] * df_bid['conv_value'] / df_bid['a']
    # df_bid['new_bid'] = df_bid.apply(lambda x: limit_bid_change(x['Bid'], x['new_bid'], 0.25), axis=1)
    # df_bid['new_bid'] = df_bid.apply(lambda x: valid_bid(x['new_bid']), axis=1)
    # return df_bid


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


def update_bid_excel(df_bid, account):
    # loop over excel file and change bids
    if optimized:
        newbid = dataframe.select_row_by_val(df_bid, 'Target Id', targetID)['new_bid'].values[0]
        if math.isnan(newbid) == False:
            input_output.write_xls_bid(wb, ws, row[0].row, str(round(newbid, 2)), campaign_type)
    input_output.save_xls_bid(wb, account)


if __name__ == "__main__":
    main()
