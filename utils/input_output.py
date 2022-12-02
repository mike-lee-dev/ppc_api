import pandas as pd
import numpy as np
from helpers import date_helpers
from utils import dataframe, mongo_db
import shutil
import global_var


# account Objects
def get_campaign(profileId):
    df_campaign = mongo_db.read_collection_account_as_df('campaigns', profileId, global_var.db)
    return df_campaign


def read_campaign_history(profileId):
    df_campaign_history = mongo_db.read_collection_account_as_df('campaign_history', profileId, global_var.db)
    if len(df_campaign_history) > 0:
        df_campaign_history['conversions'] = df_campaign_history['purchases30d']
    return df_campaign_history


def get_adgroup(profileId):
    df_adgroup = mongo_db.read_collection_account_as_df('adgroups', profileId, global_var.db)
    return df_adgroup


def read_adgroup_history(profileId):
    df_adgroup_history = mongo_db.read_collection_account_as_df('adgroup_history', profileId, global_var.db)
    if len(df_adgroup_history) > 0:
        df_adgroup_history['conversions'] = df_adgroup_history['purchases30d']
    return df_adgroup_history


def get_ads(profileId):
    df_ads = mongo_db.read_collection_account_as_df('ads', profileId, global_var.db)
    return df_ads


def get_keyword(profileId):
    df_keyword = mongo_db.read_collection_account_as_df('keywords', profileId, global_var.db)
    return df_keyword


def read_bid_history(profileId):
    df_bid_history = mongo_db.read_collection_account_as_df('bids_history', profileId, global_var.db)
    return df_bid_history


# Performance report
def read_keyword_history(profileId):
    df_history = mongo_db.read_collection_account_as_df('keyword_history', profileId, global_var.db)
    if len(df_history) > 0:
        df_history['conversions'] = df_history['purchases30d']
        df_history['sales'] = df_history['sales30d']
    return df_history


def read_target_history(profileId):
    df_history = mongo_db.read_collection_account_as_df('target_history', profileId, global_var.db)
    if len(df_history) > 0:
        df_history['conversions'] = df_history['attributedConversions30d']
        df_history['sales'] = df_history['attributedSales30d']
    return df_history


"""
#Search Querry Report
def read_sqr(account):
	account_file=account + '/raw/Sponsored Products Search term report.xlsx'
	df=pd.read_excel(io=account_file,sheet_name='Sponsored Product Search Term R',header=0, engine="openpyxl")	
	return df
"""


# read Active Listing Report - We need it to get the price

def get_price(profileId):
    df_price = mongo_db.read_collection_account_as_df('price_reports', profileId, global_var.db)
    if len(df_price) > 0:
        df_price = df_price[['sku', 'adGroupId', 'adId', 'asin', 'campaignId', 'price']]
        return df_price[df_price['price'] > 0]
    else:
        return df_price


# Clustering
def read_clustering(account):
    dfclustered = pd.read_csv(account + "/prediction/dfclustered.csv")
    RF_decoding = pd.read_csv(account + "/prediction/RF_decoding.csv")
    return dfclustered, RF_decoding


def write_clustering(dfclustered, RF_decoding, account):
    dfclustered.to_csv(account + "/prediction/dfclustered.csv")
    RF_decoding.to_csv(account + "/prediction/RF_decoding.csv", index=False)


def read_category_listing_report(account):
    account_file = account + '/raw/Category+Listings+Report.xlsx'
    df = pd.read_excel(io=account_file, sheet_name='Template', header=1, engine="openpyxl")
    return df


"""
def valid_price():
	if today> sales_from_date and today<sale_end_date:
		return df['sale_price']
	else:
		return df['standard_price']
"""


# Settings
def read_target(account):
    account_file = account + '/raw/target.xlsx'
    df = pd.read_excel(account_file, sheet_name='target', header=0, engine="openpyxl", converters={'Campaign Id': str})
    # df['Campaign Id']='\'' + df['Campaign Id']
    return df


def read_link(account):
    account_file = account + '/raw/link-manual-manual.xlsx'
    df = pd.read_excel(account_file, header=0, engine="openpyxl",
                       converters={'Ad Group source Id': str, 'Ad Group destination Id': str})
    df['Ad Group source Id'] = '\'' + df['Ad Group source Id'].fillna(0).astype(str)
    df['Ad Group destination Id'] = '\'' + df['Ad Group destination Id'].fillna(0).astype(str)
    return df


# Optimization
def read_kalman_state(account):
    R = np.loadtxt(account + "/prediction/R.txt")
    Q = np.loadtxt(account + "/prediction/Q.txt")
    P = np.loadtxt(account + "/prediction/P.txt")
    H = np.loadtxt(account + "/prediction/H.txt")
    F = np.loadtxt(account + "/prediction/F.txt")
    x = pd.read_csv(account + "/prediction/x.csv", squeeze=True)
    return R, Q, P, H, F, x


def write_kalman_state(R, Q, P, H, F, x, path):
    np.savetxt(path + "/prediction/R.txt", R)
    np.savetxt(path + "/prediction/Q.txt", Q)
    np.savetxt(path + "/prediction/P.txt", P)
    np.savetxt(path + "/prediction/H.txt", H)
    np.savetxt(path + "/prediction/F.txt", F)
    x.to_csv(path + "/prediction/x.csv")


# Prediction by date
def append_state_history(date, X_t, M_t, K_t, P_t, Q_t, R_t, S_t, y_t, account):
    X_t['Date'] = date
    try:
        x_hist = pd.read_csv(account + "/prediction/x_hist.csv", index_col=0)
    except:
        x_hist = X_t
    dfmerge = X_t.join(M_t, lsuffix='_pred', rsuffix='_mes')
    dfmerge = dfmerge.join(K_t)
    dfmerge = dfmerge.join(P_t)
    dfmerge = dfmerge.join(Q_t)
    dfmerge = dfmerge.join(R_t)
    dfmerge = dfmerge.join(S_t)
    dfmerge = dfmerge.join(y_t)
    x_hist = pd.concat([x_hist, dfmerge])
    x_hist.drop_duplicates(inplace=True)
    x_hist.to_csv(account + "/prediction/x_hist.csv")


# Bids
def backup_bid_xls(account):
    oldaccount = account + '/raw/bulk.xlsx'
    newaccount = account + '/output/bulk_to_upload.xlsx'
    shutil.copyfile(oldaccount, newaccount)


def open_wb(account, file='/output/bulk_to_upload.xlsx'):
    account_file = account + file
    wb_obj = openpyxl.load_workbook(account_file)
    return wb_obj


def open_ws(wb_obj, campaign_type):
    if campaign_type == 'SP':
        ws_name = 'Sponsored Products Campaigns'
    if campaign_type == 'SB':
        ws_name = 'Sponsored Brands Campaigns'
    if campaign_type == 'SD':
        ws_name = 'Sponsored Display Campaigns'
    ws_obj = wb_obj[ws_name]
    return ws_obj


def read_xls_bid(ws_obj, column, row):
    cell_obj = ws_obj.cell(row=row, column=column)
    return cell_obj.value


"""
def write_xls_bid(wb_obj, ws_obj, row, value, campaign_type):
	if campaign_type=='SP':
		column=28
	if campaign_type=='SB':
		#column=19
		column=22
	if campaign_type=='SD':		
		#column=20
		column=26
	ws_obj.cell(row=row, column=column).value=value
	ws_obj.cell(row=row, column=3).value='update'

def save_xls_bid(wb_obj, account, filename='/output/bulk_to_upload.xlsx'):
	wb_obj.save(account + filename)
"""
