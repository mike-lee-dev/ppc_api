# from models.conversion_rate_estimation import clustering
import pathlib
import pandas as pd

from .utils import input_output, dataframe
from .helpers import date_helpers
from .models import cluster, conversion_rate_estimation, object_structure
from datetime import datetime, date

import math


def main():
    ACOStarget = .30

    # determine slope
    path_bid = './data/raw/bulk.xlsx'
    df_bid = input_output.read_bid('SP', path_bid)
    input_output.append_bid_history_with_date()

    # Forecast conversion rates

    df_history = input_output.read_report()
    print(f"Account History: \n{df_history} ")

    # add target without data
    df_no_clicks = df_history.merge(
        df_bid,
        how='left',
        left_on=['Campaign Name', 'Ad Group Name', 'Targeting', 'Match Type'],
        right_on=['Campaign', 'Ad Group', 'Keyword or Product Targeting', 'Match Type'],
        indicator=True
    )
    df_no_clicks = df_no_clicks[df_no_clicks['_merge'] == 'left_only'].drop_duplicates(ignore_index=True)
    df_no_clicks['Date'] = df_history.iloc[-1]['Date']
    df_no_clicks = dataframe.to_datetime64(df_no_clicks)
    _df_no_clicks = df_no_clicks[
        ['Date', 'Portfolio name', 'Campaign Name', 'Ad Group Name', 'Targeting', 'Match Type']].copy()
    print(df_history)
    df_history = pd.concat([df_history, _df_no_clicks], ignore_index=True).fillna(0)
    print(df_history)

    try:
        dfclustered, RF_decoding = input_output.read_clustering()
    except:
        dfclustered, RF_decoding = cluster.clustering(df_history)
        input_output.write_clustering(dfclustered, RF_decoding)
    df_clustered = cluster.aggregate_by_node(RF_decoding, df_history)

    kf, X_t = conversion_rate_estimation.initate(df_clustered)
    print(f"DF cluster: \n{df_clustered} ")
    M_t = conversion_rate_estimation.get_kalman_measurment(X_t, dataframe.last_n_days(df_clustered, 7))

    conversion_rate_estimation.update(kf, M_t)
    conversion_rate_estimation.predict(kf, M_t, X_t)
    print(f"forecast {kf.x}")
    print(f"forecast {X_t}")

    df_forecast = X_t.join(RF_decoding.set_index('Leave'))
    print(df_forecast)

    df_bid_history = dataframe.to_datetime64(input_output.read_bid_history('SP'))
    print(f"Account Bid history: \n{df_bid_history}")
    # https://towardsdatascience.com/apply-function-to-pandas-dataframe-rows-76df74165ee4
    default_conv_val = df_history['7 Day Total Sales '].sum() / df_history['7 Day Total Orders (#)'].sum()
    df_bid = pd.merge(df_bid, df_forecast, how='left',
                      left_on=['Campaign', 'Ad Group', 'Keyword or Product Targeting', 'Match Type'],
                      right_on=['Campaign_Name', 'Ad_Group_Name', 'Targeting', 'Match_Type'])
    print(df_forecast)
    print(df_bid)

    # for every campaign, every adgroup, every target in df_bid get a + b
    dfcam = dataframe.select_row_by_val(df_bid, 'Record Type', "Campaign")
    # loop over campaign, then adgroup, then target and add output (CV and slope) to the prediction file
    for i, cam in dfcam.iterrows():  # we could use itertuples to speed up the performance but we would need to have the same format for all campaigns
        print(f"Campaign Name: {cam['Campaign']}")  # 5

        df_campaign_history = dataframe.select_row_by_val(df_history, 'Campaign Name', cam['Campaign'])
        df_campaign_bid_history = dataframe.select_row_by_val(df_bid_history, 'Campaign', cam['Campaign'])

        c = object_structure.Campaign(cam['Campaign ID'], cam['Campaign'], cam['Campaign Status'], df_campaign_history,
                                      df_campaign_bid_history, 1., default_conv_val)
        print(c.a)

        dfadgr = dataframe.select_row_by_val(df_bid, 'Record Type', "Ad Group", 'Campaign ID', c.campaign_id)
        for j, adgr in dfadgr.iterrows():
            print(f"Campaign name: {c.campaign_name}")
            print(f"Adgroup Name: {adgr['Ad Group']}")  # 11
            df_adgroup_history = dataframe.select_row_by_val(df_history, 'Campaign Name', c.campaign_name,
                                                             'Ad Group Name', adgr['Ad Group'])
            df_adgroup_bid_history = dataframe.select_row_by_val(df_bid_history, 'Campaign', c.campaign_name,
                                                                 'Ad Group', adgr['Ad Group'])
            a = object_structure.Adgroup(
                # campaign_id=c.campaign_id,
                # campaign_name=c.campaign_name,
                # campaign_status=c.campaign_status,
                # df_campaign_history=df_campaign_history,
                # df_campaign_bid_history=df_campaign_history,
                adgroup_name=adgr['Ad Group'],
                adgroup_status=adgr['Ad Group Status'],
                adgroup_bid=adgr['Max Bid'],
                df_adgroup_history=df_adgroup_history,
                df_adgroup_bid_history=df_adgroup_bid_history,
                a=c.a,
                conv_value=c.conv_value
            )

            dfkw = dataframe.select_row_by_val(df_bid, 'Record Type', "Keyword")
            dfpat = dataframe.select_row_by_val(df_bid, 'Record Type', "Product Targeting")
            dftarget = dfkw.append(dfpat)
            dftarget = dataframe.select_row_by_val(dftarget, 'Campaign ID', c.campaign_id, 'Ad Group', a.adgroup_name)
            for k, target in dftarget.iterrows():
                print(f"Campaign name: {c.campaign_name}")
                print(f"Adgroup name: {a.adgroup_name}")
                print(f"Target :{target['Keyword or Product Targeting']}")  # 13
                if target['Match Type'] in ['Targeting Expression', 'Targeting Expression Predefined', 'broad',
                                            'phrase', 'exact']:
                    df_target_history = dataframe.select_row_by_val(
                        df_history,
                        'Campaign Name', c.campaign_name,
                        'Ad Group Name', a.adgroup_name,
                        'Targeting', target['Keyword or Product Targeting'],
                        'Match Type', target['Match Type']
                    )
                    df_target_bid_history = dataframe.select_row_by_val(
                        df_bid_history,
                        'Campaign', c.campaign_name,
                        'Ad Group', a.adgroup_name,
                        'Keyword or Product Targeting', target['Keyword or Product Targeting'],
                        'Match Type', target['Match Type'])
                    t = object_structure.Target(
                        # campaign_id=c.campaign_id,
                        # campaign_name=c.campaign_name,
                        # campaign_status=c.campaign_status,
                        # df_campaign_history=df_campaign_history,
                        # df_campaign_bid_history=df_campaign_bid_history,
                        # adgroup_name=a.adgroup_name,
                        # adgroup_status=a.adgroup_status,
                        # adgroup_bid=a.adgroup_bid,
                        # df_adgroup_history=df_adgroup_history,
                        # df_adgroup_bid_history=df_adgroup_bid_history,
                        match_type=target['Match Type'],
                        target_bid=target['Max Bid'],
                        target_name=target['Keyword or Product Targeting'],
                        target_status=target['Status'],
                        df_target_history=df_target_history,
                        df_target_bid_history=df_target_bid_history,
                        a=a.a,
                        conv_value=a.conv_value
                    )
                    print(t.a)
                    dataframe.change_val_by_cond(
                        df_bid, 'a', t.a, 'Campaign ID', c.campaign_id, 'Ad Group', a.adgroup_name,
                        'Keyword or Product Targeting', t.target_name,
                        'Match Type', t.match_type
                    )
                    dataframe.change_val_by_cond(
                        df_bid, 'conv_value', t.conv_value, 'Campaign ID', c.campaign_id, 'Ad Group', a.adgroup_name,
                        'Keyword or Product Targeting', t.target_name,
                        'Match Type', t.match_type
                    )
                    print(t.conv_value)

    # acos=spend/revenue= click*cpc/(CR*CV*click)=cpc/(CR*CV)->cpc=acos*(CR*CV)=>maxcpc=acos*(CR*CV)/a
    dftarget = input_output.read_target()
    df_bid = pd.merge(df_bid, dftarget, how='left', left_on='Campaign', right_on='campaign')
    print(df_bid)
    df_bid['new_bid'] = df_bid['target_acos'] * df_bid['CR'] * df_bid['conv_value'] / df_bid['a']
    df_bid['new_bid'] = df_bid['new_bid'].replace(0, 0.02)
    print(f"New bids: {df_bid}")
    # loop over excel file and change bids
    input_output.backup_bid_xls()
    wb = input_output.open_wb()
    ws = input_output.open_ws(wb, 'SP')
    # rows = ws.getCells().getMaxDataRow(min_row=2)
    for row in ws.iter_rows(min_row=2):
        recordtype = row[1].value
        campaign = row[3].value
        adgroup = row[9].value
        oldbid = row[10].value
        target = row[11].value
        matchtype = row[13].value

        if oldbid != "" and (recordtype == 'Keyword' or recordtype == 'Product Targeting'):
            print(f"old bid: {oldbid}")
            newbid = dataframe.select_row_by_val(df_bid, 'Campaign', campaign, 'Ad Group', adgroup,
                                                 'Keyword or Product Targeting', target, 'Match Type', matchtype)[
                'new_bid'].values[0]
            optimized = dataframe.select_row_by_val(df_bid, 'Campaign', campaign, 'Ad Group', adgroup,
                                                    'Keyword or Product Targeting', target, 'Match Type', matchtype)[
                'optimized'].values[0]
            if optimized:
                print(f"new bid: {str(round(newbid, 2))}")
                print(f"row: {row[0].row}")
                print(
                    f"Could not find bid for campaign: {campaign}, adgroup: {adgroup}, target: {target}, matchtype: {matchtype}")
                df_bid.to_csv("./data/prediction/newbids.csv")
                if math.isnan(newbid) == False:
                    input_output.write_xls_bid(wb, ws, row[0].row, str(round(newbid, 2)))


# for cell in ws['K']:
# bid=input_output.read_xls_bid(ws, 11, i)
#	print(cell.value)


if __name__ == "__main__":
    main()
