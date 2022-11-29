# if a keyword/target has too fee data we fall back to adgroup default
# if a adgroup has too fee data we fall back to campaign default
# if a campaign has too fee data we fall back to account default
from models import market_curve
from models.compute_bid import compute_bid
from utils import dataframe
from utils import input_output
from dataclasses import dataclass, InitVar, field
import pandas as pd
import numpy as np
import global_var
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses

# maybe it would be cleaner to use a function to create objects:
# https://www.youtube.com/watch?v=TZ5DEugWLp8

class platform:
    min_bid = 0.02


# currency=$


@dataclass
class portfolio:
    target: float

    def __post_init__(self):
        self.target = input_output.read_target()


@dataclass
# @dataclass(kw_only=True)
class _CampaignBase:
    campaign_id: str
    campaign_name: str
    campaign_status: bool
    # df_campaign_history:InitVar[pd.core.frame.DataFrame]
    df_campaign_history: pd.core.frame.DataFrame
    df_campaign_bid_history: pd.core.frame.DataFrame  #:InitVar[pd.core.frame.DataFrame]
    a: float  # = field(init=False)
    # b: float# = field(init=False)
    conv_value: float  # = field(init=False)

    def __post_init__(self):  # , df_history, df_bid_history):
        # aggregate by date then count datapoint
        self.get_conversion_value(self.df_campaign_history, self.df_campaign_bid_history)
        try:
            self.get_slope(self.df_campaign_bid_history)
        except ValueError:
            pass

    def get_slope(self, df_bid_history):
        count_clicks = dataframe.last_n_days(df_bid_history, 30)['clicks'].count()
        if (count_clicks) <= 4:
            self.a = 1
            self.b = 0
        else:
            try:
                print(dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history), 'campaignId',
                                                  self.campaign_id).dropna(subset=['avg_cpc']))
                self.a = market_curve.market_curve(
                    dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history), 'campaignId',
                                                self.campaign_id).dropna(subset=['avg_cpc']))

            except ValueError:
                self.a = 1.
            self.b = 0
            print(f"the equation is {self.a} * maxcpc {self.b}")

    def get_conversion_value(self, df_history, df_bid_history):
        if df_bid_history['Product'].iloc[0] == 'Sponsored Brands':
            if dataframe.last_n_days(df_history, 30)['Conversions'].sum() >= 1:
                self.conv_value = (
                        dataframe.last_n_days(df_history, 30)['Sales'].sum() /
                        dataframe.last_n_days(df_history, 30)['Conversions'].sum())
            else:
                pass  # account average
        # self.conv_value=input_output.active_listing(global_var.path)['price'].mean()

        else:
            SKUs = dataframe.select_row_by_val(df_bid_history, 'Entity', 'Product Ad')
            try:
                listing = input_output.get_price(global_var.path)
                self.conv_value = SKUs.merge(listing, how='left', left_on='SKU', right_on='seller-sku')['price'].mean()
            except FileNotFoundError:
                if dataframe.last_n_days(df_history, 30)['Conversions'].sum() > 0:
                    self.conv_value = (
                            dataframe.last_n_days(df_history, 30)['Sales'].sum() /
                            dataframe.last_n_days(df_history, 30)['Conversions'].sum())
                else:
                    pass
        print(f"the conversion value is {self.conv_value}")


@dataclass
# @dataclass(kw_only=True)
class _CampaignDefaultBase:
    optimized: bool = False


# a: float = 1
# b: float = 0
# conv_value: float  = 100

@dataclass
# @dataclass(kw_only=True)
class Campaign(_CampaignDefaultBase, _CampaignBase):
    pass


@dataclass
# @dataclass(kw_only=True)
class _AdgroupBase():  # (_CampaignBase):
    adgroup_id: str
    adgroup_name: str
    adgroup_status: bool
    adgroup_bid: float
    df_adgroup_history: pd.core.frame.DataFrame  #:InitVar[pd.core.frame.DataFrame]
    df_adgroup_bid_history: pd.core.frame.DataFrame  #:InitVar[pd.core.frame.DataFrame]
    a: float  # = field(init=True)
    # b: float = field(init=True)
    conv_value: float  # = field(init=True)

    def __post_init__(self):  # , df_history,df_bid_history):
        self.get_conversion_value(self.df_adgroup_history, self.df_adgroup_bid_history)
        try:
            self.get_slope(self.df_adgroup_bid_history)
        except ValueError:
            pass

    def get_slope(self, df_bid_history):
        count_clicks = dataframe.last_n_days(df_bid_history, 30)['clicks'].count()
        if (count_clicks) <= 4:  # astype(bool) #we need 10 points to fit a straightline and we have the origin
            pass
        # used value from campaign
        else:
            self.a = market_curve.market_curve(
                dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history), 'Ad Group Id',
                                            self.adgroup_id).dropna(subset=['avg_cpc']))
            self.b = 0
            print(f"the equation is {self.a} * maxcpc {self.b}")

    def get_conversion_value(self, df_history, df_bid_history):
        if df_bid_history.empty:  # for SB
            pass
        # if df_bid_history['Product'].iloc[0]=='Sponsored Brands':
        # 	return
        else:
            if df_bid_history['Product'].iloc[0] == 'Sponsored Brands':
                pass
            else:
                SKUs = dataframe.select_row_by_val(df_bid_history, 'Entity', 'Product Ad')
                prices = input_output.get_price(global_var.path)
                print(prices)

                if not prices.empty:
                    if len(SKUs['SKU'].value_counts()) > 0:
                        self.conv_value = SKUs.merge(prices, how='left', left_on='SKU', right_on='seller-sku')[
                            'price'].mean()  # we should filter only active SKUs
                    else:
                        try:
                            self.conv_value = SKUs.merge(prices, how='left', left_on='ASIN', right_on='asin1')[
                                'price'].mean()  # for vendor
                        except:
                            pass
                else:
                    try:
                        self.conv_value = (
                                dataframe.last_n_days(df_history, 30)['Sales'].sum() /
                                dataframe.last_n_days(df_history, 30)['Conversions'].sum())
                    except:
                        pass
        print(f"the conversion value is {self.conv_value}")

    def get_status(self) -> bool:
        if self.campaignstatus == 'enabled' & self.adgroup_status == 'enabled':
            self.adg_status = 'enabled'
        else:
            self.adg_status = 'disabled'


@dataclass
# @dataclass(kw_only=True)
class _AdgroupDefaultBase():  # (_CampaignDefaultBase):
    optimized: bool = False


@dataclass
# @dataclass(kw_only=True)
class Adgroup(_AdgroupDefaultBase, _AdgroupBase):  # (Campaign, _AdgroupDefaultBase, _AdgroupBase):
    def bid(risk_level):
        bid = compute_bid(self.target, self.a, self.conv_rate, self.conv_value, self.covar, self.risk_level)


@dataclass
# @dataclass(kw_only=True)
class _TargetBase():  # (_AdgroupBase):
    target_name: str
    target_id: str
    target_status: bool
    match_type: str
    target_bid: float
    df_target_history: pd.core.frame.DataFrame  #:InitVar[pd.core.frame.DataFrame]
    df_target_bid_history: pd.core.frame.DataFrame  #:InitVar[pd.core.frame.DataFrame]
    a: float
    conv_value: float

    def __post_init__(self):
        try:
            self.get_slope(self.df_target_bid_history)
        except ValueError:
            pass

    def get_status(self):
        if self.campaign_status == 'enabled' & self.adgroup_status == 'enabled' & self.target_status == 'enabled':
            self.status = 'enabled'
        else:
            self.status = 'disabled'

    def get_conv_rate():
        df = read_kalman_state()
        # cond=df['campaignId']==self.campaign_id & df['Ad Group Name']==self.adgroup_name & df['Targeting']==self.targeting & df['Match Type'] == self.match_type
        cond = df['campaignId'] == self.campaign_id & df['Ad Group Id'] == self.adgroup_id & df[
            'Targeting'] == self.targeting & df['Match Type'] == self.match_type
        self.conv_rate = df.where(cond)  # would be cleaner to have targetingID

    def get_slope(self, df_bid_history):
        count_clicks = dataframe.last_n_days(df_bid_history, 30)['clicks'].count()
        if (count_clicks) <= 4:  # astype(bool)
            pass
        # print(f"Not enough clicks: {clicks}")
        # used value from adgroup
        else:
            # self.a=market_curve.market_curve(dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history),'Keyword or Product Targeting',self.target_name).dropna(subset=['Bid','avg_cpc']))
            self.a = market_curve.market_curve(
                dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history), 'Targeting',
                                            self.target_name).dropna(subset=['avg_cpc']))

            # print(self.a)
            # if self.target_name =='+copper +hand':
            # 	bids=dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history),'Target Id',self.target_id).dropna(subset=['avg_cpc'])['Bid']
            # 	plt.scatter(dataframe.select_row_by_val(dataframe.get_biddable_object(df_bid_history),'Target Id',self.target_id).dropna(subset=['avg_cpc'])['avg_cpc'],bids)
            # 	plt.scatter(bids*self.a, bids)
            # 	plt.show()

            self.b = 0
            print(f"the equation is {self.a} * maxcpc {self.b}")


@dataclass
# @dataclass(kw_only=True)
class _TargetDefaultBase():  # (_AdgroupDefaultBase):
    optimized: bool = False


@dataclass
# @dataclass(kw_only=True)
class Target(_TargetDefaultBase, _TargetBase):  # (Adgroup,_TargetDefaultBase, _TargetBase):
    def get_bid(self):
        if self.optimized == False:
            pass
        else:
            # self.bid=...
            pass
