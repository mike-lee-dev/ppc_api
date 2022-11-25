import pandas as pd
import numpy as np
from helpers import date_helpers
from utils import dataframe, mongo_db
import shutil
import global_var

db = mongo_db.db_connect()
account = '637e9e692b6363ad27850547'
df_price=mongo_db.read_collection_account_as_df('price_reports', account, db)
#get price for each SKU

df_price.groupby('sku')
print(df_price[['sku', 'adGroupId', 'adId', 'asin', 'campaignId', 'price']])
print(df_price[df_price['price'] > 0])
