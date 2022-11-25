import pandas as pd
import numpy as np
from helpers import date_helpers
from utils import dataframe, mongo_db, input_output
from models import cluster, conversion_rate_estimation, object_structure, compute_bid
import ppcoptimizer
import shutil
import global_var

db = mongo_db.db_connect()

def collection(name):
    col = db[name]
    # getting documents
    cursor = col.find({})
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    df = df.convert_dtypes()
    return df

df_campaign=collection('campaigns')
df_adgroup=collection('adgroups')
df_keyword=collection('keywords')
df_kw_history=collection('keyword_history')
df_bid_history=collection('bids_history')
df_price=collection('price_reports')

df_history=ppcoptimizer.merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history)

df_clustered, RF_decoding = cluster.clustering(df_history)
print(RF_decoding)
