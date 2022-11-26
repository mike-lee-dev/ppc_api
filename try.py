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
print("*******************")
print(df_campaign)

df_adgroup=collection('adgroups')
print("*******************")
print(df_adgroup)

df_keyword=collection('keywords')
print("*******************")
print(df_keyword)

df_kw_history=collection('keyword_history')
print("*******************")
print(df_kw_history)

df_history=ppcoptimizer.merge_history(df_campaign, df_adgroup, df_keyword, df_kw_history)
print("*******************")
print(df_history)

df_clustered, RF_decoding = cluster.clustering(df_history)
print("*******CLUSTERED************")
print(df_clustered)

# kf, X_t = conversion_rate_estimation.initiate(df_clustered, 'testdata/')
# print('test')
# print(X_t)
# print("-------------------------")
# print(kf)