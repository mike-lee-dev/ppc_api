import pandas as pd
from ..helpers import date_helpers
from datetime import datetime

def get_cpc(df):
	df['avg_cpc']=df['Spend'].divide(df['Clicks'].astype('float'))
	return df

def select_row_by_val(df,col1,val1,*args):
	cond=df[col1] == val1
	if args:
		count=0
		for arg in args:
			if (count%2)==0:
				col=args[count]
			else:
				val=args[count]
				cond= cond & (df[col]==val)
			count +=1
		#cond=(df[col1] == val1) & (df[args[0]] == args[1])
	if df.empty:
		print('DataFrame parse to the function is empty!')
	if df.loc[cond].empty:
		print('Dataframe after selection is empty!')
	return df.loc[cond]


def change_val_by_cond(df, colchange, valuechange, col1, val1, *args):
	cond=df[col1] == val1
	if args:
		count=0
		for arg in args:
			if (count%2)==0:
				col=args[count]
			else:
				val=args[count]
				cond= cond & (df[col]==val)
			count +=1
		
	df.loc[cond, colchange] = valuechange
	if df.empty:
		print('DataFrame parse to the function is empty!')
	if df.loc[cond].empty:
		print('Dataframe after selection is empty!')
	return df

def change_val_if_col_contains(df, colchange, valuechange, colcheck, string):
	df.loc[df[colcheck].str.contains(string), colchange] = valuechange
	return df
	

def get_biddable_object(df):
	return df[df['Entity'].isin(['Keyword', 'Product Targeting', 'Audience Targeting'])]

def add_date_dataframe(df,date):
	df['Date']=date
	#df['Date']=pd.to_datetime(df['Date'])#.dt.date
	return df

def join_bid_history(df_bid, dfnode):
	pass



def last_n_days(df,n):	
	today = pd.to_datetime("today")
	begin= pd.to_datetime(today- pd.Timedelta(days=n))
	try:
		mask=((df['Date'] > begin))# & (df['Date'] <= today))
	except KeyError:
		df=df.reset_index()
		print(df)
		mask=((df['Date'] > begin))
	df=df.loc[mask]
	if df.empty:
		print('DataFrame is empty!')
		print(f"Original DataFrame{df}")
	return df.loc[mask]

def to_datetime64(df):	
	#convert date from datetime to datetime64
	df['Date']=df['Date'].astype('datetime64')
	return df
