import pandas as pd
import numpy as np
import os
from . import dataframe



def create_keyword_pat(product, entity, campaign_id, adgroup_id, keyword_text, pat, match_type, bid, path):
	#open excel if exist
	operation='Create'
	#append keyword
	try:
		df=pd.read_excel(path + "/output/management.xlsx",index_col=None)
	except:	
		columns={'Product','Entity', 'Operation','Campaign Id','Ad Group Id',
		'State', 'Bid','Keyword Text','Product Targeting Expression', 'Match Type'}
		df=pd.DataFrame(columns=columns)

	df_to_add=pd.DataFrame({'Product': [product],'Entity': [entity], 'Operation': [operation],'Campaign Id': [campaign_id],'Ad Group Id': [adgroup_id],
		 'State': ['enabled'], 'Bid': [bid],'Keyword Text':[keyword_text],'Product Targeting Expression':[pat], 'Match Type':[match_type]
		})

	df['Campaign Id']=df['Campaign Id'].replace('\'', '', regex=True)
	df['Ad Group Id']=df['Ad Group Id'].replace('\'', '', regex=True)
	df=pd.concat([df, df_to_add], ignore_index=True).drop_duplicates()

	
	df.to_excel(path + "/output/management.xlsx", index=False, columns=['Product','Entity', 'Operation','Campaign Id','Ad Group Id',
		'State', 'Bid', 'Keyword Text', 'Product Targeting Expression', 'Match Type'])


def delete_keyword_creation(path):
	os.remove(path + "/output/management.xlsx")

def get_type(df):
	if 'Product Targeting' in df['Entity'].values:#.str.contains('Product Targeting').any():
		object_type=['PAT']
	if 'Audience Targeting' in df['Entity'].values:#.str.contains('Audience Targeting').any():
		object_type=['AT']
	if 'Keyword'in df['Entity'].values:
		match_types=dataframe.select_row_by_val(df,'State','enabled')['Match Type'].unique()
		mask=np.isin(match_types, ['broad', 'phrase','exact'])
		if len(match_types[mask])>1:
			object_type=match_types[mask]#'mix'
		else:
			object_type=match_types[mask]#[0]
	print(f"Adgroup type: {object_type}")
	return object_type


def create_adgroup_link(campaign_src_name, adgroup_src_name, campaign_src_id, adgroup_src_id, campaign_src_type,
	campaign_dst_name, adgroup_dst_name, campaign_dst_id, adgroup_dst_id, campaign_dst_type, path, name):
	try:
		df=pd.read_excel(path + "/raw/link"+ name +".xlsx",index_col=None).drop_duplicates()
	except:	
		columns={'Campaign source Name', 'Ad Group source name', 'Campaign source Id', 'Ad Group source Id', 'Campaign source type', 
		'Campaign destination Name', 'Ad Group destination name','Campaign destination Id',	'Ad Group destination Id', 'Campaign destination type'}
		df=pd.DataFrame(columns=columns)

	df_to_add=pd.DataFrame({'Campaign source Name':[campaign_src_name], 'Ad Group source name':[adgroup_src_name], 
		'Campaign source Id':[campaign_src_id], 'Ad Group source Id': [adgroup_src_id],'Campaign source type':[campaign_src_type],
		 'Campaign destination Name':[campaign_dst_name], 'Ad Group destination name': [adgroup_dst_name],
		 'Campaign destination Id':[campaign_dst_id], 'Ad Group destination Id':[adgroup_dst_id], 'Campaign destination type':[campaign_dst_type]
		})

	df=pd.concat([df, df_to_add], ignore_index=True)
	df.drop_duplicates(inplace=True)
	print(df)
	df.to_excel(path + "/raw/link"+ name +".xlsx", index=False, columns=['Campaign source Name', 'Ad Group source name', 'Campaign source Id', 'Ad Group source Id','Campaign source type', 
		'Campaign destination Name', 'Ad Group destination name','Campaign destination Id',	'Ad Group destination Id', 'Campaign destination type'])

def add_missing_keyword(df_source, df_destination, adgroup_source_id, adgroup_destination_id, path):
	df_src=df_source.copy()
	df_dest=df_destination.copy()
	df_missing_kw=pd.DataFrame()
	print(df_dest)	
	if df_src['Entity'].str.contains('Keyword').any():
		df_src.dropna(subset=['Keyword Text'], inplace=True)
		df_src=df_src[df_src['Entity']=='Keyword'] #bc of negative kw

	if df_src['Entity'].str.contains('Product Targeting').any():
			df_src.dropna(subset=['Product Targeting Expression'], inplace=True)
			df_src=df_src[df_src['Entity']=='Product Targeting'] #bc of negative kw
			df_src['Product Targeting Expression']=df_src['Product Targeting Expression'].str.lower()
	if df_dest['Entity'].str.contains('Keyword').any():
		df_dest.dropna(subset=['Keyword Text'], inplace=True)
		df_dest=df_dest[df_dest['Entity']=='Keyword']
		df_merge=pd.merge(df_src['Keyword Text'], df_dest['Keyword Text'], how='left', left_on=['Keyword Text'], right_on=['Keyword Text'], suffixes=("_src", "_dst"), indicator=True)
		missing_kw=df_merge[df_merge['_merge'] == 'left_only']['Keyword Text']
		df_missing_kw=pd.merge(missing_kw, df_src, how='left',left_on=['Keyword Text'], right_on=['Keyword Text'], suffixes=("_kw", "_src"))
	if df_dest['Entity'].str.contains('Product Targeting').any():
			df_dest.dropna(subset=['Product Targeting Expression'], inplace=True)
			df_dest=df_dest[df_dest['Entity']=='Product Targeting']
			df_dest['Product Targeting Expression']=df_dest['Product Targeting Expression'].str.lower()
			df_merge=pd.merge(df_src['Product Targeting Expression'], df_dest['Product Targeting Expression'], how='left',
	left_on=['Product Targeting Expression'], right_on=['Product Targeting Expression'], suffixes=("_src", "_dst"), indicator=True)
			missing_kw=df_merge[df_merge['_merge'] == 'left_only']['Product Targeting Expression']
			df_missing_kw=pd.merge(missing_kw, df_src, how='left',left_on=['Product Targeting Expression'], right_on=['Product Targeting Expression'], suffixes=("_kw", "_src"))
	if df_missing_kw.empty:
		print('Nothing to add, probably audience campaign')
	else:
		print(df_missing_kw)
		print('loop')
		for i, row in df_missing_kw.iterrows():
			product=df_dest['Product'].iloc[0]
			entity=row['Entity']
			campaign_id=df_dest['Campaign Id'].iloc[0]
			adgroup_id=df_dest['Ad Group Id'].iloc[0]
			match_type=row['Match Type']
			bid=row['Bid']
			keyword_text=row['Keyword Text']
			pat=row['Product Targeting Expression']
			create_keyword_pat(product, entity, campaign_id, adgroup_id, keyword_text, pat, match_type, bid, path)