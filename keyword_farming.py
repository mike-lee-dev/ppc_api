import pandas as pd
import nltk
import re
from utils import input_output, dataframe, amazon_campaign_link, campaign_management
#https://towardsdatascience.com/fuzzywuzzy-find-similar-strings-within-one-column-in-a-pandas-data-frame-99f6c2a0c212
#   KW: 	Auto -> 	Manual KW SP (broad/exact)	
#	PAT: 	Auto -> 	Manual PAT SP (PAT)
#			CAT  ->

path='./data Fititude US/'

def main():
	#link campaigns
	try:
		campaign_management.delete_keyword_creation(path)
	except:
		pass
	#for auto campaigns get SQR if conversion move kw and PAT to SP linked campaign
	df_bid_SP=input_output.read_bid('SP',path)
	try:
		linkdf=pd.read_excel(io=path + 'raw/link-auto-manual.xlsx',header=0, engine="openpyxl")	
	except:
		amazon_campaign_link.campaign_link_auto(path)
	print(linkdf)
	df_sqr=input_output.read_sqr(path)
	df_sqr_auto=df_sqr.loc[df_sqr['Campaign Name'].isin(linkdf['Campaign source Name'])]
	df_search_term_filtered=df_sqr_auto[df_sqr_auto['7 Day Total Sales ']>0]
	print(df_sqr_auto)
	for i, row in df_search_term_filtered.iterrows():
		auto_campaign_name=row['Campaign Name']
		auto_adgroup_name=row['Ad Group Name']
		adgroup_dst_ids=dataframe.select_row_by_val(linkdf,'Campaign source Name', auto_campaign_name,'Ad Group source name', auto_adgroup_name)['Ad Group destination Id']
		auto_campaign_id=dataframe.select_row_by_val(linkdf,'Campaign source Name', auto_campaign_name,'Ad Group source name', auto_adgroup_name)['Campaign source Id'].iloc[0]
		auto_adgroup_id=dataframe.select_row_by_val(linkdf,'Campaign source Name', auto_campaign_name,'Ad Group source name', auto_adgroup_name)['Ad Group source Id'].iloc[0]
		for adgroup_dst_id in adgroup_dst_ids:
			campaign_dst_id=dataframe.select_row_by_val(linkdf,'Ad Group destination Id', adgroup_dst_id)['Campaign destination Id'].iloc[0]
			df_adgroup_dst=dataframe.select_row_by_val(df_bid_SP, 'Campaign Id', campaign_dst_id, 'Ad Group Id', adgroup_dst_id)
			#if df_search_term_filtered in sqr[sqr['7 Day Total Sales ']>0]:
			auto_targeting=row['Targeting']
			match_type_dst=campaign_management.get_type(df_adgroup_dst)[0]
			bid=df_adgroup_dst['Bid'].mean()
			if re.match('^b0',row['Customer Search Term']) and match_type_dst=='PAT':
				match_type_dst='-'
				pat='asin=\"'+row['Customer Search Term'].upper()+'\"'
				entity='Product Targeting'
				campaign_management.create_keyword_pat('Sponsored Products', entity, campaign_dst_id, adgroup_dst_id, '', pat, match_type_dst, bid, path)
				entity='Negative Product Targeting'
				campaign_management.create_keyword_pat('Sponsored Products', entity, auto_campaign_id, auto_adgroup_id, '', pat, '-', '', path)

			if not re.match('^b0',row['Customer Search Term']) and match_type_dst=='mix':
				match_type_dst='broad'
				targeting=row['Customer Search Term']
				entity='Keyword'
				campaign_management.create_keyword_pat('Sponsored Products', entity, campaign_dst_id, adgroup_dst_id, targeting,'', match_type_dst, bid, path)
				entity='Negative Keyword'
				campaign_management.create_keyword_pat('Sponsored Products', entity, auto_campaign_id, auto_adgroup_id, targeting,'', 'negativePhrase', '', path)
			if not re.match('^b0',row['Customer Search Term']) and match_type_dst in ['broad', 'phrase', 'exact']:
				targeting=row['Customer Search Term']
				entity='Keyword'
				campaign_management.create_keyword_pat('Sponsored Products', entity, campaign_dst_id, adgroup_dst_id, targeting,'', match_type_dst, bid, path)
				entity='Negative Keyword'
				campaign_management.create_keyword_pat('Sponsored Products', entity, auto_campaign_id, auto_adgroup_id, targeting,'', 'negativePhrase', '', path)
		

if __name__ == "__main__":
	main()
