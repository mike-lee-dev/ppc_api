from utils import dataframe, input_output, campaign_management
import pandas as pd
import numpy as np
import constant
from itertools import combinations
from rapidfuzz.distance import Levenshtein

def jaccard_similiarity(list1, list2):
	#for advertised ASINs
	#we could add in the furtur targeted keywords
	#https://www.statology.org/jaccard-similarity-python/
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	if union==0:
		return 0
	else:
		return float(intersection) / union

def Levenshtein(stra, strb):
	#for campaign and agroup names
	return Levenshtein.distance(stra, strb)

def euclidean_distance(a,b): #a and b should be np arrays
	dist = np.linalg.norm(a-b)
	return dist


#step 1 get variation in campaigns/adgroups vs keyword + ASIN
#step 2 measure distance between campaigns&adgroup

def get_product_by_adgroup(df_bid, adgroup_id):
	productIDs=dataframe.select_row_by_val(df_bid, 'Ad Group Id', adgroup_id, 'Entity', 'Product Ad', 'State', 'enabled')['SKU'].to_numpy()
	return productIDs


def campaign_link_manual(path_bid):
	df_bid_SP=input_output.read_bid('SP',path_bid)
	df_bid_manual_list=dataframe.select_row_by_val(df_bid_SP,'Targeting Type', 'MANUAL')	
	df_bid_manual=df_bid_SP[df_bid_SP['Campaign Id'].isin(df_bid_manual_list['Campaign Id'].unique())]
	df_bid_SB=input_output.read_bid('SB',path_bid)
	df_bid_SD=input_output.read_bid('SD',path_bid)
	for df_bid in list(combinations([df_bid_manual,df_bid_SB, df_bid_SD],2)): #[('SP', 'SB'), ('SP', 'SD'), ('SB', 'SD')]
		if df_bid[0]['Product'].iloc[0]=='Sponsored Brands':
			adgroup_src_ids=dataframe.select_row_by_val(df_bid[0], 'Campaign State (Informational only)','enabled')['Ad Group Id'].unique()
		else:
			adgroup_src_ids=dataframe.select_row_by_val(df_bid[0], 'Ad Group State (Informational only)','enabled')['Ad Group Id'].unique()
		for adgroup_src_id in adgroup_src_ids:
			new_score=0.
			best_score=0.
			best_campaign_dst_name=''
			best_adgroup_dst_name=''
			best_campaign_dst_id=''
			best_adgroup_dst_id=''
			df_adgroup_src=dataframe.select_row_by_val(df_bid[0], 'Ad Group Id', adgroup_src_id)

			campaign_src_name=df_adgroup_src['Campaign Name (Informational only)'].iloc[0]
			campaign_src_id=df_adgroup_src['Campaign Id'].iloc[0]
			campaign_src_type=df_adgroup_src['Product'].iloc[0]	
			if campaign_src_type=='Sponsored Brands':
				adgroup_src_name='Not available'
			else:
				adgroup_src_name=df_adgroup_src['Ad Group Name'].iloc[0]	
			adgroup_src_type=campaign_management.get_type(df_adgroup_src)
			for to_skip in campaign_adgroup_to_skip:	
				if to_skip in campaign_dst_name.lower():
					continue
				if to_skip in adgroup_dst_name.lower():
					continue
			if df_bid[1]['Product'].iloc[0]=='Sponsored Brands':
				adgroup_dst_ids=dataframe.select_row_by_val(df_bid[1], 'Campaign State (Informational only)','enabled')['Ad Group Id'].unique()	

			else:
				print(df_bid[1])
				adgroup_dst_ids=dataframe.select_row_by_val(df_bid[1], 'Ad Group State (Informational only)','enabled')['Ad Group Id'].unique()	
			for adgroup_dst_id in adgroup_dst_ids:
				campaign_dst_name=dataframe.select_row_by_val(df_bid[1],'Ad Group Id', adgroup_dst_id)['Campaign Name (Informational only)'].iloc[0]
				campaign_dst_id = dataframe.select_row_by_val(df_bid[1],'Ad Group Id', adgroup_dst_id)['Campaign Id'].iloc[0]
				campaign_dst_type=dataframe.select_row_by_val(df_bid[1],'Ad Group Id', adgroup_dst_id)['Product'].iloc[0]
				if df_bid[1]['Product'].iloc[0]=='Sponsored Brands':
					adgroup_dst_name='Not available'
				else:	
					adgroup_dst_name=dataframe.select_row_by_val(df_bid[1],'Ad Group Id', adgroup_dst_id)['Ad Group Name (Informational only)'].iloc[0]
				df_adgroup_dst=dataframe.select_row_by_val(df_bid[1], 'Ad Group Id', adgroup_dst_id)
				adgroup_dst_type=campaign_management.get_type(df_adgroup_dst)			

	
				if np.array_equal(adgroup_src_type,adgroup_dst_type):
					print(campaign_src_name)
					#print(adgroup_src_name)
					print(adgroup_src_type)
					print(campaign_dst_name)
					print(adgroup_dst_name)
					print(adgroup_dst_type)
					if campaign_src_type=='Sponsored Brands' or campaign_dst_type=='Sponsored Brands':
						new_score = jaccard_similiarity(list(campaign_src_name), list(campaign_dst_name))
						adgroup_dst_name='Not available'
					else:
						new_score = (
							jaccard_similiarity(list(campaign_src_name.lower()), list(campaign_dst_name.lower())) *
							jaccard_similiarity(list(adgroup_src_name.lower()), list(adgroup_dst_name.lower()))
								)
				for to_skip in campaign_adgroup_to_skip:	
					if to_skip in campaign_dst_name.lower():
						continue
					if to_skip in adgroup_dst_name.lower():
						continue
				if new_score>best_score:
					best_score=new_score
					best_campaign_dst_name=campaign_dst_name
					best_adgroup_dst_name=adgroup_dst_name
					best_campaign_dst_id=campaign_dst_id
					best_adgroup_dst_id=str(adgroup_dst_id)
					best_campaign_src_type=campaign_dst_type

			if best_score!=0.:
				print(f"Best Score: {best_score}")
				print(f"Campaign source: {campaign_src_name}")
				#print(f"Ad Group source: {adgroup_src_name}")
				print(f"Campaign destination: {best_campaign_dst_name}")
				print(f"Ad Group destination: {best_adgroup_dst_name}")
				campaign_management.create_adgroup_link(campaign_src_name, adgroup_src_name, campaign_src_id, adgroup_src_id, campaign_src_type,
					best_campaign_dst_name, best_adgroup_dst_name, best_campaign_dst_id, best_adgroup_dst_id, best_campaign_src_type, path_bid, '-manual-manual'
					)

def campaign_link_auto(path):
	df_bid_SP=input_output.read_bid('SP',path)
	df_bid_auto_list=dataframe.select_row_by_val(df_bid_SP,'Targeting Type', 'AUTO', 'State', 'enabled')
	df_bid_manual_list=dataframe.select_row_by_val(df_bid_SP,'Targeting Type', 'MANUAL', 'State', 'enabled')	
	df_bid_auto=df_bid_SP[df_bid_SP['Campaign Id'].isin(df_bid_auto_list['Campaign Id'].unique())]
	df_bid_manual=df_bid_SP[df_bid_SP['Campaign Id'].isin(df_bid_manual_list['Campaign Id'].unique())]
	print(df_bid_auto)
	print(df_bid_manual)

	adgroup_src_ids=dataframe.select_row_by_val(df_bid_auto, 'Ad Group State (Informational only)','enabled')['Ad Group Id'].unique()

	for adgroup_src_id in adgroup_src_ids:
		new_score=0.
		best_score=0.
		best_campaign_dst_name=''
		best_adgroup_dst_name=''
		best_campaign_dst_id=''
		best_adgroup_dst_id=''
		campaign_src_name=dataframe.select_row_by_val(df_bid_auto,'Ad Group Id', adgroup_src_id)['Campaign Name (Informational only)'].iloc[0]
		campaign_src_id=dataframe.select_row_by_val(df_bid_auto,'Ad Group Id', adgroup_src_id)['Campaign Id'].iloc[0]
		campaign_src_type=dataframe.select_row_by_val(df_bid_auto,'Ad Group Id', adgroup_src_id)['Product'].iloc[0]
		adgroup_src_name=dataframe.select_row_by_val(df_bid_auto,'Ad Group Id', adgroup_src_id)['Ad Group Name (Informational only)'].iloc[0]
		print(f"Ad Group source ID: {adgroup_src_id}")
		df_adgroup_src=dataframe.select_row_by_val(df_bid_auto, 'Ad Group Id', adgroup_src_id)
		adgroup_src_type=campaign_management.get_type(df_adgroup_src)
		asins_src=get_product_by_adgroup(df_bid_auto, adgroup_src_id)	
		adgroup_dst_ids=dataframe.select_row_by_val(df_bid_manual, 'Ad Group State (Informational only)','enabled')['Ad Group Id'].unique()
		for adgroup_dst_id in adgroup_dst_ids:
			campaign_dst_name=dataframe.select_row_by_val(df_bid_manual,'Ad Group Id', adgroup_dst_id)['Campaign Name (Informational only)'].iloc[0]
			if 'defen' in campaign_dst_name.lower():
				continue
			if 'catch' in campaign_dst_name.lower():
				continue
			if 'competi' in campaign_dst_name.lower():
				continue
			if 'cat' in campaign_dst_name.lower():
				continue
			if 'brand' in campaign_dst_name.lower():
				continue					
			campaign_dst_id = dataframe.select_row_by_val(df_bid_manual,'Ad Group Id', adgroup_dst_id)['Campaign Id'].iloc[0]
			campaign_dst_type=dataframe.select_row_by_val(df_bid_manual,'Ad Group Id', adgroup_dst_id)['Product'].iloc[0]
			adgroup_dst_name=dataframe.select_row_by_val(df_bid_manual,'Ad Group Id', adgroup_dst_id)['Ad Group Name (Informational only)'].iloc[0]
			if 'defen' in adgroup_dst_name.lower():
				continue
			if 'catch' in adgroup_dst_name.lower():
				continue
			if 'competi' in adgroup_dst_name.lower():
				continue
			if 'cat' in adgroup_dst_name.lower():
				continue
			if 'brand' in adgroup_dst_name.lower():
				continue			
			df_adgroup_dst=dataframe.select_row_by_val(df_bid_manual, 'Ad Group Id', adgroup_dst_id)
			adgroup_dst_type=campaign_management.get_type(df_adgroup_dst)			
			asins_dst=get_product_by_adgroup(df_bid_manual, adgroup_dst_id)				

			new_score = jaccard_similiarity(asins_src, asins_dst)
			if new_score>0.8:
				best_score=new_score
				best_campaign_dst_name=campaign_dst_name
				best_adgroup_dst_name=adgroup_dst_name
				best_campaign_dst_id=campaign_dst_id
				best_adgroup_dst_id=str(adgroup_dst_id)
				campaign_management.create_adgroup_link(campaign_src_name, adgroup_src_name, campaign_src_id, adgroup_src_id, campaign_src_type,
					campaign_dst_name, adgroup_dst_name, campaign_dst_id, adgroup_dst_id, campaign_dst_type, path, '-auto-manual'
					)


def campaign_link_cat: