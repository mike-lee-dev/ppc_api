#from models.conversion_rate_estimation import clustering
import pathlib
import pandas as pd
from utils import input_output, dataframe
from helpers import date_helpers
import global_var
from models import cluster, conversion_rate_estimation, object_structure
from datetime import datetime, date
import math
import glob
import os


def main():
	paths = glob.glob("./data*")
	print(paths)
	for global_var.path  in paths:
		print(global_var.path)

		filelist = [ f for f in os.listdir(global_var.path+"/prediction/")]
		for f in filelist:
			os.remove(os.path.join(global_var.path+"/prediction/", f))
		campaign_types=[]
		
		#SP
		try:
			df_bid_SP, df_history_SP=read_history('SP', global_var.path)
			df_history_SP=merge_no_date(df_bid_SP, df_history_SP, 'SP', global_var.path)
			campaign_types.append('SP')
		except FileNotFoundError:
			print('no SP')
		except ValueError:
			print('no SP')

		if 'SP' in campaign_types:
			df_history=df_history_SP[['Date','Campaign Name','Ad Group Name','Targeting','Match Type','Clicks','Conversions','Sales']]
		
		#SB & SBV
		try:
			df_bid_SB, df_history_SB=read_history('SB', global_var.path)
			df_history_SB=merge_no_date(df_bid_SB, df_history_SB, 'SB', global_var.path)
			campaign_types.append('SB')
		except FileNotFoundError:
			print('no SB')
		except ValueError:
			print('no SB')
		try:
			df_history_SBV=input_output.read_report('SBV', global_var.path)
			campaign_types.append('SBV')
		except FileNotFoundError:
			print('no SBV')
		except ValueError:
			print('no SBV')


		if 'SB' in campaign_types:
			df_history=pd.concat([df_history,df_history_SB[['Date','Campaign Name','Ad Group Name','Targeting','Match Type','Clicks','Conversions','Sales']]])
		if 'SBV' in campaign_types:
			df_history=pd.concat([df_history,df_history_SBV[['Date','Campaign Name','Ad Group Name','Targeting','Match Type','Clicks','Conversions','Sales']]])	

		if 'SB' in campaign_types and 'SBV' in campaign_types:
			df_history_SB=pd.concat([df_history_SB,df_history_SBV[['Date','Campaign Name','Ad Group Name','Targeting','Match Type','Clicks','Conversions','Sales']]])
		
		try:
			df_history_SB=merge_no_date(df_bid_SB, df_history_SB, 'SB', global_var.path) #overwrite df_history without dupplicated label
		except FileNotFoundError:
			print('no SB')
		except ValueError:
			print('no SB')
		#SD
		try:
			df_bid_SD, df_history_SD=read_history('SD', global_var.path)
			df_history_SD=merge_no_date(df_bid_SD, df_history_SD, 'SD', global_var.path)
			campaign_types.append('SD')
		except FileNotFoundError:
			print('no SD')
		except ValueError:
			print('no SD')
		input_output.append_bid_history_with_date(global_var.path, campaign_types)
	
		if 'SD' in campaign_types:
			df_history=pd.concat([df_history,df_history_SD[['Date','Campaign Name','Ad Group Name','Targeting','Match Type','Clicks','Conversions','Sales']]])



		df_history['Ad Group Name']=df_history['Ad Group Name'].astype(str)
		df_history['Ad Group Name']=df_history['Ad Group Name'].replace('0', 'Not available')
		df_history['Match Type'].replace(0, '-', inplace=True)
		df_history['Match Type'].fillna('-', inplace=True)
		df_history.to_csv(global_var.path + "/prediction/df_history.csv")
			
		df_clustered, RF_decoding=initiate_clustering(df_history, global_var.path)


		df_forecast=conversion_rate(df_clustered, RF_decoding, global_var.path)
		input_output.backup_bid_xls(global_var.path)
		for campaign_type in campaign_types:
			if campaign_type=='SP':
				df_bid_SP=merge_forecast_bid(df_bid_SP, df_forecast)
				df_bid_history_SP=dataframe.to_datetime64(input_output.read_bid_history('SP', global_var.path))
				df_bid_SP=get_slope_conv_value(df_history_SP, df_bid_SP, df_bid_history_SP, global_var.path)
				df_bid_SP.to_csv(global_var.path + "/prediction/newbids_SP.csv")
				update_bid_excel(df_bid_SP, 'SP', global_var.path)

			if campaign_type=='SB':			
				df_bid_SB=merge_forecast_bid(df_bid_SB, df_forecast)
				df_bid_history_SB=dataframe.to_datetime64(input_output.read_bid_history('SB', global_var.path))
				df_bid_SB=get_slope_conv_value(df_history_SB, df_bid_SB, df_bid_history_SB, global_var.path)
				df_bid_SB.to_csv(global_var.path + "/prediction/newbids_SB.csv")
				update_bid_excel(df_bid_SB, 'SB', global_var.path)

			if campaign_type=='SD':
				df_bid_SD=merge_forecast_bid(df_bid_SD, df_forecast)
				df_bid_history_SD=dataframe.to_datetime64(input_output.read_bid_history('SD', global_var.path))	
				df_bid_SD=get_slope_conv_value(df_history_SD, df_bid_SD, df_bid_history_SD, global_var.path)
				df_bid_SD.to_csv(global_var.path + "/prediction/newbids_SD.csv")
				update_bid_excel(df_bid_SD, 'SD', global_var.path)


def read_history(campaign_type, path):
	path_bid=global_var.path
	print(path_bid)
	df_bid=input_output.read_bid(campaign_type,path_bid)
	df_history=input_output.read_report(campaign_type, global_var.path)
	return df_bid, df_history




def merge_no_date(df_bid, df_history, campaign_type, path):
	#add target without data
	#Get only the relevant columns from df_bid and drop duplicate rows
	if campaign_type=='SP':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['Campaign Name', 'Ad Group Name', 'Targeting', 'Match Type']]#.drop_duplicates()
		#df_bid_part['Portfolio name']='Not grouped'
	if campaign_type=='SB':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['Campaign Name', 'Targeting', 'Match Type']]
		#df_bid_part['Portfolio name']='Not grouped'
	if campaign_type=='SD':
		df_bid_part = dataframe.get_biddable_object(df_bid).loc[:, ['Campaign Name', 'Ad Group Name', 'Targeting']]#.drop_duplicates()
		#df_bid_part['Portfolio name']='-'
	df_bid_part['Currency']='USD'
	# append df_bid_no_click to a row of df_history to copy the structure of df_history, then get rid of this row by indexing
	df_bid_no_click = pd.concat([df_history.iloc[0:1,:],df_bid_part]).iloc[1:]
	max_Date = df_history['Date'].max()

	#Create a dictionary that fills na values for each column. NaN dates are replaced with the most recent date, and all
	#numerical columns are replaced with 0.
	dict_fillna = {'Date': max_Date}
	for i in df_bid_no_click.columns[7:]:
	    dict_fillna[i] = 0

	#Perform the replacement
	df_bid_final = df_bid_no_click.fillna(dict_fillna)

	#Append the final version of df_bid to df_history
	df_history = pd.concat([df_history, df_bid_final],ignore_index=True)#.reset_index(drop = True)
	df_history.Date = df_history.Date.dt.date
	df_history=date_helpers.format_date(df_history)
	df_history.to_csv(path+"/prediction/full_history.csv")
	return df_history
	
def initiate_clustering(df_history, path):
	#try:
	#	df_clustered, RF_decoding=input_output.read_clustering() We need to update the history, this can't work. We should store the rules if we don't want to cluster each time
	#except:
	df_clustered, RF_decoding=cluster.clustering(df_history, path)
	input_output.write_clustering(df_clustered, RF_decoding, path)
	#df_clustered=cluster.aggregate_by_node(RF_decoding, df_history)	
	return df_clustered, RF_decoding

def conversion_rate(df_clustered, RF_decoding, path):
	kf, X_t=conversion_rate_estimation.initiate(df_clustered, path)

	df_forecast=X_t.join(RF_decoding.set_index('Leave'), how='outer').drop_duplicates()
	df_forecast.to_csv(path+ "/prediction/df_forecast.csv")
	return df_forecast

def merge_forecast_bid(df_bid, df_forecast):
	df_bid=pd.merge(df_bid, df_forecast, how='left', left_on=['Campaign Name', 'Ad Group Name','Targeting', 'Match Type'], right_on=['Campaign_Name', 'Ad_Group_Name','Targeting', 'Match_Type'])
	return df_bid

def get_slope_conv_value(df_history, df_bid, df_bid_history, path):
	#for every campaign, every adgroup, every target in df_bid get a + b
	try: 
		default_conv_val=df_history['Sales'].sum()/df_history['Conversions'].sum()
	except ZeroDivisionError:
		default_conv_val=20.
	dfcam=dataframe.select_row_by_val(df_bid,'Entity',"Campaign", 'State', 'enabled')

	#loop over campaign, then adgroup, then target and add output (CV and slope) to the prediction file
	for i, cam in dfcam.iterrows(): # we could use itertuples to speed up the performance but we would need to have the same format for all campaigns 
		if cam['Product']=='Sponsored Display':
			if cam['Cost Type']=='vcpm':
				continue
		if not cam['Campaign Id'] in input_output.read_target(path)['Campaign Id'].values:
			input_output.add_campaign_to_target(path,cam['Campaign Id'],cam['Campaign Name'])
		df_campaign_history=dataframe.select_row_by_val(df_history,'Campaign Name',cam['Campaign Name'])
		df_campaign_bid_history=dataframe.select_row_by_val(df_bid_history,'Campaign Id', str(cam['Campaign Id']))
		c=object_structure.Campaign(cam['Campaign Id'],cam['Campaign Name'],cam['State'], df_campaign_history, df_campaign_bid_history,1.,default_conv_val)
		dataframe.change_val_if_col_contains(df_bid,'a', c.a, 'Campaign Id', str(c.campaign_id))
		dataframe.change_val_if_col_contains(df_bid,'conv_value', c.conv_value, 'Campaign Id', str(c.campaign_id))
		dfadgr=dataframe.select_row_by_val(df_bid,'Entity',"Ad Group",'Campaign Id', str(c.campaign_id))

		if dfcam['Product'].iloc[0]=='Sponsored Brands':
			try:
				adgroupID=dataframe.select_row_by_val(df_bid,'Entity',"Keyword",'Campaign Id', str(c.campaign_id))['Ad Group Id (Read only)'].iloc[0]
			except:
				adgroupID=dataframe.select_row_by_val(df_bid,'Entity',"Product Targeting",'Campaign Id', str(c.campaign_id))['Ad Group Id (Read only)'].iloc[0]
			dfadgr=pd.DataFrame({'Ad Group Id':adgroupID,'Ad Group Name':['Not available'],'State':['enable'], 'Ad Group Default Bid':[0.7]})

		for j, adgr in dfadgr.iterrows():
			df_adgroup_history=dataframe.select_row_by_val(df_history,'Campaign Name', c.campaign_name, 'Ad Group Name',adgr['Ad Group Name'])
			df_adgroup_bid_history=dataframe.select_row_by_val(df_bid_history,'Campaign Id', str(c.campaign_id), 'Ad Group Id',str(adgr['Ad Group Id']))
			a=object_structure.Adgroup(
				adgroup_id=adgr['Ad Group Id'],
				adgroup_name=adgr['Ad Group Name'], 
				adgroup_status=adgr['State'], 
				adgroup_bid=adgr['Ad Group Default Bid'], 
				df_adgroup_history=df_adgroup_history,
				df_adgroup_bid_history=df_adgroup_bid_history,
				a=c.a,
				conv_value=c.conv_value
				)

			dataframe.change_val_if_col_contains(df_bid,'a', a.a, 'Ad Group Id',str(a.adgroup_id))
			dataframe.change_val_if_col_contains(df_bid,'conv_value', a.conv_value, 'Ad Group Id', str(a.adgroup_id))
			dfkw=dataframe.select_row_by_val(df_bid,'Entity',"Keyword")
			dfpat=dataframe.select_row_by_val(df_bid,'Entity',"Product Targeting")
			dfct=dataframe.select_row_by_val(df_bid,'Entity',"Contextual Targeting")
			dfaud=dataframe.select_row_by_val(df_bid,'Entity',"Audience Targeting")
			dftarget=pd.concat([dfkw,dfpat,dfct,dfaud])
			#dftarget=dftarget.concat()
			dftarget=dataframe.select_row_by_val(dftarget, 'Campaign Id', str(c.campaign_id), 'Ad Group Id', str(a.adgroup_id))

			for k, target in dftarget.iterrows():
				target_name=target['Targeting']
				df_target_history=dataframe.select_row_by_val(
				df_history,
				'Campaign Name', c.campaign_name,
				'Ad Group Name', a.adgroup_name,
				'Targeting',target_name,
				'Match Type', target['Match Type']
				)
				df_target_bid_history=dataframe.select_row_by_val(
					df_bid_history,
					'Target Id', str(target['Target Id']),
					)				
				if target['Entity'] in ['Keyword','Product Targeting','Contextual Targeting','Audience Targeting']:
			
					t=object_structure.Target(
					match_type=target['Match Type'],
					target_bid=target['Bid'],
					target_name=target_name,
					target_id=target['Target Id'],
					target_status=target['State'],
					df_target_history=df_target_history,
					df_target_bid_history=df_target_bid_history,
					a=a.a,
					conv_value=a.conv_value
					)

					dataframe.change_val_if_col_contains(df_bid,'a', t.a, 'Target Id', t.target_id)
					dataframe.change_val_if_col_contains(df_bid,'conv_value', t.conv_value, 'Target Id', t.target_id)

									
	dftarget=input_output.read_target(path)
	df_bid=pd.merge(df_bid,dftarget,how='left',left_on='Campaign Id',right_on='Campaign Id').drop_duplicates(ignore_index=True)
	print(global_var.path)
	df_bid['new_bid']=df_bid['target_acos']*df_bid['CR']*df_bid['conv_value']/df_bid['a']
	df_bid['new_bid']=df_bid.apply(lambda x: limit_bid_change(x['Bid'],x['new_bid'], 0.25), axis=1)
	df_bid['new_bid']=df_bid.apply(lambda x: valid_bid(x['new_bid']), axis=1)
	return df_bid

def limit_bid_change(old_bid, new_bid, max_change):
	upper_limit=old_bid*(1+max_change)
	lower_limit=old_bid/(1+max_change)
	if new_bid>upper_limit:
		new_bid=upper_limit
	if new_bid<lower_limit:
		new_bid=lower_limit
	print(f"new bid after limit{new_bid}")
	return new_bid

def valid_bid(new_bid, campaign_type='SP'):
	if campaign_type=='SP':
		limit=0.02
	if campaign_type=='SB':
		limit=0.1
	if new_bid<limit:
		new_bid=limit
	if new_bid=='':
		new_bid=limit
	return new_bid

def update_bid_excel(df_bid, campaign_type, path):
	#loop over excel file and change bids

	wb=input_output.open_wb(path)
	ws=input_output.open_ws(wb,campaign_type)
	for row in ws.iter_rows(min_row=2):
		entity=row[1].value
		campaignID='\'' +str(row[3].value)
		oldbid=""
		if campaign_type=='SP' and entity in ['Keyword', 'Product Targeting']:
			adgroupID='\'' +str(row[4].value)
			if entity=='Keyword':
				targetID='\'' +str(row[7].value)
			elif entity=='Product Targeting':
				targetID='\'' +str(row[8].value)

			campaign=row[11].value
			adgroup=row[12].value
			#oldbid=row[19].value
			oldbid=row[27].value

			#target=row[20].value
			target=row[28].value
			#matchtype=row[21].value
			matchtype=row[29].value
			print(f"campaign_type: {campaign_type}")
			print(f"campaign: {campaign}")
			print(f"adgroup: {adgroup}")
			print(f"oldbid: {oldbid}")
			print(f"target: {target}")
			print(f"matchtype: {matchtype}")

		if campaign_type=='SB' and entity in ['Keyword', 'Product Targeting']:
			adgroupID=str(row[6].value)
			if entity=='Keyword':
				targetID='\'' +str(row[7].value)
			elif entity=='Product Targeting':
				targetID='\'' +str(row[8].value)
			campaign=row[10].value
			adgroup='Not available'
			#oldbid=row[18].value
			oldbid=row[21].value
			#target=row[19].value
			target=row[22].value
			#matchtype=row[20].value
			matchtype=row[21].value
			print(f"campaign_type: {campaign_type}")
			print(f"campaign: {campaign}")
			print(f"adgroup: {adgroup}")
			print(f"oldbid: {oldbid}")
			print(f"target: {target}")
			print(f"matchtype: {matchtype}")
		if campaign_type=='SD' and entity in ['Product Targeting','Contextual Targeting','Audience Targeting']:
			adgroupID='\'' +str(row[5].value)
			targetID='\'' +str(row[7].value)
			campaign=row[10].value
			adgroup=row[11].value
			#oldbid=row[19].value
			oldbid=row[25].value
			#target=row[22].value
			target=row[29].value

			matchtype="-"
			print(f"campaign_type: {campaign_type}")
			print(f"campaign: {campaign}")
			print(f"adgroup: {adgroup}")
			print(f"oldbid: {oldbid}")
			print(f"target: {target}")
			print(f"matchtype: {matchtype}")		
		
		if entity in ['Keyword', 'Product Targeting','Contextual Targeting', 'Audience Targeting']:
			print(f"old bid: {oldbid}")
			print(f"targetID: {targetID}")
			print(f"New bid: {dataframe.select_row_by_val(df_bid, 'Target Id', targetID)['new_bid']}")
			try: 
				optimized=dataframe.select_row_by_val(df_bid, 'Target Id',targetID)['optimized'].values[0]
			except:
				print("can't find this keyword!!!")
				print(f"Bid for Campaign: {campaignID}, adgroup: {adgroupID}, target: {target}, matchtype: {matchtype}")
				print(df_bid)
				print(targetID)
				print(dataframe.select_row_by_val(df_bid, 'Target Id',targetID)['optimized'])
			if optimized:
				newbid=dataframe.select_row_by_val(df_bid, 'Target Id',targetID)['new_bid'].values[0]
				if math.isnan(newbid)==False:
					input_output.write_xls_bid(wb, ws, row[0].row,str(round(newbid,2)), campaign_type)

		
	input_output.save_xls_bid(wb, path)

if __name__ == "__main__":
	main()
