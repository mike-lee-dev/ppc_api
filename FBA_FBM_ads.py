from utils import input_output, dataframe
import shutil
import pandas as pd
import math

def main():
	path='./data wwff/'
	filename='output/bulk_FBA_FBM.xlsx'
	wb_obj=input_output.open_wb(path, file=filename)
	ws=input_output.open_ws(wb_obj,'SP')
	df_list=input_output.active_listing(path)[['asin1','seller-sku','fulfillment-channel']].astype(str)
	print(df_list)
	ASINs=pd.DataFrame(df_list['asin1'].unique(), columns=['asin1']).set_index('asin1')
	print(ASINs)
	print(dataframe.select_row_by_val(df_list,'fulfillment-channel','DEFAULT'))
	SKUs=ASINs.join(dataframe.select_row_by_val(df_list,'fulfillment-channel','DEFAULT')[['asin1','seller-sku']].set_index('asin1'))#, rsuffix='_FBM')
	SKUs.rename(columns={'seller-sku': 'seller-sku_FBM'}, inplace=True)
	print(SKUs)
	SKUs=SKUs.join(dataframe.select_row_by_val(df_list,'fulfillment-channel','AMAZON_NA')[['asin1','seller-sku']].set_index('asin1'))#, rsuffix='_FBA')
	SKUs.rename(columns={'seller-sku': 'seller-sku_FBA'}, inplace=True)
	print(SKUs)
	i=0
	for row in ws.iter_rows(min_row=2):
		entity=row[1].value
		if entity =='Product Ad':
			SKU_row=row[21].value
			try:	
				FBA_SKU=dataframe.select_row_by_val(SKUs, 'seller-sku_FBM', SKU_row)['seller-sku_FBA'].iloc[0]
				if is_nan(FBA_SKU)==False:
					print(FBA_SKU)
					row[21].value=FBA_SKU
					row[6].value=str(FBA_SKU)+str(i)
					i=i+1
					row[2].value='Create'	
			except IndexError:
				pass
	
			try:	
				FBM_SKU=dataframe.select_row_by_val(SKUs, 'seller-sku_FBA', SKU_row)['seller-sku_FBM'].iloc[0]
				if is_nan(FBM_SKU)==False:
					row[21].value=FBM_SKU
					row[6].value=str(FBA_SKU)+str(i)
					i=i+1
					row[2].value='Create'	
			except IndexError:
				pass
	ws=input_output.open_ws(wb_obj,'SD')
	for row in ws.iter_rows(min_row=2):
		entity=row[1].value
		if entity =='Product Ad':
			SKU_row=row[21].value
			try:	
				FBA_SKU=dataframe.select_row_by_val(SKUs, 'seller-sku_FBM', SKU_row)['seller-sku_FBA'].iloc[0]
				if is_nan(FBA_SKU)==False:
					row[21].value=FBA_SKU
					row[6].value=str(FBA_SKU)+str(i)
					i=i+1
					row[2].value='Create'	
			except IndexError:
				pass
	
			try:	
				FBM_SKU=dataframe.select_row_by_val(SKUs, 'seller-sku_FBA', SKU_row)['seller-sku_FBM'].iloc[0]
				if is_nan(FBM_SKU)==False:
					row[21].value=FBM_SKU
					row[6].value=str(FBA_SKU)+str(i)
					i=i+1
					row[2].value='Create'	
			except IndexError:
				pass
	input_output.save_xls_bid(wb_obj, path,filename)
def is_nan(x):
    return (x != x)



if __name__ == "__main__":
	main()
