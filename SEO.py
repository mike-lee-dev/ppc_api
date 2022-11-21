import pandas as pd
import nltk
from utils import input_output, dataframe
from nltk.corpus import stopwords
nltk.download('stopwords')

def main():
    path='./data wwff/'

    dfsqr = input_output.read_sqr(path)

    ## making sure the 7 days order is >=1 
    dfsqr = dfsqr.drop(dfsqr[dfsqr['7 Day Total Sales '] < 1].index)


    dfsqr_filtered = pd.DataFrame(dfsqr[['Customer Search Term',  '7 Day Total Sales ', 'Campaign Name', 'Ad Group Name']])


    dfbid = input_output.read_bid('SP',path)
    dfproductad = dataframe.select_row_by_val(dfbid, 'Entity','Product Ad')
    dfproductad = pd.DataFrame(dfproductad[['Campaign Id',  'Ad Group Id', 'Campaign Name', 'Ad Group Name', 'SKU']])
    print(dfproductad)

    SKU_list = []
    for index, row in dfsqr_filtered.iterrows():
        if row['Customer Search Term'][0:2] == 'b0':
            dfsqr_filtered = dfsqr_filtered.drop(index)
            continue
        #if row['Campaign Name'] == 'gloves manual':
            #row['Campaign Name'] = 'Gloves - SP - manual'
        #print(row)
        campaign_id = dataframe.select_row_by_val(dfproductad,'Campaign Name',row['Campaign Name'])['Campaign Id'].iloc[0]
        #print(campaign_id)
        ad_group_id = dataframe.select_row_by_val(dfproductad,'Campaign Name',row['Campaign Name'],'Ad Group Name',row['Ad Group Name'])['Ad Group Id'].iloc[0]
        #campaign_id = dfproductad['Campaign Id'][dfproductad['Campaign Name'] == row['Campaign Name']].tolist()#[0]
        #ad_group_id = dfproductad['Ad Group Id'][dfproductad['Ad Group Name'] == row['Ad Group Name']].tolist()
        #print(campaign_id)
        #print(ad_group_id)

        #sku_values = dfproductad['SKU'][(dfproductad['Campaign Id'] == campaign_id) & (dfproductad['Ad Group Id'].isin(ad_group_id))]
        #for value in sku_values:
        #    if not pd.isnull(value):
        #        sku= value
                #break
        #        SKU_list.append(sku)
        sku_values=dataframe.select_row_by_val(dfproductad, 'Campaign Id', campaign_id, 'Ad Group Id', ad_group_id)#['SKU'].tolist()
        #sku_values = dfproductad['SKU'][(dfproductad['Campaign Id'] == campaign_id) & (dfproductad['Ad Group Id'].isin(ad_group_id))]
        for value in sku_values:
            if not pd.isnull(value):
                sku= value
                break
        SKU_list.append(sku)
    dfsqr_filtered['SKU'] = SKU_list
    print(dfsqr_filtered)
    

    category_listing = input_output.read_category_listing_report(path)
    print(category_listing)
    #category_listing.columns= category_listing.iloc[0]
    category_listing.drop(index = [0], inplace = True)
    #category_listing.columns
    print(category_listing)

    category_listing = pd.DataFrame(category_listing[['Seller SKU', 'Search Terms', 'Product Name']])
    print(category_listing)

    results = pd.DataFrame(columns = ['SKU', 'missing word', '7 Day Total Sales'])
    for index, row in dfsqr_filtered.iterrows():
        search_terms = category_listing['Search Terms'][category_listing['Seller SKU'] == row['SKU']].tolist()[0]
        product_names = category_listing['Product Name'][category_listing['Seller SKU'] == row['SKU']].tolist()[0]
        missing_words = []
        for word in row['Customer Search Term'].split():
            if word in set(stopwords.words('english')):
                continue
            if word not in search_terms and word.lower() not in product_names.lower() :
                results = results.append({'SKU': row['SKU'], 'missing word': word, '7 Day Total Sales': row['7 Day Total Sales ']}, ignore_index = True)

    duplicates = results.duplicated(keep='first')
    results= results[~duplicates]

    # # Save to excel file
    # results.to_excel('missing_word.xlsx')

    # results = pd.DataFrame(columns = ['SKU', 'missing word', '7 Day Total Sales'])
    # for index, row in dfsqr_filtered.iterrows():
    #     search_terms = category_listing['Search Terms'][category_listing['Seller SKU'] == row['SKU']].tolist()[0]
    #     missing_words = ''
    #     for word in row['Customer Search Term'].split():
    #         if word in set(stopwords.words('english')):
    #             continue
    #         if word not in search_terms:
    #             missing_words+= word + ' '
        
    #     results = results.append({'SKU': row['SKU'], 'missing word': missing_words, '7 Day Total Sales': row['7 Day Total Sales ']}, ignore_index = True)	

    # duplicates = results.duplicated(keep='first')
    # results= results[~duplicates]
    # results.shape

    # results = pd.DataFrame(columns = ['SKU', 'missing word', '7 Day Total Sales'])
    # for index, row in dfsqr_filtered.iterrows():
    #     search_terms = category_listing['Search Terms'][category_listing['Seller SKU'] == row['SKU']].tolist()[0]
    #     missing_words = ''
    #     for word in row['Customer Search Term'].split():
    #         if word in set(stopwords.words('english')):
    #             continue
    #         if word not in search_terms:
    #             missing_words+= word + ' '
        
    #     results = results.append({'SKU': row['SKU'], 'missing word': missing_words, '7 Day Total Sales': row['7 Day Total Sales ']}, ignore_index = True)

    # results.shape

    # duplicates = results.duplicated(keep='first')
    # results= results[~duplicates]
    # results.shape

    # Save to excel file
    results.to_excel('missing_words_list.xlsx')

if __name__ == "__main__":
    main()
