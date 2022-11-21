import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

def main():
	df=pd.read_csv('C:/Users/jtran/Google Drive/growth scalers/Sonoma linen/Sponsored Products Targeting report.csv',header=0,)

	df.columns = [c.replace(' ', '_') for c in df.columns]
	df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

	############################################ Enconding ###########################################################

	# the more word a keyword has, the higher the CR
	df['Number_of_word']=df['Targeting'].str.count(' ')+1
	print(df)

	# encoding campaign, adgroup and match type
	lecampaign=LabelEncoder()
	leadgroup=LabelEncoder()
	#letargeting=LabelEncoder()
	lematchtype=LabelEncoder()
	RF_preproc=df
	RF_preproc['Portfolio_name_enc']=lecampaign.fit_transform(df['Portfolio_name'])
	RF_preproc['Campaign_Name_enc']=lecampaign.fit_transform(df['Campaign_Name'])
	RF_preproc['Ad_Group_Name_enc']=leadgroup.fit_transform(df['Ad_Group_Name'])
	#RF_preproc['Targeting']=letargeting.fit_transform(df['Targeting']) -> This can lead to overfitting
	RF_preproc['Match_Type_enc']=lematchtype.fit_transform(df['Match_Type'])

	#split word by word. This might be useful if one word is better than another
	vectorizer = CountVectorizer()
	print(df['Targeting'])
	vectorizer.fit(df['Targeting'])
	print(vectorizer.get_feature_names())
	word_vect = vectorizer.transform(df['Targeting'])

	# Todo check if it removes symbols like []+
	word=pd.DataFrame(data=word_vect.toarray(), index=RF_preproc.index, columns=vectorizer.get_feature_names())
	RF_preproc=RF_preproc.join(word)
	
	#split by pair of words to get the order.
	vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
	vectorizer2.fit(df['Targeting'])
	word_vect2 = vectorizer2.transform(df['Targeting'])

	word2=pd.DataFrame(data=word_vect2.toarray(), index=RF_preproc.index, columns=vectorizer2.get_feature_names())
	word2.columns=word2.columns.str.replace(' ','_')	 		# This is needed to eval the filter later.
	RF_preproc=RF_preproc.join(word2)

	print(RF_preproc)
	RF_filter=RF_preproc
	
	############################################ Feature Selection ########################################################
	# we want more robust estimate, daily CR is too noisy. Therefore we take weekly
	RF_filter=RF_filter.set_index('Date')
	#RF_filter.resample('W') 
	#print("resample")
	print(RF_filter)
	to_drop = [
		'Portfolio_name','Campaign_Name', 'Ad_Group_Name','Match_Type',
		'Currency','Click-Thru_Rate_(CTR)','Spend',
		'Total_Advertising_Cost_of_Sales_(ACoS)_','7_Day_Conversion_Rate', '7_Day_Advertised_SKU_Sales_', '7_Day_Other_SKU_Sales_',
		'Targeting', 'Impressions', 'Total_Return_on_Advertising_Spend_(RoAS)', '7_Day_Total_Orders_(#)',#'7_Day_Total_Units_(#)',
		'7_Day_Total_Sales_', '7_Day_Advertised_SKU_Units_(#)','7_Day_Other_SKU_Units_(#)', #'CR'
	]
	RF_filter.drop(to_drop, axis = 1, inplace=True)
	RF_filter=RF_filter.dropna(axis = 0)

	print(RF_filter)

	############################################ Random Forest #######################################################
	RF_filter['Cost_Per_Click_(CPC)']=RF_filter['Cost_Per_Click_(CPC)'].str.replace("$","")
	y=RF_filter['Cost_Per_Click_(CPC)'].values
	print(y)
	X=RF_filter.drop(['Cost_Per_Click_(CPC)'], axis = 1).values
	print(X)

	regr = RandomForestRegressor(min_impurity_decrease= 0.1, min_samples_split=2)
	regr.fit(X, y)
	print(regr.feature_importances_)
	feat_importances = pd.Series(regr.feature_importances_, index=RF_filter.drop(['Cost_Per_Click_(CPC)'], axis = 1).columns)
	print("Random Forest feature importance \n: ")
	print(feat_importances.sort_values(ascending=False))
	#feat_importances.sort_values(ascending=False).nlargest(10).plot(kind='barh')

if __name__ == "__main__":
	main()
