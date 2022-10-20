import array
import pandas as pd
import datetime
import numpy as np
from numpy.linalg import inv
from numpy import fft
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
import re
from sklearn import tree
from sklearn.tree import export_graphviz
import pydot
from treelib import Node, Tree as btree
#from pykalman import UnscentedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def main():
	#df  is the orignial dataframe
	#Zpreproc is the dataframe with feature
	#Z_filter is Zpreproc resampled weekly, aggregated by feature and filtered if feature has low observation


	df=pd.read_csv('C:/Users/jtran_000/Google Drive/growth scalers/Sonoma linen/Sponsored Products Targeting report.csv',header=0,)	#  index_col=0)	
	#We want to cluster data with Random Forest (segregate )good and bad conversion rate) and then forecast CR
	#We use feature like campaign, adgroup, keyword, Number_of_word, Match_Type, word in keywords which explain the most CR.
	#todo add Portfolio_name as feature
	#after clustering the data with Random Forest, we use a Kalman Filter to predict the expected conversion rate.

	print(df)
	df.columns = [c.replace(' ', '_') for c in df.columns]
	df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

	df['CR']=df['7_Day_Total_Units_(#)']/df['Clicks']

	#the more word a keyword has, the higher the CR
	df['Number_of_word']=df['Targeting'].str.count(' ')+1

	print(df)
	df=df.dropna()

	#encoding campaign, adgroup and match type
	lecampaign=LabelEncoder()
	leadgroup=LabelEncoder()
	#letargeting=LabelEncoder()
	lematchtype=LabelEncoder()
	#Z_preproc=pd.DataFrame(columns=['Campaign_Name','Ad_Group_Name','Match_Type','Number_of_word'])
	Z_preproc=df
	Z_preproc['Portfolio_name_enc']=lecampaign.fit_transform(df['Portfolio_name'])
	Z_preproc['Campaign_Name_enc']=lecampaign.fit_transform(df['Campaign_Name'])
	Z_preproc['Ad_Group_Name_enc']=leadgroup.fit_transform(df['Ad_Group_Name'])
	#Z_preproc['Targeting']=letargeting.fit_transform(df['Targeting'])
	Z_preproc['Match_Type_enc']=lematchtype.fit_transform(df['Match_Type'])
	#Z_preproc['Number_of_word']=df['Number_of_word'].values

	#enc=OneHotEncoder()
	vectorizer = CountVectorizer()
	print(df['Targeting'])
	vectorizer.fit(df['Targeting'])
	print(vectorizer.get_feature_names())
	word_vect = vectorizer.transform(df['Targeting'])

	#Todo remove symbols like []+
	print(word_vect.toarray())
	word=pd.DataFrame(data=word_vect.toarray(), index=Z_preproc.index, columns=vectorizer.get_feature_names())
	print(word)
	Z_preproc=Z_preproc.join(word)#,how='left')
	#Z_preproc[word_vect]=Z_preproc.lookup(word.index,word_vect)
	print("join")
	print(Z_preproc)
	Z_filter=Z_preproc

	#we want more robust estimate, daily CR is too noisy. Therefore we take weekly
	Z_filter=Z_filter.set_index('Date')
	Z_filter.resample('W') 
	print("resample")
	print(Z_filter)
	to_drop = [
		'Portfolio_name','Campaign_Name', 'Ad_Group_Name','Match_Type',
		'Currency','Click-Thru_Rate_(CTR)','Cost_Per_Click_(CPC)','Spend',
		'Total_Advertising_Cost_of_Sales_(ACoS)_','7_Day_Conversion_Rate', '7_Day_Advertised_SKU_Sales_', '7_Day_Other_SKU_Sales_',
		'Targeting', 'Impressions', 'Total_Return_on_Advertising_Spend_(RoAS)', '7_Day_Total_Orders_(#)',#'7_Day_Total_Units_(#)',
		'7_Day_Total_Sales_', '7_Day_Advertised_SKU_Units_(#)','7_Day_Other_SKU_Units_(#)', 'CR'
	]
	Z_filter.drop(to_drop, axis = 1, inplace=True)

	#TODO We aggregate by any feature but Targeting
	print(Z_filter)

	#filter out week with too few observation
	Z_filter=Z_filter[Z_filter['Clicks']>10]

	#We drop volum data to keep only features
	#Z_filter=Z_filter.drop(['Targeting', 'Impressions','Clicks'], axis = 1) 
	#Z_filter=Z_filter.drop(['Total_Return_on_Advertising_Spend_(RoAS)','7_Day_Total_Orders_(#)'], axis = 1) 
	#Z_filter=Z_filter.drop(['7_Day_Total_Units_(#)','7_Day_Advertised_SKU_Units_(#)','7_Day_Other_SKU_Units_(#)','CR'], axis = 1) 
	#Z_filter=Z_filter.drop(to_drop, axis = 1) 

	print(Z_filter.columns)

	#We calculate the weekly CR for each feature. 
	#TODO take CR=conv/(Click-1) to get the unbiased conversion rate (low number of clicks)
	Z_filter['CR_7days']=Z_filter['7_Day_Total_Units_(#)']/Z_filter['Clicks']*100
	CR_filter=Z_filter['CR_7days'].values
	Z_filter.drop(['Clicks','CR_7days','7_Day_Total_Units_(#)'], axis = 1, inplace=True)

	regr = RandomForestRegressor(min_impurity_decrease= 0.5, min_samples_split=2)
	regr.fit(Z_filter, CR_filter)
	feat_importances = pd.Series(regr.feature_importances_, index=Z_filter.columns)
	print("Random Forest feature importance \n: ")
	print(feat_importances.sort_values(ascending=False))
	feat_importances.sort_values(ascending=False).nlargest(10).plot(kind='barh')
	#plt.show()

	#This code below is not elegant at all but I couldn't find anything better
	#################################################################################################################
	# Pull out one tree from the forest
	tree = regr.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names =  Z_filter.columns, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')

	#tag each keyword to the belonging leave
	f = open('tree.dot', 'r')
	read_data = f.read()
	#out = StringIO()
	print(read_data)
	f.close()

	#read the tree in a more user-friendly way and save rules in dataframe
	dfrules=pd.DataFrame(columns=['Node','Rule'])
	
	#root=Node(0)
	parent=0
	child=0
	bintree=btree()
	bintree.create_node(0,0)
	bintree.show()
	for line in read_data.splitlines():
		#btree=line[0:line.find(' ')]
		if line.find('headlabel=\"True\"')>0: #first split
			parent=int(line[0:1])
			child=int(line[5:6])
			if bintree.get_node(parent).is_leaf():
				print()
				sign="+"
			else:
				sign="-"
			bintree.create_node(child,child,parent,sign)
		elif line.find('headlabel=\"False\"')>0:# last split
			parent=int(line[0:1])
			child=int(line[5:line.find(' [')])
			if bintree.get_node(parent).is_leaf():
				sign="+"
			else:
				sign="-"
			bintree.create_node(child,child,parent,sign)

		else:
			if line.find('->')>0:	#New node
				parent = int(line[0:line.find('->')-1])
				child = int(line[line.find('->')+3:line.find(';')-1])
				if bintree.get_node(parent).is_leaf():
					sign="+"
				else:
					sign="-"
				bintree.create_node(child,child,parent,sign)
			if line.find('<=')>0:
				rule=line[line.find('label=')+7:line.find('\\')]
				node_nb=int(line[0:line.find(' [')])
				dfrules=dfrules.append([{'Node':node_nb,'Rule':rule}])	

	bintree.show()
	dfrules['Node']=dfrules['Node'].astype(int)
	dfrules=dfrules.set_index('Node')
	print(dfrules)

	#some usefull functions
	#print(bintree.depth())
	#print(bintree.get_node("15"))
	#print(bintree.paths_to_leaves())
	############################################################################################################
	#tag data with the leave number
	Node_level_max="Node_"+str(bintree.depth())
	for path in bintree.paths_to_leaves():
		filters=""
		print(path)
		lvl=0
		for node in path [1:]:#-1]:
			Node_lvl="Node_level_"+str(lvl)
			#for each node we check if they are + or - and get the rule from the parent
			parent_rule=dfrules.loc[int(bintree.parent(node).identifier)]['Rule']
			if bintree.get_node(node).data =="-":
				parent_rule=parent_rule.replace("<",">")		
			#filters = filters + "(" + parent_rule + ")" + " & " #we take the unfiltered dataframe incl low volume keywords
			filters = filters + "(" + parent_rule + ")"
			Z_preproc.loc[Z_preproc.eval(filters),Node_lvl]=node#path[-1])
			filters = filters + " & "
			lvl=lvl+1
		#for the last item we don't want to add &
		#parent_rule=dfrules.loc[int(bintree.parent(path[-1]).identifier)]['Rule']
		#if bintree.get_node(path[-1]).data =="-":
		#	parent_rule=parent_rule.replace("<",">")

		#print(Node_lvl)
		#print(Node_level_max)
		#filters = filters + "Z_preproc." + parent_rule
		#filters = filters + "(" + parent_rule + ")"  #we take the unfiltered dataframe incl low volume keywords
		#Z_preproc.loc[Z_preproc.eval(filters),Node_lvl]=node#path[-1])
	print(Z_preproc)


	#now we need to have the clicks and the conversion rate for each node of the tree.
	XNode_KW=pd.DataFrame(columns=['Date','Clicks','7_Day_Total_Units_(#)'])
	node_cols = [col for col in Z_preproc.columns if 'Node_' in col]
	for node_col in node_cols:
		XNode_KW=XNode_KW.append(Z_preproc.reset_index().groupby([pd.Grouper(freq='1D', key='Date'),node_col])['Date','Clicks','7_Day_Total_Units_(#)'].sum().reset_index(level='Date'))
	XNode_KW.index=XNode_KW.index.set_names('Targeting')#that messy because nodes are not Targeting but for the sake of time...
	XNode_KW=XNode_KW.reset_index()

	print(Z_preproc[['Date','Targeting','Clicks','7_Day_Total_Units_(#)']])
	XNode_KW=XNode_KW.append(Z_preproc[['Date','Targeting','Clicks','7_Day_Total_Units_(#)']])
	XNode_KW['CR']=XNode_KW['7_Day_Total_Units_(#)']/XNode_KW['Clicks']
	print("unique Targeting :", XNode_KW['Targeting'].drop_duplicates())
	lastdayfrom = pd.to_datetime('09/15/2020')

	#We use a Unscented Kalman Filter because the distribution is not Gaussian and we want to predict the CR
	#for each keyword/biddable object.

	#Some reading about Kalman Filter
	#https://www.youtube.com/watch?v=fpRb1sjFz_M
	#https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
	#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
	#https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
	#https://stats.stackexchange.com/questions/183118/difference-between-hidden-markov-models-and-particle-filter-and-kalman-filter
	#http://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/
	#https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data

	#Some reading about Unscented Kalman Filter
	#https://github.com/pykalman/pykalman/blob/master/examples/unscented/plot_filter.py
	#https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
	#https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d
	#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb

	#Z is our observation. We want Z to be a vector of the conversion rate for each node and for each keyword(leaf). 
	#We need to estimate the Conversion rate accurately from the top to the buttom. We also want to inherate information from the top to the bottom
	#[Z=CR(node1,2,3....,keyword 1,2,3...)]


	#For each Node we inherite 20%/number of node in the branch, for instance :
	#if there is 5 nodes in the branch and we have an average of 1000 clicks for the root and the conversion rate of the root is 5%  
	#and the first node has 300  clicks with 1 conversion we add 20% of the information of the node/5 so 4% so if 300 clicks*4%=12 click@5%=0.6 conversions
	#so the conversion rate will be 1.6/(312)=0.512% instead of 1/300=0.333%


	# Initial Conditions
	#t = 1  # Difference in time one day

	# Initial State Matrix: we take the conversion rate of the 3 last months
	XNode_KW=XNode_KW.set_index('Date')
	XNode_KW=XNode_KW.sort_index()

	#get the CR for each Node/keyword for the last 30 days as initial state.
	X_hat_t=pd.DataFrame(index=XNode_KW['Targeting'].drop_duplicates(), columns=['CR'])
	X_hat_t['CR'] = XNode_KW.reset_index().groupby('Targeting')['7_Day_Total_Units_(#)'].sum() / (
		XNode_KW.reset_index().groupby('Targeting')['Clicks'].sum()) # We add one click, better would be to add click and conversion from parent
	
	#M is the measurment or conversion rate for each node 
	M_t=pd.DataFrame(index=XNode_KW['Targeting'].drop_duplicates(), columns=['7_Day_Conversion_Rate'])
	M_t= XNode_KW.loc[lastdayfrom - pd.Timedelta(days=7):lastdayfrom].reset_index().groupby('Targeting')['7_Day_Total_Units_(#)'].sum() / (
		XNode_KW.loc[lastdayfrom - pd.Timedelta(days=7):lastdayfrom].reset_index().groupby('Targeting')['Clicks'].sum())
	M_t=M_t.dropna()


	n=len(X_hat_t.dropna())
	'''

	#A = np.identity(n)
	A = pd.DataFrame(np.identity(n), index=X_hat_t.dropna().index, columns=X_hat_t.dropna().index)
	the following code can be use to get some information from the higher level
		for column in A:


		try: 
			A.loc[column].at[bintree.parent(column).identifier]=0.05/bintree.depth(column)	#we get 5	% of info from top level
			#A.at[column,column]=A.at[column,column]-0.1/bintree.depth(column)		#we rely less on measure and more on upper level
		except: 
			pass
	'''

	# Initial Estimation Covariance Matrix/Process Matrix. At first, we assume Q=0: There is no error in the process
	#P = covariance2d(error_est_x, error_est_v)
	#P_t= np.identity(n)*0.5
	P_t=pd.DataFrame(data=np.identity(n)*10, index=X_hat_t.dropna().index, columns=X_hat_t.dropna().index)
	
	#Measurment covartiance matrix
	#R_t= np.identity(n)*5
	#R_t=pd.DataFrame(data=np.identity(n)*10, index=X_hat_t.dropna().index, columns=X_hat_t.dropna().index)

	diagR=np.diag(np.diag(XNode_KW.reset_index()[['Date','Targeting','CR']].pivot_table(index='Date', columns='Targeting', values='CR', aggfunc=np.sum).cov().to_numpy()))
	R_t=pd.DataFrame(data=diagR,index=X_hat_t.dropna().index, columns=X_hat_t.dropna().index)/100
	print(R_t)
	X_pred=pd.DataFrame(index=['Date'],columns=['CR_forecast','CR'])

	#R_t=R_t.loc[X_hat_t.dropna().index]
	#R_t=R_t.cov()
	#R_t=R_t.fillna(0.8)#if no value we take high covariance to rely more on other values
	#R_t=R_t.clip(upper=0.8) #we cap covarience to avoid inf
	print(R_t)




	# create sigma points and weights
	mean=X_hat_t['CR'].to_numpy()
	print(mean)
	print(P_t.to_numpy())
	sigmas = MerweScaledSigmaPoints(n=1, alpha=.3, beta=2., kappa=.1)#alpha=.3, beta=2., kappa=.1)
	print(sigmas)
	# get fourrier fft for forecasting
	zs=df.reset_index().groupby('Date')['7_Day_Total_Units_(#)'].sum() / (
		df.reset_index().groupby('Date')['Clicks'].sum())

	zs=zs.to_numpy()

	global ampli
	global phase
	global freq
	global t 
	(ampli, phase, freq) = fourier_para(zs) #here we should take the CR for the last x
	print(ampli)
	print(phase)

	#ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=1., hx=hx, fx=fx, points=sigmas)
	ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=1., hx=hx, fx=fx, points=sigmas)
	#ukf.x= np.array([-1., 1., -1., 1])
	ukf.x= np.array([0.1])
	ukf.P *= 0.2
	#ukf.R = np.diag([0.1**2, 0.1**2])#.5
	ukf.R = np.diag([1**2])#.5 #we need to set R to a high number if we have few mesurement. R=cov + 10/#click
	from filterpy.common import Q_discrete_white_noise
	ukf.Q =[[0.01]]#Q_discrete_white_noise(2, dt=1., var=0.01**2, block_size=2)
	print(ukf.Q)
	#zs = [[i+randn()*0.1, i+randn()*0.1] for i in range(50)] # measurements
	#zs=[[0.1],[0.11],[0.1],[0.14],[0.16]]#df['CR'].to_numpy()



	x, cr = [], []

	t=0
	for z in zs:

		ukf.predict()
		t+=1
		ukf.update(z)
		print(ukf.x, 'log-likelihood', ukf.log_likelihood)
		x.append(ukf.x)
		cr.append(z)
	plt.plot(x)
	#zs = [t + random.randn()*4 for t in range (40)]
	print(zs)
	plt.plot(cr)
	plt.show()
	
	(mu, cov) = ukf.batch_filter(zs)
	(x, P, K) = ukf.rts_smoother(mu, cov)
	plt.plot(x)
	plt.plot(cr)
	#plt.plot(cr, ls='--', lw=2, c='k')
	plt.show()
	'''
	#This is as prove of concept to see how well it predicts. We want to compare prediction again historical values.
	for day in XNode_KW.loc[lastdayfrom:pd.to_datetime('12/01/2020')].index.unique():#pd.to_datetime("today")].index.unique(): #for every day
		print(day)#.to_numpy())

		print(XNode_KW[day - pd.Timedelta(days=7):day])							#get last 7 days CR
		M_t=pd.DataFrame(index=XNode_KW['Targeting'].drop_duplicates(), columns=['7_Day_Conversion_Rate'])
		#M_t=M_t.rename('Targeting')
		
		#not_NA = lambda s1, s2: s1 if s1.notna else s2
		M_t= XNode_KW.loc[day - pd.Timedelta(days=1):day].reset_index().groupby('Targeting')['7_Day_Total_Units_(#)'].sum() / (
		XNode_KW.loc[day - pd.Timedelta(days=1):day].reset_index().groupby('Targeting')['Clicks'].sum())#,not_NA)#, how='outer')
		print("last 7 days measure :",M_t)


		#https://math.stackexchange.com/questions/982982/kalman-filter-with-missing-measurement-inputs
		#For a missing measurement, just use the last state estimate as a measurement but set the covariance matrix of the measurement to essentially infinity. 
		#(If the system uses inverse covariance just set the values to zero.) This would cause a Kalman filter to essentially ignore the new measurement since 
		#the ratio of the variance of the prediction to the measurement is zero. The result will be a new prediction that maintains velocity/acceleration but 
		#whose variance will grow according to the process noise.

		M_t=X_hat_t.join(M_t.rename('Measurment'), on='Targeting', how='outer')# no need to update conversion rate if we don't have new observation.
		M_t['CR'] = np.where(M_t['Measurment'].notna(), M_t['Measurment'], M_t['CR'])
		M_t=M_t.drop('Measurment', axis=1)


		ukf.predict()
		print("predicted state :",ukf.x[0])
		#X_hat_t, P_hat_t  = prediction(X_hat_t, P_t, A)
		#print("predicted state :",X_hat_t)

		#H_t = np.identity(n)								#H is a transformation matrix to convert P to K
		H_t=pd.DataFrame(data=np.identity(len(X_hat_t.dropna())), index=X_hat_t.dropna().index, columns=X_hat_t.dropna().index)


		ukf.update(M_t.to_numpy())
		'''
	'''
		X_t,P_t=update(X_hat_t,P_hat_t,M_t,R_t,H_t)
		X_hat_t=X_t
		P_hat_t=P_t
		print("updated state :",X_t)
		print("process covariance:",P_t)
		print("measurement covariance:",R_t)
		X_pred=X_pred.append(pd.DataFrame(data=[X_t['CR'].iloc[0]], columns=['CR_forecast'],index=[pd.to_datetime(day).strftime("%Y-%m-%d")]))
	'''
	'''print(X_pred)
	print(df.groupby('Date')['7_Day_Total_Units_(#)'].sum() / df.groupby('Date')['Clicks'].sum())
	X_pred['CR']=(df.groupby('Date')['7_Day_Total_Units_(#)'].sum() / df.groupby('Date')['Clicks'].sum())
	print(X_pred)
	X_pred[['CR_forecast']].plot()
	plt.show()
	(df.groupby('Date')['7_Day_Total_Units_(#)'].sum() / df.groupby('Date')['Clicks'].sum()).plot()
	plt.show()'''

def get_conversion_rate(DataFrame, fromdate, untildate, groupby):

	print("do something")

def fourier_para(cr):
	# https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
	print(cr)
	n = cr.size
	n_harm = 5                     	# number of harmonics in model
	t = np.arange(0, n)
	p = np.polyfit(t, cr, 1)         	# find linear trend in x
	cr_notrend = cr - p[0] * t  		# detrended x
	cr_freqdom = fft.fft(cr_notrend)	# detrended x in frequency domain
	f = fft.fftfreq(n)              	# frequencies
	indexes = list(range(n))

	# sort indexes by frequency, lower -> higher
	indexes.sort(key = lambda i: np.absolute(f[i]))

	#t = np.arange(0, n + n_predict)
	#restored_sig = np.zeros(t.size)
	ampli = []
	phase = []
	freq = []
	'''
	for i in indexes[:1 + n_harm * 2]:
		ampli = np.absolute(x_freqdom[i]) / n   # amplitude
		phase = np.angle(x_freqdom[i])          # phase
		restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
	return restored_sig + p[0] * t'''
	for i in indexes[:1 + n_harm * 2]:
		if np.absolute(1/f[i])>=7:
			ampli.append(np.absolute(cr_freqdom[i]) / n)  # amplitude
			phase.append(np.angle(cr_freqdom[i]))         # phase
			freq.append(f[i])
	return ampli, phase, freq



def prediction(X_hat_t_1, P_t_1, A):

	#X State matrix= Cr of each of the leaves of the random forrest and of the CR at porfolio level/campaign
	# The CR of adgroup and campaign/porfolio is a weighted average agregate value of the keywords.
	#A= id(len(leaves)) for the diagonal. In other word, CR doesn't change with time
	#B= information from the higher level
    #A = np.identity(len(X_hat_t_1))
    #A = pd.DataFrame(data=np.identity(len(X_hat_t_1.dropna())), index=X_hat_t_1.dropna().index, columns=X_hat_t_1.dropna().index)##better to use panda because numpy doesn't handle NaN well
    X_hat_t= pd.DataFrame(index=X_hat_t_1.index, columns=X_hat_t_1.columns)
    #P_t=pd.DataFrame(index=X_hat_t_1.index, columns=X_hat_t_1.index)

    #B is the information from the upper level. To make it easier right now consider B to be 0. 
    #Later we can consider B as being 20% / number of node in the tree. At the end, we might need to consider the volume ratio
    #B = dfVolNode_ration * 20%/number_of_node
    #B=np.zeros(len(X_hat_t_1))										#B is CR of the parent of X
    B=pd.DataFrame(np.zeros(len(X_hat_t_1.dropna())))
    ##X_prime = A.dot(X) + B.dot(X)
    ##return X_prime

    #X_hat_t=A.dot(X_hat_t_1)#+(B_t.dot(U_t).reshape(B_t.shape[0],-1) )
    #X_hat_t=A.dot(np.nan_to_num(X_hat_t_1))	
    #print(X_hat_t_1)
    #X_hat_t=A.dot(X_hat_t_1)
    #X_hat_t.loc[X_hat_t_1.dropna().index]=A.dot(X_hat_t_1.dropna())
    X_hat_t.loc[X_hat_t_1.dropna().index]=A.dot(X_hat_t_1.loc[X_hat_t_1.dropna().index])
    #print("X_hat_t:\n",X_hat_t)

    X_hat_t=X_hat_t.clip(lower=0) #conversion rate can not be negatif
    #print(P_t_1)
    #print(np.diag(np.diag(A.dot(P_t_1).dot(A.transpose()))))
    #P_t=np.diag(np.diag(A.dot(P_t_1).dot(A.transpose())))#+Q_t
    P_t=pd.DataFrame(data=np.diag(np.diag(A.dot(P_t_1).dot(A.transpose()))),index=X_hat_t_1.dropna().index, columns=X_hat_t_1.dropna().index)

    return X_hat_t, P_t

def update(X_hat_t,P_t, M_t,R_t,H_t):
   
	S=pd.DataFrame(H_t.dot(P_t).dot(H_t) + R_t, X_hat_t.dropna().index, X_hat_t.dropna().index)
	print(S)
	K_prime=P_t.dot(H_t).div(S)									#We can use div instead of inverting because S is a diagonal matrix
	#K_prime.iloc[0,0]=0.9
	#K_prime=P_t.dot(H_t.transpose()).dot( np.linalg.inv ( H_t.dot(P_t).dot(H_t.transpose()) +R_t ) )   
	K_prime=K_prime.fillna(0)									#high K means we rely more on measurment, low K we rely more on upper level
	print("K_prime :", K_prime)				
	#X_t=X_hat_t+K_prime.dot(M_t-H_t.dot(X_hat_t))
	print("H_t*Xhat :",H_t.dot(X_hat_t))
	X_t=X_hat_t+K_prime.dot(M_t - H_t.dot(X_hat_t))
	P_t=P_t-K_prime.dot(H_t).dot(P_t)

	return X_t,P_t


def covariance2d(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return np.diag(np.diag(cov_matrix))

def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors



def fx(x, dt):
	# state transition function - predict next state based
	# to predict conversion rate we assume that only weekly seasonality matters,
	# we calculate by how much the CR increase for a the day of the week and apply this to all nodes.
	# Therefore, we use FFT to predict or ARIMA. Another option would be to pass as input the CR by day + for this specific day of the week
	#https://www.thelearningpoint.net/computer-science/stats-ml-data-time-series-forecasting-arima-fourier-regression-methods
	#https://towardsdatascience.com/analyzing-seasonality-with-fourier-transforms-using-python-scipy-bb46945a23d3
	'''F = np.array([[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]], dtype=float)'''
	#t=1
	F=np.ones((len(x),1)) 
	for i in range(len(ampli)) :
		F += ampli[i] * np.cos(2 * np.pi * freq[i] * t + phase[i])	
	print("F is ",F)
	#F = np.array([[1]])
	return np.dot(F, x)
	
def hx(x):
	# measurement function - convert state into a measurement
	# where measurements are [x_pos, y_pos]
	#return np.array([x[0], x[2]])#np.array([[1]])#np.array([x[0], x[2]])
	return np.array([x[0]])




def frequency():
	#we can check how often keywords convert and take 2 times the freq max in consideration for the Kalman weight
	#actually we can take the number of conversion /day for each node
	print("")






def merge():
	#to merge we need to group kw with low volume whitch are in the same leave and the same adrgoup.
	#in doing so we avoid bid adjustment and other settings to interfere not evenly
	#
	print("")

if __name__ == "__main__":
	main()
