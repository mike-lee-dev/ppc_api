import sys
import array
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import math
from numpy.linalg import inv
from numpy import fft
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import cholesky
from scipy.linalg import block_diag
from sklearn import tree
from sklearn.tree import export_graphviz
import scipy
import pydot
from treelib import Node, Tree as btree
#from pykalman import UnscentedKalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver 
from filterpy.common import Q_discrete_white_noise
#from filterpy.common import Q_discrete_white_noise, pretty_str

#######https://filterpy.readthedocs.io/en/latest/kalman/EnsembleKalmanFilter.html
#Todo we should estimate order rate (DT), purchase per order (DT) and value per perchase (price avg) separately


def main():
	# df  is the orignial dataframe
	# RFpreproc is the dataframe with feature
	# RF_filter is RFpreproc resampled weekly, aggregated by feature and filtered if feature has low observation
	lastdayfrom = date_ymd(pd.to_datetime('15/11/2020'))

	############################################ Data prep ###########################################################	
	df=pd.read_csv('C:/Users/jtran_000/Google Drive/growth scalers/Sonoma linen/Sponsored Products Targeting report.csv',header=0,)	#  index_col=0)	
	# We want to cluster data with Random Forest (segregate )good and bad conversion rate) and then forecast CR
	# We use feature like campaign, adgroup, keyword, Number_of_word, Match_Type, word in keywords which explain the most CR.
	# todo add Portfolio_name as feature
	# after clustering the data with Random Forest, we use a Kalman Filter to predict the expected conversion rate.

	print(df)
	df.columns = [c.replace(' ', '_') for c in df.columns]
	df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

	'''
	############################################ Enconding ###########################################################

	# the more word a keyword has, the higher the CR
	df['Number_of_word']=df['Targeting'].str.count(' ')+1

	print(df)
	df=df.dropna()

	# encoding campaign, adgroup and match type
	lecampaign=LabelEncoder()
	leadgroup=LabelEncoder()
	#letargeting=LabelEncoder()
	lematchtype=LabelEncoder()
	#RF_preproc=pd.DataFrame(columns=['Campaign_Name','Ad_Group_Name','Match_Type','Number_of_word'])
	RF_preproc=df
	RF_preproc['Portfolio_name_enc']=lecampaign.fit_transform(df['Portfolio_name'])
	RF_preproc['Campaign_Name_enc']=lecampaign.fit_transform(df['Campaign_Name'])
	RF_preproc['Ad_Group_Name_enc']=leadgroup.fit_transform(df['Ad_Group_Name'])
	#RF_preproc['Targeting']=letargeting.fit_transform(df['Targeting'])
	RF_preproc['Match_Type_enc']=lematchtype.fit_transform(df['Match_Type'])
	#RF_preproc['Number_of_word']=df['Number_of_word'].values

	#enc=OneHotEncoder()
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
	RF_filter.resample('W') 
	print("resample")
	print(RF_filter)
	to_drop = [
		'Portfolio_name','Campaign_Name', 'Ad_Group_Name','Match_Type',
		'Currency','Click-Thru_Rate_(CTR)','Cost_Per_Click_(CPC)','Spend',
		'Total_Advertising_Cost_of_Sales_(ACoS)_','7_Day_Conversion_Rate', '7_Day_Advertised_SKU_Sales_', '7_Day_Other_SKU_Sales_',
		'Targeting', 'Impressions', 'Total_Return_on_Advertising_Spend_(RoAS)', '7_Day_Total_Orders_(#)',#'7_Day_Total_Units_(#)',
		'7_Day_Total_Sales_', '7_Day_Advertised_SKU_Units_(#)','7_Day_Other_SKU_Units_(#)', #'CR'
	]
	RF_filter.drop(to_drop, axis = 1, inplace=True)

	# TODO We aggregate by any feature but Targeting
	print(RF_filter)

	# filter out week with too few observation
	RF_filter=RF_filter[RF_filter['Clicks']>10]


	print(RF_filter.columns)

	############################################ Random Forest #######################################################

	# We calculate the weekly CR for each feature. 
	#TODO take CR=conv/(Click-1) to get the unbiased conversion rate (low number of clicks)
	RF_filter['CR_7days']=RF_filter['7_Day_Total_Units_(#)']/RF_filter['Clicks']*100
	CR_filter=RF_filter['CR_7days'].values
	RF_filter.drop(['Clicks','CR_7days','7_Day_Total_Units_(#)'], axis = 1, inplace=True)

	regr = RandomForestRegressor(min_impurity_decrease= 0.1, min_samples_split=2)
	regr.fit(RF_filter, CR_filter)
	feat_importances = pd.Series(regr.feature_importances_, index=RF_filter.columns)
	print("Random Forest feature importance \n: ")
	print(feat_importances.sort_values(ascending=False))
	feat_importances.sort_values(ascending=False).nlargest(10).plot(kind='barh')

	############################################ Binear tree ##########################################################

	# This code below is not elegant at all but I couldn't find anything better
	###################################################################################################################
	# Pull out one tree from the forest
	tree = regr.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names =  RF_filter.columns, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')

	# tag each keyword to the belonging leave
	f = open('tree.dot', 'r')
	read_data = f.read()
	#out = StringIO()
	print(read_data)
	f.close()

	# read the tree in a more user-friendly way and save rules in dataframe
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



	# tag data with the leave number and save matrix of hierarchy
	depth=bintree.depth()
	all_nodes=len(bintree.all_nodes())
	inheritance_ratio=0.4
	global matrix_tree
	matrix_tree=np.zeros((all_nodes,all_nodes))
	#matrix_tree[0,0]=1
	Node_level_max="Node_"+str(bintree.depth())
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(linewidth=3200)
	matrix_tree[0,0]=1
	RF_preproc=RF_preproc.assign(Node_level_0=0) #node 0 is all data	
	for path in bintree.paths_to_leaves():
		filters=""
		print(path)
		#leaf=path[-1]
		previous=0
		for node in path: #save hierarchy and coefficient
			if node>0:
				matrix_tree[previous,node]=inheritance_ratio 
				matrix_tree[node,node]=1-inheritance_ratio
				#matrix_tree[leaf,node]=inheritance_ratio/(len(path))/2 
			previous=node


		lvl=1 #tag data with leave number

		for node in path[1:]:
			Node_lvl="Node_level_"+str(lvl)
			# for each node we check if they are + or - and get the rule from the parent
			parent_rule=dfrules.loc[int(bintree.parent(node).identifier)]['Rule']
			if bintree.get_node(node).data =="-":
				parent_rule=parent_rule.replace("<",">")		
			# we take the unfiltered dataframe incl low volume keywords
			filters = filters + "(" + parent_rule + ")"
			RF_preproc.loc[RF_preproc.eval(filters),Node_lvl]=node#path[-1])
			filters = filters + " & "
			lvl=lvl+1
	print(RF_preproc)

	
	# now we need to have the clicks and the conversion rate for each node of the tree.
	BTNode=pd.DataFrame(columns=['Date','Clicks','7_Day_Total_Units_(#)'])
	node_cols = [col for col in RF_preproc.columns if 'Node_' in col]
	for node_col in node_cols:
		BTNode=BTNode.append(RF_preproc.reset_index().groupby([pd.Grouper(freq='1D', key='Date'),node_col])['Date','Clicks','7_Day_Total_Units_(#)'].sum().reset_index(level='Date'))
	BTNode.index=BTNode.index.set_names('Node')#that messy because nodes are not Targeting but for the sake of time...
	BTNode=BTNode.reset_index()

	#BTNode=BTNode.append(RF_preproc[['Date','Targeting','Clicks','7_Day_Total_Units_(#)']])
	BTNode['CR']=BTNode['7_Day_Total_Units_(#)']/BTNode['Clicks']
	'''
	#print("unique Targeting :", BTNode['Targeting'].drop_duplicates())


	############################################ Kalman Filter & Prediction ##########################################
	# We ássume the distribution is Gaussian and the problem is a binomial problem: either a click convert or not. 
	# We want to predict the CR for each keyword/biddable object.

	# variance and sample:
	# https://keithmcnulty.github.io/bayes_analysis/global_water.html
	# https://courses.lumenlearning.com/introstats1/chapter/mean-or-expected-value-and-standard-deviation/#:~:text=To%20calculate%20the%20standard%20deviation%20(%CF%83)%20of%20a%20probability%20distribution,and%20take%20the%20square%20root.
	# the probability density function is given by:
	# P(C,N|p)=(C+N)!/(C!N!)*p^C*(1-p)^N where p is the probability
	# C the number of click which converted, N the number of clicks which didn't converted
	# https://docs.google.com/spreadsheets/d/1uFsp0xXAw9jiyzEgQhlBqGHef7RzQbnnBJgwmHDktzQ/edit#gid=1270622899
	# the mean and standard deviation can be calculated either by the sum product of the probability distribution or as follow:
	# μ=conversion/# of clicks
	# σ=(1/#clicks)*(#conversion/#clicks)*(#not conversion/#clicks)

	# Some reading about Kalman Filter
	# https://www.youtube.com/watch?v=fpRb1sjFz_M
	# https://medium.com/@jaems33/understanding-kalman-filters-with-python-2310e87b8f48
	# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
	# https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
	# https://stats.stackexchange.com/questions/183118/difference-between-hidden-markov-models-and-particle-filter-and-kalman-filter
	# http://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/
	# https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data

	# Some reading about Unscented Kalman Filter
	# https://github.com/pykalman/pykalman/blob/master/examples/unscented/plot_filter.py
	# https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html
	# https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d
	# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb

	# Z is our observation. We want Z to be a vector of the conversion rate for each node and for each keyword(leaf). 

	###################################################################################################################
	# not sure if we will do what follow:
	# For each Node we inherite 20%/number of node in the branch, for instance :
	# if there is 5 nodes in the branch and we have an average of 1000 clicks for the root and the conversion rate of the root is 5%  
	# and the first node has 300  clicks with 1 conversion we add 20% of the information of the node/5 so 4% so if 300 clicks*4%=12 click@5%=0.6 conversions
	# so the conversion rate will be 1.6/(312)=0.512% instead of 1/300=0.333%
	###################################################################################################################

	# Kalman variables:
	# X [1,n] is the state of the process (the conversion rate for each node)
	# M [1,n] is the measurment matrix
	# P [n,n] is the process covariance matrix, will be update by the Kalman filter. That the variance of the system.
	# Q is the noise in the process.(covariance matrix associated with the noise in states)
	# R [n,n] is the measurement variance.
	# H [n,n] is the measurement function. Converting the measurement to the state so right now 1 because we have 1 state, 1 measure.	
	# F [n,n](also called A in the litterature) is the transition matrix to predict the next step based on the current state. 
	# It can be either an identity matrix or predict from the parent-children relationship
	# B [1,n] is control transition matrix. The matrix models weekly seasonality
	# K [n,n] increase and we trust more the data. if P smaller than 1, K decreases and we trust more the prediction


	# Initial Conditions
	#t = 1  # Difference in time one day

	# get the CR for each Node for each day.
	#pd.DataFrame(index=BTNode['Node'].unique(), columns=['CR'])
	M_t=pd.DataFrame(columns=['Clicks','Conversions','CR'])
	M_t['Clicks']= df.groupby('Date')['Clicks'].sum()#pd.DataFrame(index=BTNode['Node'].unique(), columns=['CR'])
	M_t['Conversions']= df.groupby('Date')['7_Day_Total_Units_(#)'].sum()
	M_t['CR'] = M_t['Conversions'].values / (M_t['Clicks'])
	print(M_t)

	X_0=pd.DataFrame(columns=['Clicks','Conversions','CR'])
	X_0['CR'] = M_t['Clicks'].sum() / (M_t['Conversions'].sum()) 
	
	# Initial State Matrix: we take the conversion rate of the 3 last months
	'''BTNode['Date']=BTNode['Date'].dt.date
	BTNode=BTNode.set_index('Date')
	BTNode=BTNode.sort_index()
	'''
	# M is the measurment or conversion rate for each node 
	M_0=pd.DataFrame(columns=['Clicks','Conversions','CR'])
	#M=pd.DataFrame(columns=['Clicks','Conversions','CR'])
	datebegin=date_ymd(lastdayfrom - pd.Timedelta(days=30))
	
	M_0=M_t.iloc[0]
	'''M_0['Conversions']=BTNode.loc[datebegin:lastdayfrom].reset_index().groupby('Node')['7_Day_Total_Units_(#)'].sum()
	M_0['CR']=(BTNode.loc[datebegin:lastdayfrom].reset_index().groupby('Node')['7_Day_Total_Units_(#)'].sum() / (
		BTNode.loc[datebegin:lastdayfrom].reset_index().groupby('Node')['Clicks'].sum()))
	M_0['CR']= M_t.where(pd.notna(M_t), M_t.iloc[0]['CR'])['CR']
	'''
	print("M_0 :", M_0)
	#n=len(X_t.dropna())
	#print(n)
	n=1
	kf = KalmanFilter(dim_x=1, dim_z=1)


	# Initial Estimation Covariance Matrix/Process Matrix. 
	# P_t= np.identity(n)*0.5
	#P_t=0.5#pd.DataFrame(data=np.identity(n)*0.5,  columns=M_t.columns)
	
	# Measurment covariance matrix
	# R_t= np.identity(n)*5
	#r=BTNode.reset_index()[['Date','Node','CR']].pivot_table(index='Date', columns='Node', values='CR', aggfunc=np.sum).cov().to_numpy()
	r=[[abs(1/M_0['Clicks']*M_0['Conversions']/M_0['Clicks']*(M_0['Clicks']-M_0['Conversions'])/M_0['Clicks'])]] #=σ^2
	#r=r.fillna(10)
	#R_t=pd.DataFrame(data=diagR,index=X_t.dropna().index, columns=X_t.dropna().index)/100
	#R_t=pd.DataFrame(data=np.diag(r),index=X_t.dropna().index, columns=X_t.dropna().index)
	print(kf.R)
	kf.R=r#R_t.to_numpy()


	#print("R_t:", R_t)

	X_pred=pd.DataFrame(index=['Date'],columns=['CR_forecast','CR'])

	kf.H=np.diag(np.diag(np.ones(1)))
	kf.x=M_0['CR']+0.01#X_t.to_numpy()+0.01
	
	kf.P *= 0.02 #2% variance in conversion rate
	print("P :",kf.P)

	stdev=.005

	kf.Q=[[abs(1/M_t['Clicks'].sum()*M_t['Conversions'].sum()/M_t['Clicks'].sum()*(M_t['Clicks'].sum()-M_t['Conversions'].sum())/M_t['Clicks'].sum())]]#block_diag(pd.pivot_table(BTNode, values='CR', index='Date', columns='Node', aggfunc=np.sum).std())

	print("Q :",kf.Q)
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
	kf.F=np.diag(np.ones(1))

	X_plot=[]
	M_plot=[]

	t=0
	#This is as prove of concept to see how well it predicts. We want to compare prediction again historical values.
	for day in M_t.loc[lastdayfrom:pd.to_datetime('21/01/2021')].index.unique():#pd.to_datetime("today")].index.unique(): #for every day
		print(day)#.to_numpy())
		datebegin=date_ymd(day - pd.Timedelta(1))

		print(M_t.loc[datebegin:day])
		print(M_t.loc[datebegin:day]['Clicks'].sum())
		print(M_t.loc[datebegin:day]['Clicks'].sum().item(0))
		M=[M_t.loc[datebegin:day]['Clicks'].sum()]#reset_index().groupby('Node')['Clicks'].sum()
		print(M)
		print(M_t.loc[datebegin:day]['Conversions'].sum())
		M.append(M_t.loc[datebegin:day]['Conversions'].sum())#reset_index().groupby('Node')['7_Day_Total_Units_(#)'].sum()
		print(M)
		M.append(M[1]/M[0])#(M_t.loc[datebegin:day].reset_index().groupby('Node')['7_Day_Total_Units_(#)'].sum() / (
		#M_t.loc[datebegin:day].reset_index().groupby('Node')['Clicks'].sum()))
		print(M)
		# Handling of missing data:
		# https://math.stackexchange.com/questions/982982/kalman-filter-with-missing-measurement-inputs
		# For a missing measurement, just use the last state estimate as a measurement but set the covariance matrix of the measurement to essentially infinity. 
		# (If the system uses inverse covariance just set the values to zero.) This would cause a Kalman filter to essentially ignore the new measurement since 
		# the ratio of the variance of the prediction to the measurement is zero. The result will be a new prediction that maintains velocity/acceleration but 
		# whose variance will grow according to the process noise.

		#M['CR']= M.where(pd.notna(M_t), X_t)['CR']
		#M=M.sort_index(axis=0, ascending=True)
		#M_t['Clicks']=M_t['Clicks'].fillna(0)
		#M_t['Conversions']=M_t['Conversions'].fillna(0)
		print("Measurment: ",M)
		print("Previous states", kf.x)

		#P=kf.P
		kf.predict()

		r=[[abs(1/M[0]*M[1]/M[0]*(M[0]-M[1])/M[0])]]
		#r=r.fillna(10)
		#r= r.where(pd.isna(M_t['Clicks']), 100000) # if no clicks the uncertainty is large
		print(r)
		#R_t=pd.DataFrame(data=np.diag(r),index=X_t.dropna().index, columns=X_t.dropna().index)
		#print("R_t :",R_t)
		kf.R=r#R_t.to_numpy()
		print("x: ",kf.x)
		print("P :", kf.P)
		print("Kalman gain :", kf.K)
		print("residual ",kf.y )

		t+=1
		#kf.update(z)
		print([[M[2]]])#.to_numpy())
		kf.update([[M[2]]])#.to_numpy())
		###kf.R -=mask.to_numpy()
		#s.save()  # save the current state
		X_plot.append(kf.x[0][0])
		M_plot.append(M[2])

	print([*range(len(X_plot))])
	print(X_plot)
	print(M_plot)
	plt.plot([*range(len(X_plot))],X_plot,[*range(len(M_plot))],M_plot)
	plt.show()

	from prophet import Prophet

	dp=M_t['CR'].reset_index()
	dp.columns=['ds','y']
	m =Prophet()
	m.fit(dp)
	future = m.make_future_dataframe(periods=7)
	forecast=m.predict(future)
	fig=m.plot(forecast)
	print(forecast)

def sqrt_func(x):
	# https://github.com/rlabbe/filterpy/issues/62
    try:
        result = scipy.linalg.cholesky(x)
    except scipy.linalg.LinAlgError:
        x = (x + x.T)/2
        result = scipy.linalg.cholesky(x)
    return result

def date_ymd(d):
	return datetime.date(d.year, d.month, d.day)


def get_conversion_rate(df, fromdate, untildate):

	df['Conv_Rate']=df.loc[fromdate:untildate]['7_Day_Total_Units_(#)'].sum().divide(
		df.loc[fromdate:untildate]['Clicks'].sum())	

	#BTNode.loc[day - pd.Timedelta(days=1):day].reset_index().groupby('Targeting')['7_Day_Total_Units_(#)'].sum() / (
	#	BTNode.loc[day - pd.Timedelta(days=1):day].reset_index().groupby('Targeting')['Clicks'].sum())
'''

def fx(x, dt):
	# state transition function - predict next state based
	# to predict conversion rate we assume that only weekly seasonality matters,
	# we calculate by how much the CR increase for a the day of the week and apply this to all nodes.
	# Therefore, we use FFT to predict or ARIMA. Another option would be to pass as input the CR by day + for this specific day of the week
	# https://www.thelearningpoint.net/computer-science/stats-ml-data-time-series-forecasting-arima-fourier-regression-methods
	# https://towardsdatascience.com/analyzing-seasonality-with-fourier-transforms-using-python-scipy-bb46945a23d3
	F = np.array([[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]], dtype=float)
	#t=1
	#F=np.ones((len(x)))#,1)) 
	return x#np.dot(matrix_tree, x)
	
def hx(x):
	# measurement function - convert state into a measurement
	# where measurements are [Node1, Node2....]
	#return np.array([x[0], x[2]])#np.array([[1]])#np.array([x[0], x[2]])
	
	return x#np.array([x[0]])


def selfsampling(BTNode):
	#we want to get the lowest number of day so that we have at least 100 clicks as observation.
	clicks=300
	sampling=1/BTNode*clicks*20
	sampling=sampling.apply(np.ceil)
	return sampling
'''


def get_last_measure(BTNode,lastdayfrom,samplingdays):
	# Get the last measure given the # of day from selfsampling. 
	# This should return the data for each node with similar obsevation number (clicks). 
	lastdayfrom=date_ymd(lastdayfrom)
	df=BTNode.loc[BTNode.index<=lastdayfrom] #get the value until today
	#df=df.set_index('Node')
	df2=pd.DataFrame(columns=['Clicks','7_Day_Total_Units_(#)'])
	for nodedata in df.groupby('Node'):
		node=nodedata[0]
		date_begin=lastdayfrom-datetime.timedelta(samplingdays[[node]].values[0])
		df2=df2.append(nodedata[1].loc[date_begin:].groupby('Node').agg('sum'))
	df2['CR']=df2['7_Day_Total_Units_(#)']/df2['Clicks']
	print("conversion rate sampled: ",df2)
	return df2

def merge():
	# to merge we need to group kw with low volume whitch are in the same leave and the same adrgoup.
	# in doing so we avoid bid adjustment and other settings to interfere not evenly
	#
	print("")

if __name__ == "__main__":
	main()
