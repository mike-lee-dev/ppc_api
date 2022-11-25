import math
import pandas as pd
import numpy as np
import sys
from utils import dataframe
from numpy.linalg import inv
from numpy import fft
from treelib import Node, Tree as btree
import nltk
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import cholesky, block_diag
from sklearn import tree
from sklearn.tree import export_graphviz
import scipy
import pydot

def clustering(df):
	# df is the orignial dataframe
	# RFpreproc is the dataframe with feature
	# RF_filter is RFpreproc resampled weekly, aggregated by feature and filtered if feature has low observation
	RF_preproc=cleansing(df)
	RF_preproc=resample(RF_preproc)
	#RF_preproc, leportofolio, lecampaign, leadgroup, letargeting, lematchtype=encoding(RF_preproc)
	RF_preproc,lecampaign, leadgroup, letargeting, lematchtype=encoding(RF_preproc)

	RF_filter=feature_selection(RF_preproc)
	regr=random_forest(RF_filter) #group node with similar conversion rate
	data_tree=read_binear_tree(regr, RF_filter)
	RF_preproc, bintree=binear_tree(data_tree, RF_preproc)
	#RF_decoding=decoding_node(RF_preproc, leportofolio, lecampaign, leadgroup, letargeting, lematchtype)
	RF_decoding=decoding_node(RF_preproc, lecampaign, leadgroup, letargeting, lematchtype)

	dfclustered=aggregate_by_node(RF_decoding, df)
	# dfclustered.to_csv(path + "/prediction/merge_test.csv")

	dfclustered, RF_decoding=kill_small_leave(RF_decoding, df, bintree)
	# dfclustered.to_csv(path + "/prediction/kill_small_leave.csv")
	return dfclustered, RF_decoding


def cleansing(df):
		
	# We want to cluster data with Random Forest (segregate) good and bad conversion rate) and then forecast CR
	RF_preproc=df.copy(deep=True)
	RF_preproc.columns = [c.replace(' ', '_') for c in RF_preproc.columns]
	return RF_preproc

def resample(RF_preproc):
	RF_preproc=RF_preproc.set_index('date')
	RF_preproc=RF_preproc.groupby([pd.Grouper(freq='W'),'campaignName','adGroupName','targeting','matchType']).sum().reset_index()
	return RF_preproc


def encoding(RF_preproc):
	# todo add Portfolio_name as feature
	# todo improve matchType: 1 exact, 2 phrase, 3 broad match modifier, 4 broad match

	# the more word a keyword has, the higher the CR
	RF_preproc['Number_of_word']=RF_preproc['targeting'].str.count(' ')+1

	# encoding
	leportofolio=LabelEncoder()
	lecampaign=LabelEncoder()
	leadgroup=LabelEncoder()
	letargeting=LabelEncoder()
	lematchtype=LabelEncoder()

	#RF_preproc['Portfolio_name_enc']=leportofolio.fit_transform(RF_preproc['Portfolio_name'])
	RF_preproc['campaignName_enc']=lecampaign.fit_transform(RF_preproc['campaignName'])
	RF_preproc['adGroupName_enc']=leadgroup.fit_transform(RF_preproc['adGroupName'])
	RF_preproc['targeting_enc']=letargeting.fit_transform(RF_preproc['targeting'])# -> This can lead to overfitting
	RF_preproc['matchType_enc']=lematchtype.fit_transform(RF_preproc['matchType'])

	#split word by word. This might be useful if one word is better than another
	vectorizer = CountVectorizer()# --> Actually we don't need count because the number of occurence is irrelevant
	vectorizer.fit(RF_preproc['targeting'])
	#print(vectorizer.get_feature_names())
	word_vect = vectorizer.transform(RF_preproc['targeting'])

	# Todo check if it removes symbols like []+
	#word=pd.DataFrame(data=word_vect.toarray(), index=RF_preproc.index, columns=vectorizer.get_feature_names())
	#word=pd.DataFrame(data=word_vect.toarray(), index=RF_preproc.index, columns=vectorizer.get_feature_names_out)
	#RF_preproc=RF_preproc.join(word)
	
	#split by pair of words to get the order.
	#vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
	#vectorizer2.fit(df['targeting'])
	#word_vect2 = vectorizer2.transform(df['targeting'])

	#word2=pd.DataFrame(data=word_vect2.toarray(), index=RF_preproc.index, columns=vectorizer2.get_feature_names())
	#word2.columns=word2.columns.str.replace(' ','_')	 		# This is needed to eval the filter later.
	#RF_preproc=RF_preproc.join(word2)

	#return RF_preproc, leportofolio, lecampaign, leadgroup, letargeting, lematchtype
	return RF_preproc, lecampaign, leadgroup, letargeting, lematchtype

	
def feature_selection(RF_preproc):
	RF_filter=RF_preproc.copy()
	#to_keep = ['Portfolio_name_enc','campaignName_enc','adGroupName_enc','targeting_enc','matchType_enc', 'Number_of_word', 'clicks','conversions']
	to_keep = ['campaignName_enc','adGroupName_enc','targeting_enc','matchType_enc', 'Number_of_word', 'clicks','conversions']
	RF_filter=RF_filter[to_keep]
	return(RF_filter)


def random_forest(RF_filter):
	RF_filter['CR_7days']=(RF_filter['conversions'])/(RF_filter['clicks']+1)*100
	CR_filter=RF_filter['CR_7days'].values
	RF_filter.drop(['clicks','CR_7days','conversions'], axis = 1, inplace=True)
	regr = RandomForestRegressor(min_samples_leaf=5)#min_impurity_decrease= 0.01)#0.01, min_samples_split=2)

	regr.fit(RF_filter, CR_filter)
	feat_importances = pd.Series(regr.feature_importances_, index=RF_filter.columns)
	print("Random Forest feature importance \n: ")
	print(feat_importances.sort_values(ascending=False))
	return regr

def read_binear_tree(regr, RF_filter):
	# This code below is not elegant at all but I couldn't find anything better

	# Get one tree from the forest
	tree = regr.estimators_[5]
	# Export the image to a dot file
	export_graphviz(tree, out_file = 'tree.dot', feature_names =  RF_filter.columns, rounded = True, precision = 1)
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')
	# tag each keyword to the belonging leave
	f = open('tree.dot', 'r')
	data_tree = f.read()
	#out = StringIO()
	print(data_tree)
	f.close()
	return data_tree

def binear_tree(data_tree,RF_preproc): 

	# read the tree in a more user-friendly way and save rules in dataframe
	dfrules=pd.DataFrame(columns=['Node','Rule'])
	
	#root=Node(0)
	parent=0
	child=0
	bintree=btree()
	bintree.create_node(0,0)
	bintree.show()
	for line in data_tree.splitlines():
		#btree=line[0:line.find(' ')]
		if line.find('headlabel=\"True\"')>0: #first split
			parent=int(line[0:1])
			child=int(line[5:6])
			if bintree.get_node(parent).is_leaf():
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

	leaves= [leave.identifier for leave in bintree.leaves()]
	print("leaves :",leaves)

	# tag data with the leave number and save matrix of hierarchy
	#depth=bintree.depth()
	all_nodes=len(bintree.all_nodes())
	#inheritance_ratio=0.4
	#global matrix_tree
	#matrix_tree=np.zeros((all_nodes,all_nodes))
	#matrix_tree[0,0]=1
	Node_level_max="Node_"+str(bintree.depth())
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(linewidth=3200)
	#matrix_tree[0,0]=1
	RF_preproc=RF_preproc.assign(Node_level_0=0) #node 0 is all data	
	for path in bintree.paths_to_leaves():
		filters=""
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
			if node in leaves:
				RF_preproc.loc[RF_preproc.eval(filters),'Leave']=node
			filters = filters + " & "
			lvl=lvl+1
	RF_preproc['Leave']=RF_preproc['Leave'].astype(int)
	return RF_preproc, bintree


#def decoding_node(RF_preproc, leportofolio, lecampaign, leadgroup, letargeting, lematchtype):
def decoding_node(RF_preproc, lecampaign, leadgroup, letargeting, lematchtype):
	#RF_preproc=RF_preproc.groupby('Leave').first()
	RF_decoding=RF_preproc.copy()
	#RF_decoding['Portfolio_name']=leportofolio.inverse_transform(RF_decoding['Portfolio_name_enc'])
	RF_decoding['campaignName']=lecampaign.inverse_transform(RF_decoding['campaignName_enc'])
	RF_decoding['adGroupName']=leadgroup.inverse_transform(RF_decoding['adGroupName_enc'])
	RF_decoding['targeting']=letargeting.inverse_transform(RF_decoding['targeting_enc'])
	RF_decoding['matchType']=lematchtype.inverse_transform(RF_decoding['matchType_enc'])
	#RF_decoding=RF_decoding[['Leave','Portfolio_name', 'campaignName', 'adGroupName', 'targeting', 'matchType']]
	RF_decoding=RF_decoding[['Leave','campaignName', 'adGroupName', 'targeting', 'matchType']]
	RF_decoding=RF_decoding.drop_duplicates(subset=['campaignName', 'adGroupName', 'targeting', 'matchType'])
	print(f"decoding: \n{RF_decoding}")
	return RF_decoding

def aggregate_by_node(RF_decoding, df):
	# now we need to have the clicks and the conversion rate for each node of the tree.
	RF_decoding_copy=RF_decoding.copy(deep=True)
	RF_decoding_copy.columns=[c.replace('_', ' ') for c in RF_decoding_copy.columns] ####This would only work if the name has no_
	dfclustered=df.merge(RF_decoding_copy, 
		on=['campaignName','adGroupName','targeting','matchType'], 
		#right_on=['campaignName', 'adGroupName','targeting','matchType'],
		how='left').drop_duplicates(subset=['date','campaignName','adGroupName','targeting','matchType'],ignore_index=True)
	dfclustered=dfclustered.groupby(by=['date','Leave'], as_index=False).agg({'clicks':'sum','conversions':'sum','sales':'sum'}).set_index('date')
	dfclustered.sort_values(by=['date'])
	return dfclustered

def kill_small_leave(RF_decoding, df, bintree):
	dfcopy=df.copy()
	dfcopy=aggregate_by_node(RF_decoding, df) #to do if we save history: only use last 90 days
	dfcopy=dfcopy.groupby('Leave', as_index=False).agg({'clicks':'sum','conversions':'sum','sales':'sum'})
	l=len(dfcopy)
	i=0
	dfcopy=dfcopy[dfcopy['clicks']<=20]
	dfcopy.sort_values(by='clicks', ascending=True, inplace=True)
	while not dfcopy.empty:
		Leave=dfcopy['Leave'].iloc[0]
		parent=bintree.parent(Leave).identifier
		if parent ==0:
			RF_decoding['Leave'].loc[RF_decoding['Leave']==Leave]=dfcopy['Leave'].iloc[1]
			dfcopy['Leave'].loc[dfcopy['Leave']==Leave]=dfcopy['Leave'].iloc[1]
		else:
			children=bintree.children(parent)			
			RF_decoding['Leave'].loc[RF_decoding['Leave']==children[0].identifier]=parent
			RF_decoding['Leave'].loc[RF_decoding['Leave']==children[1].identifier]=parent
			dfcopy['Leave'].loc[dfcopy['Leave']==children[0].identifier]=parent
			dfcopy['Leave'].loc[dfcopy['Leave']==children[1].identifier]=parent
		dfcopy=aggregate_by_node(RF_decoding, df)
		dfcopy=dfcopy.groupby('Leave', as_index=False).agg({'clicks':'sum','conversions':'sum','sales':'sum'})
		dfcopy=dfcopy[dfcopy['clicks']<=20]
		dfcopy.sort_values(by='clicks', ascending=True, inplace=True)
		i=i+1
		if i>l:
			break
	dfclustered=aggregate_by_node(RF_decoding, df)
	return dfclustered, RF_decoding
