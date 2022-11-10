import sys
import array
import pandas as pd
import datetime
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk import word_tokenize
from nltk.wsd import lesk
from ortools.linear_solver import pywraplp

def main():

	df=pd.read_csv('C:/Users/jtran_000/Google Drive/growth scalers/Sonoma linen/Sponsored Products Search term report.csv',header=0,)	#  index_col=0)	
	# We want to get for each keyword a entity detection of each word. For instance for the word blue, we want to get the entity color.
	# Once this is done, we should create a dictionary of synonyms and in a next steps of translation. 
	# https://sunscrapers.com/blog/8-best-python-natural-language-processing-nlp-libraries/
	# https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
	# spacy doesn't identify colors
	

	# https://en.wikipedia.org/wiki/Ontology_learning
	# https://en.wikipedia.org/wiki/Automatic_taxonomy_construction
	# http://www.nltk.org/howto/wordnet.html
	# https://github.com/wordnet/wordnet
	# https://www.linkedin.com/pulse/wordnet-word-sense-disambiguation-wsd-nltk-aswathi-nambiar/
	# https://stackoverflow.com/questions/17671081/how-to-find-semantic-relationship-between-two-synsets-in-wordnet
	# https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
	# -> https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
	# -> https://opensource.com/article/20/8/nlp-python-nltk


	# word sense disambiguation
	# This doesn't work by using the search query. We should try with the info on website/product description/ads
	# https://towardsdatascience.com/a-simple-word-sense-disambiguation-application-3ca645c56357
	# https://github.com/alvations/pywsd
	# http://www.nltk.org/howto/wsd.html
	# https://stackoverflow.com/questions/20896278/word-sense-disambiguation-algorithm-in-python
	# https://stackoverflow.com/questions/3699810/word-sense-disambiguation-in-nltk-python
	# https://paperswithcode.com/task/word-sense-disambiguation


	# 1) POS (Part of Speach) tagging : is it a noun/verb/adjectif
	# 1) find out which word share the same hypernyms and consider this synset as the right meaning
	# 2) get the right meaning of each word
	# 3) get if they are synonym
	# 4) get variation, for instance colors: blue, white, grey..
	# optional get pair of words like "navy blue"	

	#https://avidml.wordpress.com/2018/12/02/semantic-similarity-approach-understand-build-and-evaluate/
	#https://medium.com/@pragadesh/semantic-similarity-using-wordnet-ontology-b12219943f23


	#POS Tagging
	#text_tagged = pos_tag(word_tokenize(text_data))

	
	df=df[['Campaign Name','Ad Group Name','Targeting','Customer Search Term','Spend']]
	df=df.drop_duplicates()
	print(df)

	solver = pywraplp.Solver.CreateSolver('GLOP')
	X1=[]
	X2=[]

	hypo = lambda s: s.hyponyms()
	hyper = lambda s: s.hypernyms()
	#wn_list=[]
	# 1) understand words of campaign
	vectorizer = CountVectorizer()
	vectorizer.fit(pd.concat([df['Campaign Name'],df['Ad Group Name']]))
	words=vectorizer.get_feature_names()
	print(words)
	total_similarity=0
	for i, word1 in enumerate(words):
		wn_word1 = wn.synsets(word1)
		print("\n",word1)
		for wns in wn_word1:
			print(wns.hypernyms())
		if len(wn_word1)==0:
			print("this word is not in the dictionary : ", word1)
			words.remove(word1)
		else:
			#print(wn_word1)
			#wn_list.append(wn_word1)
			X1.append(solver.IntVar(0.0, len(wn_word1), word1)) #As input we want to change syset index to maximize similarity
			for word2 in words[i+1:]:
				wn_word2 = wn.synsets(word2)
				if len(wn_word2)==0:
					print("this word is not in the dictionary : ", word2)
					words.remove(word2)
				else:	
					#print(wn_word2)
					X2.append(solver.IntVar(0.0, len(wn_word2), word2))

	#print("X :", X)
	#print ("wn_list", wn_list)
	#for comb in itertools.combinations(wn_list,2):

	#	print("comb : ", comb)
		
	solver.Maximize(sum(wn_word1[X1].path_similarity(wn_word2[X2]))) 


	###I should try to use the code below to get the right sysnet
	###I should also check BERT https://arxiv.org/abs/1810.04805
	def SimScore(synsets1, synsets2):
		"""
		Purpose: Computes sentence similarity using Wordnet path_similarity().
		Input: Synset lists representing sentence 1 and sentence 2.
		Output: Similarity score as a float
		"""

		print("-----")
		print("Synsets1: %s\n" % synsets1)
		print("Synsets2: %s\n" % synsets2)

		sumSimilarityscores = 0
		scoreCount = 0

		# For each synset in the first sentence...
		for synset1 in synsets1:

			synsetScore = 0
			similarityScores = []

			# For each synset in the second sentence...
			for synset2 in synsets2:

				# Only compare synsets with the same POS tag. Word to word knowledge
				# measures cannot be applied across different POS tags.
				if synset1.pos() == synset2.pos():

					# Note below is the call to path_similarity mentioned above.
					synsetScore = synset1.path_similarity(synset2)

					if synsetScore != None:
						print("Path Score %0.2f: %s vs. %s" % (synsetScore, synset1, synset2))
						similarityScores.append(synsetScore)

					# If there are no similarity results but the SAME WORD is being
					# compared then it gives a max score of 1.
					elif synset1.name().split(".")[0] == synset2.name().split(".")[0]:
						synsetScore = 1
						print("Path MAX-Score %0.2f: %s vs. %s" % (synsetScore, synset1, synset2))
						similarityScores.append(synsetScore)

					synsetScore = 0

			if(len(similarityScores) > 0):
				sumSimilarityscores += max(similarityScores)
				scoreCount += 1

		# Average the summed, maximum similarity scored and return.
		if scoreCount > 0:
			avgScores = sumSimilarityscores / scoreCount

		print("Func Score: %0.2f" % avgScores)
		return(avgScores)
		'''
		solver = pywraplp.Solver.CreateSolver('GLOP')
	
		infinity = solver.infinity()
		# x and y are integer non-negative variables.
		x = solver.IntVar(0.0, infinity, 'x')
		y = solver.IntVar(0.0, infinity, 'y')
	
		solver.Maximize(simlarity)
		solver.Solve()
	
		print('Number of variables =', solver.NumVariables())
		'''



		'''
		df= pd.DataFrame()
		for i, word1 in enumerate(words):
			print("\n",word1)
			wn_word1 = wn.synsets(word1)
			if len(wn_word1)==0:
				print("this word is not in the dictionary : ", word1)
				words.remove(word1)
			else:			
				print(wn_word1[0].lemma_names())
				for word2 in words[i+1:]:
					wn_word2=wn.synsets(word2)
					if len(wn_word2)==0:
						print("this word is not in the dictionary : ", word2)
						words.remove(word2)
					else:				
						wn_word2 = wn.synsets(word2)
	
						common_hyponyms= set(wn_word1.closure(hypo)).intersection(set(wn_word2.closure(hypo)))
						print(common_hyponyms)
						print(word1 + " " + word2 + "simlarity :",wn_word1[0].path_similarity(wn_word2[0]))
						#common_lemmas = len(set(wn_word1.lemma_names).intersection(set(wn_word2.lemma_names)))
						#print(common_lemmas)'''
		'''
				wn_word = wn.synsets(word)
				if len(wn_word)==0:
					print("this word is not in the dictionary : ", word)
				else:	
					print("Synset :", wn_word)
					print("Hypernyms :", wn_word[0].hypernyms())
					print("Member holonyms :", wn_word[0].member_holonyms())
					common_lemmas = len(set(house.lemma_names).intersection(set(station.lemma_names)))
		'''

if __name__ == "__main__":
	main()
