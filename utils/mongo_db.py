import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd


def db_connect():
	#point the client at mongo URI
	client = pymongo.MongoClient("mongodb+srv://python:Growth12#@cluster0.j2aw7jf.mongodb.net/?retryWrites=true&w=majority")
	#select database
	db = client.test
	#db = client['Cluster0']
	return db


def read_collection_as_df(collection, db):
	#select the collection within the database
	col=db[collection]	
	#getting documents
	cursor = col.find()	

	#Converting the Cursor to Dataframe
	list_cur = list(cursor)
	df = pd.DataFrame(list_cur)
	df=df.convert_dtypes()
	return df

def read_collection_account_as_df(collection, account, db):
	#select the collection within the database
	col=db[collection]	
	#getting documents
	cursor = col.find({"account": ObjectId(account)})
	# cursor = col.find({})
	#Converting the Cursor to Dataframe
	list_cur = list(cursor)
	df = pd.DataFrame(list_cur)
	df=df.convert_dtypes()
	return df

def write_df(collection, db):
	db.collection.insert_many(df.to_dict('records'))


def write_df_into_collection(df, collection, db):
	col = db[collection]
	col.insert_many(df.to_dict('records'))

if __name__ == "__main__":
	main()
