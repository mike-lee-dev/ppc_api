import pymongo
from pymongo import MongoClient
import pandas as pd


def main():
    # point the client at mongo URI

    client = pymongo.MongoClient(
        "mongodb+srv://python:Growth12#@cluster0.j2aw7jf.mongodb.net/?retryWrites=true&w=majority")
    # select database
    db = client.test
    # db = client['Cluster0']
    # select the collection within the database
    KW_col = db['keywords']

    KW_report_col = db['keyword_reports']
    # getting documents
    KW_cursor = KW_col.find()

    # Converting the Cursor to Dataframe
    KW_list_cur = list(KW_cursor)
    dfKW = pd.DataFrame(KW_list_cur)
    print(dfKW)
    # getting documents
    KW_report_cursor = KW_report_col.find()

    # Converting the Cursor to Dataframe
    KW_report_list_cur = list(KW_report_cursor)
    dfKW_report = pd.DataFrame(KW_report_list_cur)
    print(dfKW_report)


# convert entire collection to Pandas dataframe
# df = pd.DataFrame(list(collection.find()))
# for x in collection.find():
#	print(x)
if __name__ == "__main__":
    main()