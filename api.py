from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from pymongo import MongoClient
import os
import ppcoptimize.ppcoptimizer as ppcoptimizer

app = Flask(__name__)
api = Api(app)

# Connect mongo DB
db_link = os.getenv("DATABASE")
db_password = os.getenv("DATABASE_PASSWORD")
# mongo = MongoClient(db_link.replace("<PASSWORD>", db_password))
mongo = MongoClient(db_link, PASSWORD=db_password)
db = mongo['test']

# Models
accounts = db['accounts']


@app.route('/test', methods=['GET', 'POST'])
def index():
    # call function from ppcoptimizer and store it into Mongo DB
    # then return the results to the endpoint of node js or frontend
    save_or_response = ppcoptimizer.main()
    pass


if __name__ == '__main__':
    app.run(debug=True)
