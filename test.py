import json
from flask import Flask
import ppcoptimizer

app = Flask(__name__)


@app.route('/')
def index():
    # ppcoptimizer.main()
    return json.dumps({'name': 'alice',
                       'email': 'alice@outlook.com'})


app.run()
