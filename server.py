import json
from flask import Flask
import ppcoptimizer

app = Flask(__name__)


@app.route('/')
def index():
    updated_info =  ppcoptimizer.main()
    return json.dumps(updated_info)


app.run()
