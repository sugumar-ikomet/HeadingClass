import os
import json
import pandas as pd
import numpy as np
from config import *
from x_prep import *
from feature_create import *
from classification import *
from rules_final import *
from rewrite_html import *

import os
import urllib.request
raw_dir = "D:/ML_Cobe/ikomet/scoring/raw_html/JTD_TestData6_HTMLCleanup.html"
HtmlFile = open(raw_dir, 'r', encoding='utf-8')
source_code = HtmlFile.read()

#print(source_code)
#returnbuf = Flask API call API call POST buffer http://localhost:3800/converth1h6/ POST buffer
#filesave(returnbuf, c:\test\JTD_0001_out.html)



import json
'''
from flask import Flask
from flask import jsonify, request


# Create the application instance
app = Flask(__name__)

# Create a URL route in our application for "/"
@app.route('/')
def home():
    return jsonify({"about" : "Hello World!"})

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)

'''

from flask import Flask, request, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class h1_h6(Resource):
    def get(self):
        return source_code

    def post(self):
        df1 = x_prep(source_code)
        df1.to_csv(temp_dir + 'temp1.csv', index=False)
        df1 = pd.read_csv(temp_dir + "temp1.csv")
        df2 = feature_create(df1)
        df2.to_csv(temp_dir + 'temp2.csv', index=False)
        df2 = pd.read_csv(temp_dir + "temp2.csv")
        df3 = lag_lead_feature(df2)
        df3.to_csv(temp_dir + 'temp3.csv', index=False)
        df3 = pd.read_csv(temp_dir + "temp3.csv")
        df4 = model_classification(df3)
        df4.to_csv(temp_dir + 'temp4.csv', index=False)
        df4 = pd.read_csv(temp_dir + "temp4.csv")
        df5 = rule_final(df4)
        df5.to_csv(temp_dir + "final_output.csv", index=False)

        return jsonify(df5)

class Multi(Resource):
    def get(self,num):
        return {"result" : num*10}

api.add_resource(h1_h6,'/')
api.add_resource(Multi,'/multi/<int:num>')

if __name__ == '__main__':
    app.run(debug=True)




