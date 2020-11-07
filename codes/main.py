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
from bs4 import BeautifulSoup

#raw_dir = "D:/ML_Cobe/ikomet/scoring/raw_html/JTD_TestData6_HTMLCleanup.html"
#HtmlFile = open(raw_dir, 'r', encoding='utf-8')
#source_code = HtmlFile.read()

from flask import request
from flask import Flask, render_template, redirect, url_for, make_response
from flask import Flask
app = Flask(__name__)

@app.route('/heading/detection', methods=['POST'])

def h1_h6_heading_detection():

    req = request.get_json()
    #print(req)
    #print(req['rawfilepath'])
    #raw_dir = req['rawfilepath']
    #print(req['rewrite'])
    #rewrite_dir = req['rewrite']

    df1 = x_prep(raw_dir)
    df1.to_csv(temp_dir+'temp1.csv',index=False)
    df1 = pd.read_csv(temp_dir+"temp1.csv")
    df2 = feature_create(df1)
    df2.to_csv(temp_dir+'temp2.csv',index=False)
    df2 = pd.read_csv(temp_dir+"temp2.csv")
    df3 = lag_lead_feature(df2)
    df3.to_csv(temp_dir+'temp3.csv',index=False)
    df3 = pd.read_csv(temp_dir+"temp3.csv")
    df4 = model_classification(df3)
    df4.to_csv(temp_dir+'temp4.csv',index=False)
    df4 = pd.read_csv(temp_dir+"temp4.csv")
    df5 = rule_final(df4)
    df5.to_csv(temp_dir+"final_output.csv",index=False)
    html_create(df5)
    #print(df5)
    #soup = BeautifulSoup(df5, 'lxml')
    #return soup

    '''with open('D:\\ML_Cobe\\ikomet\\scoring\\structured_html\\JTD.html', 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')

    print("Soup: ",soup)
    return soup'''


if __name__ == '__main__':
    app.run()