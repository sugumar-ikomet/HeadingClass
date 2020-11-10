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
from datetime import datetime
import paramiko
import base64
#raw_dir = "D:/ML_Cobe/ikomet/scoring/raw_html/JTD_TestData6_HTMLCleanup.html"
#HtmlFile = open(raw_dir, 'r', encoding='utf-8')
#source_code = HtmlFile.read()

from flask import request
from flask import Flask, render_template, redirect, url_for, jsonify, make_response
from flask import Flask
app = Flask(__name__)

@app.route('/heading/detection', methods=['POST'])

def h1_h6_heading_detection():
    try:
        dt = datetime.now()
        current_time = dt.microsecond
        print('original request',request);
        req = request.get_json();
        
        print('after jsonify',req);
        # return make_response(jsonify({"error":None}),200)

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(remote_host, remote_port, remote_user, remote_pass)
        sftp_client = ssh_client.open_sftp()
        remote_file_path = os.path.join(remote_path,req['projectName'],req['workObjectID'],'HTML',req['fileName'])
        print('remote_file_path:',remote_file_path)
        remote_file = sftp_client.open(remote_file_path)
        file_name = str(current_time)+'---separator---'+req['fileName'];

        print(req);
        f = open(raw_dir+'/'+file_name, "wb")
        f.write(remote_file.read())
        f.close()
        remote_file.close()

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
        sftp_client.put(rewrite_dir+file_name,remote_file_path);
        print('success');
        os.remove(rewrite_dir+file_name);
        os.remove(raw_dir+file_name);
        return make_response(jsonify({"error":None}),200)
    except Exception as e:
        print(e);
        return make_response(jsonify({"error":e}),500)
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