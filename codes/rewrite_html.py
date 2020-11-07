import re
from bs4 import BeautifulSoup
from random import sample
import bs4
import re
from config import *
import pandas as pd
import numpy as np


def html_create(data,raw_dir=raw_dir,rewrite_dir=rewrite_dir):
        p1 = data.copy()
        fileids = data[["file_id"]].astype('int')
        id_df = pd.read_csv(temp_dir+"filename_ids.csv")
        id_df["file_id"] = id_df["file_id"].astype('int')
        file_df = fileids.merge(id_df,on=["file_id"],how="inner")
        assert(fileids.shape[0]==file_df.shape[0])
        #file1 = list(set(data['file_id']))
        for row in file_df.itertuples():
            path = str(raw_dir) +row.filename
            f = open(path,"r")
            soup = BeautifulSoup(f.read(),'html.parser')
            f.close()
            adiv = soup.find("div", { "class" : 'BodyMatter'})
            p2 = p1[p1['file_id']==row.file_id]
            x1 = p2[p2.new_pred16.str.contains('Heading')]
            list_ = list(x1.parent_id.values)
            text_ = list(x1.text_raw.values)
            n_text = list(x1.ntext.values)
            head_ = list(x1.new_pred16.values)
            prob_ = list(x1.model_prob.values)
            rule_ind = list(x1.ruleind2.values)
            word_len = list(x1.word_len.values)
            # print(list_)
            level = 1
            parent_id = 0
            for i,j in enumerate(adiv.children):
                if issubclass(type(j),bs4.element.Tag):
                    parent_id = i+1 if level == 1 else parent_id
                    if parent_id in list_:
                        if (n_text[list_.index(parent_id)] == 1)|((n_text[list_.index(parent_id)]<10)&(word_len[list_.index(parent_id)]<15)):
                            j['class'] = head_[list_.index(parent_id)]
                            j['head_prob'] = round(prob_[list_.index(parent_id)],3)
                            j['rule_ind'] = rule_ind[list_.index(parent_id)]
                            # print([parent_id,level,j.name,j.text])
                        else:
                            k = [p for _,p in enumerate(j.children) if issubclass(type(p),bs4.element.Tag)]
                            # print(parent_id,k[0])
                            k = k[0]
                            k['class'] = head_[list_.index(parent_id)]
                            k['head_prob'] = round(prob_[list_.index(parent_id)],3)
                            k['rule_ind'] = rule_ind[list_.index(parent_id)]
                            # print([parent_id,level,k.name,k.text])
            file_out = str(rewrite_dir)+row.filename
            with open(file_out, "w",encoding="utf-8") as file:
                file.write(str(soup))


