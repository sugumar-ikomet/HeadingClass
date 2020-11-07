import os
import codecs
import bs4
import re
import spacy
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from config import *
import nltk
nltk.data.path.append(nltk_dir)
from scispacy.abbreviation import AbbreviationDetector
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
#nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

def style_attr(x):
    try:
        b=x.attrs['style'].split(';')
        out={}
        for i in b:
            if len(i) >0 :
                temp = i.split(":")[0].strip()
                out[temp]=i.split(":")[1]
        return out
    except:
        return None

def extract_num(x):
    if x==None:
        return x
    else:
        numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        if len(rx.findall(x))!=0:
            return float(rx.findall(x)[0])

def extract_class(x):
    if x==None or len(x)==0:
        return None
    else:
        return x[0]

def extract_class(x):
    if x==None or len(x)==0:
        return None
    else:
        return x[0]

def extract_features(pr_id,level,node):
    t = node.text
    tp = node.parent.text if level!=1 else "first_level"
    classp = extract_class(node.get('class'))
    attr  = style_attr(node)
    # attribute extract
    exec("global nh_"+str(level)+";nh_"+str(level)+" = nh_"+str(level-1)+"+node.name+'_'")
    exec("global nh ; nh = nh_"+str(level))
    font_size = extract_num(attr.get('font-size')) if attr is not None else None
    align = attr.get('text-align') if attr is not None else None
    font_fam = attr.get('font-family') if attr is not None else None
    bottom = extract_num(attr.get('margin-bottom')) if attr is not None else None
    left = extract_num(attr.get('margin-left')) if attr is not None else None
    right = extract_num(attr.get('margin-right')) if attr is not None else None
    top = extract_num(attr.get('margin-top')) if attr is not None else None
    indent = extract_num(attr.get('text-indent')) if attr is not None else None
    margin = extract_num(attr.get('margin')) if attr is not None else None
    return [pr_id,level,t,tp,node.name,classp,nh,font_size,align,font_fam,bottom,left,right,top,indent,margin]

def extract_all(child,level,arg_):
    global parent_id
    for i,j in enumerate(child.children):
        if issubclass(type(j),bs4.element.Tag):
            parent_id =i+1 if level==1 else parent_id  
            exec("global feats_{} ;feats_{} = extract_features({},{},j)".format(str(level),str(level),str(parent_id),str(level)))
            #if arg_ =='x':
            exec("global feats_"+str(level)+" ; feats_"+str(level)+" = [a if a is not None else b for a,b in zip(feats_"+str(level)+",feats_"+str(level-1)+")]")
            exec("features_list.append(feats_"+str(level)+")")
            if issubclass(type(j),bs4.element.Tag):
                 extract_all(j,level+1,arg_)


# In[6]:


nlp1 = spacy.load("en_core_web_sm")
abbreviation_pipe1 = AbbreviationDetector(nlp1)
nlp1.add_pipe(abbreviation_pipe1)
nlp2 = spacy.load("en_core_sci_sm")
abbreviation_pipe2 = AbbreviationDetector(nlp2)
nlp2.add_pipe(abbreviation_pipe2)


def final_heading(path,file_id,fln):
    f=open(path,"r")
    soup= BeautifulSoup(f.read(),'html.parser')
    f.close
    adiv=soup.find_all("div", { "class" : 'BodyMatter'})
    child=[]
    mydiv=adiv
    if len(mydiv)==0:
        print("file "+str(fln)+" doesn't have content inside bodymatter")
        return "error file"
    global feats_0,features_list,nh_0
    nh_0=''
    features_list =[]
    feats_0 = [None]*16
    parent_id=0
    extract_all(mydiv[0],1,'x')
    columns=['parent_id','child_level','text','text1','tag','class','nh','font_size','align','font_famliy','margin_bottom','margin_left','margin_right','margin_top','text_indent','marign']
    data_x =pd.DataFrame(features_list,columns=columns)
    if data_x.shape[0]==0:
        print("file "+str(fln)+" doesn't have content inside bodymatter")
        return "error file"    
    data_x['x_ind'] = 1
    data_x['file_id'] = file_id
    df_x = data_x.copy()
    # remove sub, sup, br
    mask = df_x['tag'].isin(['sub', 'sup', 'br'])
    df_x = df_x[~mask]
    df_x['text_raw'] = df_x['text']
    df_x = df_x.dropna(subset=['text'])
    if df_x['parent_id'].shape[0]>0:
        df_x['rel_parent_id'] = (df_x['parent_id']/df_x['parent_id'].max())
    #       df_x['textmatch'] = df_x.apply(lambda x: 1 if (x.text1 in x.text)&(x.child_level!=1) else 0, axis=1)
        try:
            df_x['max_font_size'] = df_x['font_size'].max()
            df_x['min_font_size'] = df_x['font_size'].min()
            df_x['rel_font_size'] = np.where(df_x['font_size']==df_x['font_size'].max(), 1, 0)
        except:
            df_x['max_font_size'] = np.nan
            df_x['min_font_size'] = np.nan
            df_x['rel_font_size'] = np.nan
        df_x['child_level_max']=df_x.groupby(['parent_id'])['child_level'].transform('max')
        df_x['rel_child_level'] = round((df_x['child_level']/df_x['child_level_max']),2)
    def acronym(text):
        doc = nlp1(text)
        acro = doc._.abbreviations
        return acro
    def bio_term(text):
        doc = nlp2(text)
        list_ = [str(i) for i in list(doc.ents)]
        return list_

    def remove_words(text):
        text1 = re.sub(r"[^a-zA-Z]", " ", text)
        tagged_sent = pos_tag(text1.split())
        noun = [word for word,pos in tagged_sent if pos == 'NNP']
        acro = acronym(text)
        boi_ = list(bio_term(text))
        Exemptions = ['pH']
        final_list = list(set().union(noun, acro,Exemptions))
        return final_list

    def remove_number(text):
        clean_txt = re.sub(r"(^[\d.-][)])|(^[\d.-]+\s*)|(^[a-z0-9A-Z][[{.:(+*)])|(^[[{.: (+*)][a-z0-9A-Z][[{.:(+*)])|(^[C|I|L|V|X]{0,6}[.-]+\s*)", "", text)
        clean_txt1 = re.sub(r"[^a-zA-Z]", " ",clean_txt)
        return clean_txt1

    def extra_stopwords(text):
        tag_2 = pos_tag(word_tokenize(text),tagset='universal')
        new_stop = [word.lower() for word,pos in tag_2 if pos in ['ADP','CONJ','DET']]
        stop_words1 = list(stop_words) + new_stop if new_stop is not None else stop_words
        return stop_words1

    def proper_noun(text):
        text1 = re.sub(r"[^a-zA-Z]", " ", text)
        tag_1 = pos_tag(word_tokenize(text1))
        noun = [word for word,pos in tag_1 if pos == 'NNP']
        stop_words1 = extra_stopwords(text1)
        words = [i for i in word_tokenize(text1) if i.lower() not in stop_words1]
        pre_ = len(noun)/len(words) if len(words)>0 else 0
        if (len(noun)==len(words))&(len(words)>0):
            value = 1
        else:
            value = 0
        return [value,noun,pre_]

    def clean_text(text,arg):
        clean_txt1 = remove_number(text)
        stop_words1 = extra_stopwords(clean_txt1)
        tokens = word_tokenize(clean_txt1)
        if arg:
            remove_ = [str(i).lower() for i in remove_words(clean_txt1) if i]
            stop_words2 = stop_words1 + remove_
        else:
            stop_words2 = stop_words1
        words = [i for i in tokens if i.lower() not in stop_words2]
        return tokens, words

    rules = [lambda s: all(x.isupper() for x in s.split()),
            lambda s: all(x.islower() for x in s),
            lambda s: all(x.isdigit() for x in s.split()),
            ]
    def sentence_case(s1):
        b1,words = clean_text(s1,True)
        b2 = remove_number(s1)
        a = [word for word in words if word[0].isupper()]
        div_ = len(words)/len(b1) if len(b1)>0 else 0
        if ((len(a)==1)&(~rules[0](b2))&(len(b1)>1)&(div_>0.5))|(rules[1](words)&(len(words)!=0)):
            value = 1
        else:
            value = 0
        return value
    def title_case(s1):
        _,words = clean_text(s1,True)
        b1 = remove_number(s1)
        if (b1.istitle())|((' '.join(words)).istitle())|(proper_noun(b1)[0]==1)|((len(words)==0)&len(proper_noun(b1)[1])>0.9):
            value = 1
        else:
            value = 0
        return value
    def upper_case(s1):
        tokens,words = clean_text(s1,True)
        a = [word for word in words if word.strip().isupper()]
        b1 = remove_number(s1)
        if (rules[0](b1)&(b1.strip()!=''))|((len(a)==len(tokens))&(len(a)>0)):
            value = 1
        else:
            value = 0
        return value
    def lower_case(s1):
        if rules[1](s1):
            value = 1
        else:
            value = 0
        return value
    df_x['sentence_case'] = df_x['text'].apply(lambda x: sentence_case(x))
    df_x['title_case'] = df_x['text'].apply(lambda x: title_case(x))
    df_x['upper_case'] = df_x['text'].apply(lambda x: upper_case(x))
    df_x['lower_case'] = df_x['text'].apply(lambda x: lower_case(x))
    #no text remain
    df_x['no_text'] = df_x['text'].apply(lambda x: 1 if len(clean_text(x,1)[1])==0 else 0)
    #starts with a number
    df_x['startswthnum']=np.where(df_x['text'].str[0].str.isdigit(),1,0)
    #count no of dots
    df_x["dots"]=df_x['text'].str.split(' ').str[0].str.count('[.]')
    df_x['word_len']=df_x['text_raw'].str.findall(r'\w+').str.len()
    df_x['text'] = df_x['text'].str.strip()
    df_x['text'] = df_x['text'].str.lower().str.replace(r'[^a-zA-Z0-9]','')
    df_x['id']=df_x['text'].apply(lambda x: x[:50])
    df_x['id'] = df_x['id'].str.strip()
    df_x['id'] = df_x['id'].replace('', np.nan)
    df_x = df_x.dropna(subset=['id'])
    return df_x

def x_prep(raw_dir):
    file_list = os.listdir(raw_dir)
    num_dupe_files = len(file_list)-len(set(file_list))
    if num_dupe_files!=0:
        print("Duplicate filenames in the raw directory\nRetrieving duplicate files")
        num_dup=0
        dup_files=[]
        for i in file_list:
            if i not in dup_files:
                num_i=file_list.count(i)-1
                if num_i>0:
                    num_dup+=num_i
                    dup_files.append(i)
                    if num_dup ==num_dupe_files:
                        print("The duplicate filenames are \n",dup_files)
                        exit()
    file_list.sort()
    list_df = pd.DataFrame(file_list,columns=["filename"])
    list_df["file_id"] = np.array(range(1,len(file_list)+1)).astype('str')
    list_df.to_csv(temp_dir+"filename_ids.csv",index=False)
    data_x = pd.DataFrame()
    no_bm_filenames = []
    for row in list_df.itertuples():
        print(row.file_id,row.filename)
        path= str(raw_dir) +row.filename
        file_id =row.file_id.zfill(4)
        a=final_heading(path,file_id,row.filename)
        if isinstance(a,str):
            no_bm_filenames.append(row.filename)
            continue
        data_x = a if data_x.shape[0]==0 else data_x.append(a)
    if len(no_bm_filenames)>0:
        pd.DataFrame(no_bm_filenames,columns=["filename"]).to_csv(temp_dir+"null_bodymatter.csv",index=False)
    return data_x

