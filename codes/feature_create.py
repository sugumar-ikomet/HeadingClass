import re
import pandas as pd
import numpy as np
import nltk
import joblib
from config import *
if nltk_dir not in nltk.data.path:
    nltk.data.path.append(nltk_dir)
from clean_text import *
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import OneHotEncoder
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


def feature_create(data_x, ohe_tag_dir=ohe_tag_dir): 
    x = data_x.copy()
    x['h1'] = x['nh'].apply(lambda x: 1 if 'h1' in x.split('_') else 0 )
    x['h2'] = x['nh'].apply(lambda x: 1 if 'h2' in x.split('_') else 0 )
    x['h3'] = x['nh'].apply(lambda x: 1 if 'h3' in x.split('_') else 0 )
    x['h4'] = x['nh'].apply(lambda x: 1 if 'h4' in x.split('_') else 0 )
    x['h5'] = x['nh'].apply(lambda x: 1 if 'h5' in x.split('_') else 0 )
    x['h6'] = x['nh'].apply(lambda x: 1 if 'h6' in x.split('_') else 0 )
    x['b_strong'] = x['nh'].apply(lambda x: 1 if 'b' in x.split('_') or 'strong' in x.split('_') else 0 ) 
    x['em_i'] = x['nh'].apply(lambda x: 1 if 'i' in x.split('_') or 'em' in x.split('_') else 0 ) 
    x['u'] = x['nh'].apply(lambda x: 1 if 'u' in x.split('_') else 0 ) 
    x['index']=x.index
    #one hot encoding
    enc = joblib.load(ohe_tag_dir)
    x=x.fillna({'class':'missing','tag':'missing'})
    xenc=enc.transform(x['tag'].values.reshape(-1,1))
    df_enc=pd.DataFrame(xenc,columns=list(enc.get_feature_names()))
    print(xenc.shape,df_enc.shape)
    df_x=pd.concat([x,df_enc],axis=1)
    gp = {i:["max","min"] for i in list(enc.get_feature_names())}
    gp['word_len'] = ['max','min']
    gp['font_size'] = ['max','min']
    gp['sentence_case'] = ['max','min']
    gp['title_case'] = ['max','min']
    gp['margin_bottom'] = ['max','min']
    gp['margin_left'] = ['max','min']
    gp['margin_right'] = ['max','min']
    gp['margin_top'] = ['max','min']
    gp['text_indent'] = ['max','min']
    gp['marign'] = ['max','min']
    gp['h1'] = ['max','min']
    gp['h2'] = ['max','min']
    gp['h3'] = ['max','min']
    gp['h4'] = ['max','min']
    gp['h5'] = ['max','min']
    gp['h6'] = ['max','min']
    gp['b_strong'] = ['max','min']
    gp['em_i'] = ['max','min']
    gp['u'] = ['max','min']
    x1=df_x.groupby(['file_id','parent_id']).aggregate(gp).reset_index()
    x1.columns = ['_'.join(col) for col in x1.columns.values]
    x1.reset_index(inplace=True)
    x1.drop(columns=['index'],inplace=True)
    x1.rename(columns={'file_id_':'file_id','parent_id_':'parent_id'}, inplace=True)
    # parent level data
    x2_1 = x[x['child_level']==1][['file_id','parent_id','text','text_raw','tag','word_len',
                                 'class','sentence_case','title_case','upper_case','lower_case',
                                 'startswthnum','text_indent','align','max_font_size','min_font_size','no_text']].reset_index()
    print(x2_1.shape)
    # level in order of parent id. DO NOT SHUFFLE
    x2_2=x.groupby(['file_id','parent_id']).cumcount().reset_index()
    x2_2.rename(columns={0:'level'},inplace=True)
    print(x2_2.shape)
    #merge overwritten x only
    x=x.merge(x2_2,on="index",how="inner")
    print(x.shape)
    # important to check if all file_id, parent_id has second level
    df_c=x.groupby(['file_id','parent_id']).agg({'level':'max'})
    df_c[df_c['level']==0].shape
    x2_3=x[x['level']==1][['file_id','parent_id','text_raw','tag','class','align','word_len','font_size','text_indent','marign']]
    x2_3['slevel']=1
    x2_3.rename(columns={'text_raw':'s_text_raw','tag':'s_tag',
                         'class':'s_class','align':'s_align','word_len':'s_word_len',
                         'font_size':'s_font_size',
                          'text_indent':'s_text_indent',
                           'marign':'s_margin'},inplace=True)
    print(x2_3.shape)
    x2 = x2_1.merge(x2_3,on=['file_id','parent_id'],how='left')
    x2=x2.fillna({'s_text_raw':'smissing','s_tag':'missing','s_class':'missing','s_align':'missing','s_word_len':0,
              's_font_size':0,'s_text_indent':0,'s_margin':0,'slevel':0})
    x3=x.groupby(['file_id','parent_id']).text.nunique().reset_index()
    x3.rename(columns={'text':'ntext'}, inplace=True)
    x_m1=x1.merge(x2,on=['file_id','parent_id'])
    x_m2=x3.merge(x_m1,on=['file_id','parent_id'])
    # feature create        
    x_m2['run_text'] = x_m2['text_raw'].apply(lambda x: re.sub(r"(^[\d.-][)])|(^[\d.-]+\s*)|(^[a-z0-9A-Z][[{.:(+*)])|(^[[{.: (+*)][a-z0-9A-Z][[{.:(+*)])|(^[a-z0-9A-Z]{0,2}[.-]+\s*)|(^[I|L|V|X]{1,6}[.])", "", x[:450]))
    x_m2[['run_text', 'no_run_sent']] = x_m2['run_text'].apply(lambda x: pd.Series(sent_ext(x[:450])))
    x_m2['run_word_len'] = x_m2['run_text'].str.findall(r'\w+').str.len()
    x_m2['caps_colon'] = x_m2['text_raw'].apply(lambda x: caps_after_colon(x[:450]))
    x_m2['contain_colon'] = x_m2['text_raw'].apply(lambda x: 1 if ':' in x[:450] else 0)
    x_m2['before_colon'] = x_m2['text_raw'].apply(lambda x: word_before_colon(x[:450]))
    x_m2['after_colon'] = x_m2['text_raw'].apply(lambda x: word_after_colon(x[:450]))
    x_m2['qus_mark'] = x_m2['text_raw'].apply(lambda x: 1 if '?' in x[:450] else 0)
    x_m2['pre_caps'] = x_m2['text_raw'].apply(lambda x: pre_caps(x[:450]))
    x_m2['pre_pro_noun'] = x_m2['text_raw'].apply(lambda x: round(proper_noun(x[:450])[2],3))
    x_m2['word_present'] = x_m2['text_raw'].apply(lambda x: word_present(x[:450]))
    x_m2['math_expresion'] = x_m2['text_raw'].apply(lambda x: math_expression(x[:50]))
    x_m2['number_only'] = x_m2['text_raw'].apply(lambda x: number_only(x[:450]))
    x_m2['roman_only'] = x_m2['text_raw'].apply(lambda x: roman_only(x[:450]))
    x_m2['start_dot'] = x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().startswith(('.','·')) else 0)
    x_m2['start_minus'] = x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().startswith('-') else 0)
    x_m2['start_equal'] = x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().startswith('=') else 0)
    x_m2['start_bracket']=  x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().startswith(("(", "[", "{", "<", "<<", "<<<", "<<<<")) else 0)
    x_m2['start_notes']=  x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().startswith(("Note", "Notes", "note", "notes")) else 0)
    x_m2['end_comma'] = x_m2['text_raw'].apply(lambda x: 1 if x[:450].strip().endswith(',') else 0)
    x_m2['DNA_seq'] = x_m2['text_raw'].apply(lambda x: DNA_seq(x[:450]))
    x_m2['no_after_word'] = x_m2['text_raw'].apply(lambda x: no_after_word(x[:450]))
    x_m2['number_hir'] = x_m2['text_raw'].apply(lambda x: number_hir(x[:450]))
    x_m2['year_persent'] = x_m2['text_raw'].apply(lambda x: year_persent(x[:450]))
    x_m2['sign_persent'] = x_m2['text_raw'].apply(lambda x: sign_persent(x[:450]))
    x_m2['startswthnum']=x_m2['text_raw'].apply(lambda x: startswthnum(x[:450]))
    x_m2['startswthalpha']=x_m2['text_raw'].apply(lambda x: startswthalpha(x[:450]))
    x_m2['startswthroman']=x_m2['text_raw'].apply(lambda x: startswthroman(x[:450]))
    x_m2['state_words']=x_m2['text_raw'].apply(lambda x: state_words(x[:450]))
    return x_m2

def lag_lead_feature(df2):
    df = df2.copy()
    list_class= ['H1','H2','H3','H4','H5','H6','Heading1','Heading2','Heading3','Heading4',
                 'Heading5','Heading6','DisplayEquation','Paratext','Paragraph','Body','Normal','missing']
    list_s_class = ['H1','H2','H3','H4','H5','H6','Heading1','Heading2','Heading3','Heading4',
                    'Heading5','Heading6','DisplayEquation','Paratext','Paragraph','Body','Normal','missing']
    def match(x):
        for i in list_class:
            if i.lower() in x.lower().replace(r'-/*',''):
                return i
        return 'others'
    df['class1'] = df['class'].apply(lambda x: match(x))
    df['s_class1'] = df['s_class'].apply(lambda x: match(x))
    df1 =df.groupby('file_id').agg({'parent_id':['max','min']})
    df1.columns = ['_'.join(col) for col in df1.columns.values]
    df1.reset_index(inplace=True)
    df=df.merge(df1,on='file_id',how="inner")
    df['word_ind']=np.where(df['word_len']<20,1,0)
    df['char_len']=df['run_text'].str.len()
    df['char_ind']=np.where(df['char_len'].between(3,100),1,0)
    df['word_char_ind'] = np.where((df['char_ind']==1) & (df['word_ind']==1) ,1,0)
    # lag and lead varibles
    df['lag_align']=df['align'].shift(periods=1)
    df['lead_align']=df['align'].shift(periods=-1)
    df['lag_tag']=df['tag'].shift(periods=1)
    df['lead_tag']=df['tag'].shift(periods=-1)
    df['lag_stag']=df['s_tag'].shift(periods=1)
    df['lead_stag']=df['s_tag'].shift(periods=-1)
    df['lag_class1']=df['class1'].shift(periods=1)
    df['lead_class1']=df['class1'].shift(periods=-1)
    df['lag_s_class1']=df['s_class1'].shift(periods=1)
    df['lead_s_class1']=df['s_class1'].shift(periods=-1)
    df['lag_ntext']=df['ntext'].shift(periods=1)
    df['lead_ntext']=df['ntext'].shift(periods=-1)
    df['lag_word_len']=df['word_len'].shift(periods=1)
    df['lead_word_len']=df['word_len'].shift(periods=-1)
    df['lag_s_font_size']=df['s_font_size'].shift(periods=1)
    df['lead_s_font_size']=df['s_font_size'].shift(periods=-1)
    df['lag_title_case']=df['title_case'].shift(periods=1)
    df['lead_title_case']=df['title_case'].shift(periods=-1)
    df['lag_startswthnum']=df['startswthnum'].shift(periods=1)
    df['lead_startswthnum']=df['startswthnum'].shift(periods=-1)
    df['lag_startswthalpha']=df['startswthalpha'].shift(periods=1)
    df['lead_startswthalpha']=df['startswthalpha'].shift(periods=-1)
    # fillna for a file
    df.loc[df['parent_id']==df['parent_id_min'],'lag_align']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_align']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_tag']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_tag']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_stag']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_stag']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_class1']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_class1']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_s_class1']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_s_class1']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_ntext']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_ntext']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_word_len']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_word_len']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_s_font_size']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_s_font_size']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_title_case']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_title_case']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_startswthnum']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_startswthnum']=np.nan
    df.loc[df['parent_id']==df['parent_id_min'],'lag_startswthalpha']=np.nan
    df.loc[df['parent_id']==df['parent_id_max'],'lead_startswthalpha']=np.nan
    # missing value
    df=df.fillna({'run_text':' ','align':'missing', 'lag_align': 'missing',
                  'lead_align': 'missing','lag_tag': 'missing','lead_tag': 'missing',
                  'lag_stag': 'missing','lead_stag': 'missing','lag_class1': 'missing',
                  'lead_class1': 'missing','lag_s_class1': 'missing','lead_s_class1': 'missing'})
    df = df.fillna(0)
    return df
