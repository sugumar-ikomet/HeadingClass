import os
import re
import joblib
import numpy as np
import pandas as pd
import random as rn
from clean_text import *
from config import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Activation, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, add, multiply, concatenate
# print(tf.__version__)


def preprocessing(data_x,le_dir=le_dir,ohe_model_dir=ohe_model_dir,tk_dir=tk_dir):
	df = data_x.copy()
	def clean(x):
	    return ' '.join(['<num>' if i.isnumeric() else i for i in str(x).split()])
	df['run_text'] = df['run_text'].apply(lambda x: ' '.join(x.split()[:20])) # first 20 words
	df['run_text'] = df['run_text'].apply(lambda x:clean(x))
	df['run_text'] = df['run_text'].str.replace('\d+','<num>')
	df['run_text'] = df['run_text'].str.replace('[^\w\s]',' ')
	df['run_text'] = df['run_text'].str.replace(r'[\s]+', ' ')
	df['run_text'] = df['run_text'].str.replace(r'[^A-Za-z0-9]+', ' ')
	df['run_text'] = df['run_text'].str.lower()
	#label encoding
	dict1 = {'align':'align','lag_align':'lag_align','lead_align':'lead_align','lag_tag':'lag_tag',
	         'lead_tag':'lead_tag','s_tag':'s_tag','lag_stag':'lag_stag','lead_stag':'lead_stag',
	         's_align':'s_align','class1':'class1','s_class1':'s_class1','lag_class1':'lag_class1',
	         'lead_class1':'lead_class1','lag_s_class1':'lag_s_class1', 
	         'lead_s_class1':'lead_s_class1'}
	le_dict = joblib.load(le_dir)
	for (k,v) in dict1.items():
	    try:
	        df[k] = le_dict[k].transform(df[v])
	    except:
	        df[v] = df[v].map(lambda s: '<unknown>' if s not in le_dict[k].classes_ else s)
	        le_dict[k].classes_ = np.append(le_dict[k].classes_, '<unknown>')
	        df[k] = le_dict[k].transform(df[v])
	# one-hot encoding
	ohe = joblib.load(ohe_model_dir)
	categorical_cols = ['align','lag_align','lead_align','lag_tag','lead_tag','s_tag',
	                    'lag_stag','lead_stag','s_align','class1','s_class1',
	                    'lag_class1','lead_class1','lag_s_class1','lead_s_class1']
	feature_arr = ohe.transform(df[categorical_cols])
	features = pd.DataFrame(feature_arr, columns=list(ohe.get_feature_names()))
	df1=pd.concat([df,features],axis=1)
	x_cols = ['ntext', 'x0_a_max', 'x0_a_min', 'x0_alternatives_max', 'x0_alternatives_min', 'x0_b_max', 'x0_b_min', 'x0_cite_max', 'x0_cite_min', 'x0_div_max', 'x0_div_min', 'x0_graphic_max', 'x0_graphic_min', 'x0_h1_max', 'x0_h1_min', 'x0_h2_max', 'x0_h2_min', 'x0_h3_max', 'x0_h3_min', 'x0_h4_max', 'x0_h4_min', 'x0_h5_max', 'x0_h5_min', 'x0_h6_max', 'x0_h6_min', 'x0_i_max', 'x0_i_min', 'x0_inline-graphic_max', 'x0_inline-graphic_min', 'x0_li_max', 'x0_li_min', 'x0_math_max', 'x0_math_min', 'x0_mfrac_max', 'x0_mfrac_min', 'x0_mi_max', 'x0_mi_min', 'x0_mmultiscripts_max', 'x0_mmultiscripts_min', 'x0_mn_max', 'x0_mn_min', 'x0_mover_max', 'x0_mover_min', 'x0_mroot_max', 'x0_mroot_min', 'x0_mrow_max', 'x0_mrow_min', 'x0_msqrt_max', 'x0_msqrt_min', 'x0_mstyle_max', 'x0_mstyle_min', 'x0_msub_max', 'x0_msub_min', 'x0_msubsup_max', 'x0_msubsup_min', 'x0_msup_max', 'x0_msup_min', 'x0_mtable_max', 'x0_mtable_min', 'x0_mtd_max', 'x0_mtd_min', 'x0_mtext_max', 'x0_mtext_min', 'x0_mtr_max', 'x0_mtr_min', 'x0_munder_max', 'x0_munder_min', 'x0_munderover_max', 'x0_munderover_min', 'x0_nh_max', 'x0_nh_min', 'x0_ol_max', 'x0_ol_min', 'x0_p_max', 'x0_p_min', 'x0_pre_max', 'x0_pre_min', 'x0_s_max', 'x0_s_min', 'x0_span_max', 'x0_span_min', 'x0_strong_max', 'x0_strong_min', 'x0_table_max', 'x0_table_min', 'x0_tbody_max', 'x0_tbody_min', 'x0_td_max', 'x0_td_min', 'x0_tex-math_max', 'x0_tex-math_min', 'x0_thead_max', 'x0_thead_min', 'x0_tr_max', 'x0_tr_min', 'x0_tt_max', 'x0_tt_min', 'x0_u_max', 'x0_u_min', 'x0_ul_max', 'x0_ul_min', 'word_len_max', 'word_len_min', 'font_size_max', 'font_size_min', 'sentence_case_max', 'sentence_case_min', 'title_case_max', 'title_case_min', 'margin_bottom_max', 'margin_bottom_min', 'margin_left_max', 'margin_left_min', 'margin_right_max', 'margin_right_min', 'margin_top_max', 'margin_top_min', 'text_indent_max', 'text_indent_min', 'marign_max', 'marign_min', 'h1_max', 'h1_min', 'h2_max', 'h2_min', 'h3_max', 'h3_min', 'h4_max', 'h4_min', 'h5_max', 'h5_min', 'h6_max', 'h6_min', 'b_strong_max', 'b_strong_min', 'em_i_max', 'em_i_min', 'u_max', 'u_min', 'word_len', 'sentence_case', 'title_case', 'upper_case', 'lower_case', 'startswthnum', 'text_indent', 'max_font_size', 'min_font_size', 'no_text', 's_word_len', 's_font_size', 's_text_indent', 's_margin', 'slevel', 'run_text', 'no_run_sent', 'run_word_len', 'caps_colon', 'contain_colon', 'before_colon', 'after_colon', 'qus_mark', 'pre_caps', 'pre_pro_noun', 'word_present', 'math_expresion', 'number_only', 'roman_only', 'start_dot', 'start_minus', 'start_equal', 'start_bracket', 'start_notes', 'end_comma', 'DNA_seq', 'no_after_word', 'number_hir', 'year_persent', 'sign_persent', 'startswthalpha', 'startswthroman', 'state_words', 'word_ind', 'char_len', 'char_ind', 'word_char_ind', 'lag_ntext', 'lead_ntext', 'lag_word_len', 'lead_word_len', 'lag_s_font_size', 'lead_s_font_size', 'lag_title_case', 'lead_title_case', 'lag_startswthnum', 'lead_startswthnum', 'lag_startswthalpha', 'lead_startswthalpha', 'x0_0', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'x1_0', 'x1_1', 'x1_2', 'x1_3', 'x1_4', 'x2_0', 'x2_1', 'x2_2', 'x2_3', 'x2_4', 'x3_0', 'x3_1', 'x3_2', 'x3_3', 'x3_4', 'x3_5', 'x3_6', 'x3_7', 'x3_8', 'x3_9', 'x3_10', 'x3_11', 'x3_12', 'x3_13', 'x3_14', 'x3_15', 'x3_16', 'x3_17', 'x4_0', 'x4_1', 'x4_2', 'x4_3', 'x4_4', 'x4_5', 'x4_6', 'x4_7', 'x4_8', 'x4_9', 'x4_10', 'x4_11', 'x4_12', 'x4_13', 'x4_14', 'x4_15', 'x4_16', 'x4_17', 'x5_0', 'x5_1', 'x5_2', 'x5_3', 'x5_4', 'x5_5', 'x5_6', 'x5_7', 'x5_8', 'x5_9', 'x5_10', 'x5_11', 'x5_12', 'x5_13', 'x5_14', 'x6_0', 'x6_1', 'x6_2', 'x6_3', 'x6_4', 'x6_5', 'x6_6', 'x6_7', 'x6_8', 'x6_9', 'x6_10', 'x6_11', 'x6_12', 'x6_13', 'x6_14', 'x7_0', 'x7_1', 'x7_2', 'x7_3', 'x7_4', 'x7_5', 'x7_6', 'x7_7', 'x7_8', 'x7_9', 'x7_10', 'x7_11', 'x7_12', 'x7_13', 'x7_14', 'x8_0', 'x8_1', 'x8_2', 'x8_3', 'x8_4', 'x9_0', 'x9_1', 'x9_2', 'x9_3', 'x9_4', 'x9_5', 'x9_6', 'x9_7', 'x9_8', 'x9_9', 'x9_10', 'x9_11', 'x9_12', 'x9_13', 'x9_14', 'x9_15', 'x10_0', 'x10_1', 'x10_2', 'x10_3', 'x10_4', 'x10_5', 'x10_6', 'x10_7', 'x10_8', 'x10_9', 'x10_10', 'x10_11', 'x10_12', 'x10_13', 'x10_14', 'x10_15', 'x11_0', 'x11_1', 'x11_2', 'x11_3', 'x11_4', 'x11_5', 'x11_6', 'x11_7', 'x11_8', 'x11_9', 'x11_10', 'x11_11', 'x11_12', 'x11_13', 'x11_14', 'x11_15', 'x12_0', 'x12_1', 'x12_2', 'x12_3', 'x12_4', 'x12_5', 'x12_6', 'x12_7', 'x12_8', 'x12_9', 'x12_10', 'x12_11', 'x12_12', 'x12_13', 'x12_14', 'x12_15', 'x13_0', 'x13_1', 'x13_2', 'x13_3', 'x13_4', 'x13_5', 'x13_6', 'x13_7', 'x13_8', 'x13_9', 'x13_10', 'x13_11', 'x13_12', 'x13_13', 'x13_14', 'x13_15', 'x14_0', 'x14_1', 'x14_2', 'x14_3', 'x14_4', 'x14_5', 'x14_6', 'x14_7', 'x14_8', 'x14_9', 'x14_10', 'x14_11', 'x14_12', 'x14_13', 'x14_14', 'x14_15']
	df2 = df1[x_cols]
	df_texts = df2['run_text'].values
	df_texts = [str(s).lower() for s in df_texts]
	tk = joblib.load(tk_dir)
	df_texts = tk.texts_to_sequences(df_texts)
	df_data = pad_sequences(df_texts, maxlen=300, padding='post')
	df_db = df2.drop('run_text',axis=1).values
	return df_data, df_db

def model_classification(df, le_dir=le_dir, ohe_model_dir=ohe_model_dir, tk_dir=tk_dir, model_dir=model_dir):
	df2=df.copy()
	df_data, df_db = preprocessing(df2)
	le_dict = joblib.load(le_dir)
	loaded_model = tf.keras.models.load_model(model_dir)
	y_pred = loaded_model.predict([df_data,df_db])
	y_pred1 = np.argmax(y_pred,axis=1)
	y_class = le_dict['target'].inverse_transform(y_pred1)
	df2['pred'] = y_class
	df2['model_prob']=np.max(y_pred,axis=1)
	return df2

