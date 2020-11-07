import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix



def rule_final(df1):
    # Rule1
    x1 = df1.copy()
    x1=x1.drop(['parent_id_max','parent_id_min'],axis=1)
    df1 =x1.groupby('file_id').agg({'parent_id':['max','min']})
    df1.columns = ['_'.join(col) for col in df1.columns.values]
    df1.reset_index(inplace=True)
    x1=x1.merge(df1,on='file_id',how="inner")
    x1['lag_pred']=x1['pred'].shift(periods=1)
    x1.loc[x1['parent_id']==x1['parent_id_min'],'lag_pred']=np.nan
    x1.loc[x1['parent_id']==x1['parent_id_max'],'lag_pred']=np.nan
    x1=x1.fillna({'lag_pred':'missing'})

    x1['lag_align']=x1['align'].shift(periods=1)
    x1.loc[x1['parent_id']==x1['parent_id_min'],'lag_align']=np.nan
    x1.loc[x1['parent_id']==x1['parent_id_max'],'lag_align']=np.nan
    x1=x1.fillna({'lag_align':'missing'})

    x1['lag_b_strong_max']=x1['b_strong_max'].shift(periods=1)
    x1.loc[x1['parent_id']==x1['parent_id_min'],'lag_b_strong_max']=np.nan
    x1.loc[x1['parent_id']==x1['parent_id_max'],'lag_b_strong_max']=np.nan
    x1=x1.fillna({'lag_b_strong_max':0})
    def same_text(df):
        a = df['text_raw'].strip().split()
        b = df['s_text_raw'].strip().split()
        if a[0]==b[0]:
            return 1
        else:
            return 0
    x1['same_text'] = x1.apply(same_text, axis=1) 
    x1['rule1_ind'] = np.where((x1["pred"]=="ParaText")&(x1['lag_pred'].str.contains('Heading'))
                          &(x1['run_word_len']<10)&(x1['s_word_len'].between(1,5))&(x1['b_strong_max']==1)
                        &(x1['s_tag']=='b')&(x1['sign_persent']==0)&(x1['lag_b_strong_max']==1)&(x1['same_text']==1)&
                        (x1['state_words']==0)&(x1['align']==x1['lag_align']),1,0)
    x1['new_pred0'] = np.where((x1['rule1_ind']==1),x1['lag_pred'],x1['pred'])
    # Rule2
    x1=x1.drop(['parent_id_max','parent_id_min'],axis=1)
    df = x1.copy()
    df1 = df[df.pred.str.contains('Heading')].groupby(['file_id']).agg({'parent_id':'min','align' : lambda x: x.iloc[0]}).reset_index()
    df1.rename(columns={'parent_id':'min_pid','align':'min_align'},inplace=True)
    df2 = df.merge(df1,on="file_id",how="left")
    df2['rule2_ind']=np.where((df2['parent_id']==df2['min_pid']),1,0)
    df2['new_pred1']=df2['new_pred0']
    df2.loc[df2['rule2_ind']==1,'new_pred1']='Heading1'

    # Rule3
    df2 = df2.sort_values(by=['file_id','parent_id'])
    def lhead(x):
        try:
            return int(x[-1:])+1 
        except:
            return 0
    df2["rule3_ind"] = np.where((df2['rule2_ind']==0)&(df2["new_pred1"]=="Heading1")&(df2["min_align"]=="center")&(df2["align"]!="center"),1,0)
    df2["new_pred2"] = np.where(df2["rule3_ind"]==1,"Heading2",df2["new_pred1"])

    # Rule4
    df1 =df2.groupby('file_id').agg({'parent_id':['max','min']})
    df1.columns = ['_'.join(col) for col in df1.columns.values]
    df1.reset_index(inplace=True)
    df2=df2.merge(df1,on='file_id',how="inner")
    df2['lag_new_pred2']=df2['new_pred2'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_new_pred2']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_new_pred2']=np.nan
    df2['lag_ntext']=df2['ntext'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_ntext']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_ntext']=np.nan
    df2['lag_align']=df2['align'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_align']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_align']=np.nan
    df2['lag_text_indent']=df2['text_indent'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_text_indent']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_text_indent']=np.nan
    df2=df2.fillna({'lag_new_pred2':'missing','lag_ntext':0,'lag_align':'missing','lag_text_indent':0})
    df2['last_head']=df2['lag_new_pred2'].apply(lambda x: lhead(x))
    df2['rule4_ind']=np.where(((df2['rule2_ind']==0)&(df2['ntext']==1) & (df2['lag_ntext']==1) &
               (df2.new_pred2.str.contains('Heading')) & (df2.lag_new_pred2.str.contains('Heading'))&(df2['lag_word_len']<20)
                               &(df2['word_len']<20)&(df2['sign_persent']==0)),1,0)

    df2['new_pred3']=df2['new_pred2']
    df2.loc[df2['rule4_ind']==1,'new_pred3'] = df2.loc[df2['rule4_ind']==1,'last_head'].apply(lambda x: 'Heading'+str(x) if x!=0 else 'missing')

    # Rule 5
    df2['lag_rule4_ind']=df2['rule4_ind'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_rule4_ind']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_rule4_ind']=np.nan
    df2['lag_new_pred3']=df2['new_pred3'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_new_pred3']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_new_pred3']=np.nan
    df2=df2.fillna({'lag_new_pred3':'missing','lag_rule4_ind':0})
    df2["rule5_ind"] = np.where((df2['rule2_ind']==0)&(df2["rule4_ind"]==1)&(df2["lag_rule4_ind"]==1),1,0)
    df2['last_head_r4']=df2['lag_new_pred3'].apply(lambda x: lhead(x))
    df2['new_pred4']=df2['new_pred3']
    df2.loc[df2['rule5_ind']==1,'new_pred4'] = df2.loc[df2['rule5_ind']==1,'last_head_r4'].apply(lambda x: 'Heading'+str(x) if x!=0 else 'missing')

    # Rule 6
    df2['lag_em_i_max']=df2['em_i_max'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_em_i_max']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_em_i_max']=np.nan
    df2['lag_b_strong_max']=df2['b_strong_max'].shift(periods=1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lag_b_strong_max']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_b_strong_max']=np.nan
    df2['lead_em_i_max']=df2['em_i_max'].shift(periods=-1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lead_em_i_max']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lead_em_i_max']=np.nan
    df2['lead_b_strong_max']=df2['b_strong_max'].shift(periods=-1)
    df2.loc[df2['parent_id']==df2['parent_id_min'],'lead_b_strong_max']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lead_b_strong_max']=np.nan
    df2=df2.fillna({'lag_em_i_max':0,'lead_em_i_max':0,'lag_b_strong_max':0,'lead_b_strong_max':0})
    df2['rule6_ind'] =  np.where((df2['rule2_ind']==0)&(df2['em_i_max']==1)&((df2['lag_em_i_max']==1)|(df2['lead_em_i_max']==1))
                                 &(df2['no_run_sent']>=1)&(df2['new_pred4']=='ParaText')&((df2['startswthalpha']==1)|(df2['startswthnum']==1))
                                 &(df2['run_word_len']<15)&(df2['model_prob']>0.90)&(df2['word_len']>20.0)
                                 &(df2['s_word_len']<15)&((df2['lag_word_len']>20.0)|(df2['lead_word_len']>20.0)),1,0)


    df2["new_pred5"] = np.where(df2["rule6_ind"]==1,"Heading4",df2["new_pred4"])
    # Rule 7
    def lhead2(x):
        try:
            return int(x[-1:])
        except:
            return 0
    df3 = df2[(df2.new_pred5.str.contains('Heading'))&(df2['number_hir']>0)]
    df3=df3.drop(['parent_id_max','parent_id_min'],axis=1)
    df31 =df3.groupby('file_id').agg({'parent_id':['max','min']})
    df31.columns = ['_'.join(col) for col in df31.columns.values]
    df31.reset_index(inplace=True)
    df3=df3.merge(df31,on='file_id',how="inner")
    df3['lag_number_hir']=df3['number_hir'].shift(periods=1)
    df3.loc[df3['parent_id']==df3['parent_id_min'],'lag_number_hir']=np.nan
    df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_number_hir']=np.nan
    df3=df3.fillna({'lag_number_hir':0})
    df3['rule7_ind'] = np.where(((df3['rule2_ind']==0)&(df3['lag_number_hir']>0)&(df3['number_hir']>0)),1,0)
    #########################
    df3['new_pred6'] = df3['new_pred5']
    for i in range(50):
        df3['lag_file_id']=df3['file_id'].shift(periods=1)
        df3.loc[df3['parent_id']==df3['parent_id_min'],'lag_file_id']=np.nan
        df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_file_id']=np.nan
        df3=df3.fillna({'lag_file_id':0})  
        df3['lag_new_pred6']=df3['new_pred6'].shift(periods=1)
        df3.loc[df3['parent_id']==df3['parent_id_min'],'lag_new_pred6']=np.nan
        df2.loc[df2['parent_id']==df2['parent_id_max'],'lag_new_pred6']=np.nan
        df3=df3.fillna({'lag_new_pred6':'missing'})
        df3['last_head2']=df3['lag_new_pred6'].apply(lambda x: lhead2(x))
        df3['hir_diff']=df3['number_hir']-df3['lag_number_hir']
        df3['hir_diff'] = df3['hir_diff'].astype(int)
        df3['add_hir'] = df3['last_head2']+df3['hir_diff']
        df3['add_hir'] = df3['add_hir'].astype(str)
        df3['new_pred6'] = np.where((df3['rule7_ind']==1)&(df3['file_id']==df3['lag_file_id']),'Heading'+df3['add_hir'],df3['new_pred5'])
    df4 = df3[['file_id','parent_id','new_pred6','rule7_ind']]
    df5 = df2.merge(df4,on=['file_id','parent_id'],how='left')
    df5['new_pred7'] = np.where((df5['new_pred6'].isna()),df5['new_pred5'],df5['new_pred6'])

    # Rule 8
    df5['rule8_ind'] = np.where((df5['state_words']==1),1,0)
    df5['new_pred8'] = np.where((df5['rule8_ind']==1),'Statement',df5['new_pred7'])

    # Rule 9
    df5['rule9_ind'] = np.where((df5.tag.str.contains('ul'))|(df5.tag.str.contains('table')),1,0)
    df5['new_pred9'] = np.where((df5['rule8_ind']==0)&(df5['rule9_ind']==1),'missing',df5['new_pred8'])

    # Rule 10    
    df6 = df5[(df5['startswthroman']==1)]
    df6['rule10_ind'] = np.where((df6['rule8_ind']==0)&(df6['rule9_ind']==0)&(df6['rule2_ind']==0)&(df6['startswthroman']==1),1,0)
    #########################
    df6['new_pred10'] = df6['new_pred9']
    for i in range(20):
        df6['lag_file_id']=df6['file_id'].shift(periods=1)
        df6.loc[df6['parent_id']==df6['parent_id_min'],'lag_file_id']=np.nan
        df6=df6.fillna({'lag_file_id':0})
        df6['lag_new_pred10']=df6['new_pred10'].shift(periods=1)
        df6.loc[df6['parent_id']==df6['parent_id_min'],'lag_new_pred10']=np.nan
        df6=df6.fillna({'lag_new_pred10':'missing'})
        df6['new_pred10'] = np.where((df6['rule10_ind']==1)&(df6['file_id']==df6['lag_file_id']),df6['lag_new_pred10'],df6['new_pred10'])
    df7 = df6[['file_id','parent_id','new_pred10','rule10_ind']]
    df8 = df5.merge(df7,on=['file_id','parent_id'],how='left')
    df8['new_pred11'] = np.where((df8['new_pred10'].isna()),df8['new_pred9'],df8['new_pred10'])

    # Rule 11
    # # new exclusion files
    def non_english(text):
        pattern = re.compile('[^a-zA-Z. ]')
        a = pattern.sub('', text)
        b = (len(a)/len(text))
        if b<0.4:
            return 1
        else:
            return 0
    df8['non_english'] = df8['text_raw'].apply(lambda x: non_english(x[:50]))
    df8['rule11_ind'] = np.where((df8['rule2_ind']==0)&(df8['rule8_ind']==0)&(df8['rule9_ind']==0)&(df8['non_english']==1)&(df8['year_persent']==0)&(df8['sign_persent']==0),1,0)
    df8['new_pred12'] = np.where((df8['rule11_ind']==1),'ParaText',df8['new_pred11'])

    # Rule 12
    df9 = df8[(df8.new_pred12.str.contains('Heading'))]
    df9=df9.drop(['parent_id_max','parent_id_min'],axis=1)
    df1 =df9.groupby('file_id').agg({'parent_id':['max','min']})
    df1.columns = ['_'.join(col) for col in df1.columns.values]
    df1.reset_index(inplace=True)
    df9=df9.merge(df1,on='file_id',how="inner")
    df9['lag_new_pred12']=df9['new_pred12'].shift(periods=1)
    df9.loc[df9['parent_id']==df9['parent_id_min'],'lag_new_pred12']=np.nan
    df9.loc[df9['parent_id']==df9['parent_id_max'],'lag_new_pred12']=np.nan
    df9=df9.fillna({'lag_new_pred12':'missing'})
    def lhead3(x):
        try:
            return int(x[-1:])
        except:
            return 100
    df9['last_head3']=df9['new_pred12'].apply(lambda x: lhead3(x))
    df9['last_head4']=df9['lag_new_pred12'].apply(lambda x: lhead3(x))
    def temp2(df9):
        if (df9['rule12_ind']==1)&(df9['file_id']==df9['lag_file_id']):
            a= df9['last_head3']-df9['diff_']+1
            return 'Heading'+str(a)
        else:
            return df9['new_pred12']
    df9['diff_'] = df9['last_head3']-df9['last_head4']
    df9['diff_'] = df9['diff_'].astype(int)
    df9['rule12_ind'] = np.where((df9['rule2_ind']==0)&(df9['rule11_ind']==0)&(df9['rule8_ind']==0)&(df9['rule9_ind']==0)&(df9['diff_']>1),1,0)
    for i in range(10):
        df9['lag_file_id']=df9['file_id'].shift(periods=1)
        df9.loc[df9['parent_id']==df9['parent_id_min'],'lag_file_id']=np.nan
        df9.loc[df9['parent_id']==df9['parent_id_max'],'lag_file_id']=np.nan
        df9=df9.fillna({'lag_file_id':0})
        df9['lag_diff_']=df9['diff_'].shift(periods=1)
        df9.loc[df9['parent_id']==df9['parent_id_min'],'lag_diff_']=np.nan
        df9.loc[df9['parent_id']==df9['parent_id_max'],'lag_diff_']=np.nan
        df9=df9.fillna({'lag_diff_':0})    
        df9['lag_diff_'] = df9['lag_diff_'].astype(int)
        df9['diff_'] = np.where(df9['diff_']==0,df9['lag_diff_'],df9['diff_'])
        df9['new_pred13'] = df9.apply(temp2,axis=1)
    df9_ = df9[['file_id','parent_id','new_pred13','rule12_ind']]
    df10 = df8.merge(df9_,on=['file_id','parent_id'],how='left')
    df10['new_pred14'] = np.where((df10['new_pred13'].isna()),df10['new_pred12'],df10['new_pred13'])

    # Rule 13
    def check1(text):
        a = word_tokenize(text)
        b = [i for i in a if (i.strip().lower()=='insert') | (i.strip().lower()=='here')]
        if len(b)>0:
            return 1
        else:
            return 0
    df10["rule13_ind"] = df10['text_raw'].apply(lambda x: check1(x[:50]))
    df10["new_pred15"] = np.where((df10['rule2_ind']==0)&(df10['rule11_ind']==0)&(df10['rule8_ind']==0)&(df10['rule9_ind']==0)&(df10["rule13_ind"]==1),"ParaText",df10["new_pred14"])

    # Rule 14
    df10["uch1"]= np.where((df10["upper_case"]==1)&(df10["new_pred15"]=="Heading1"),1,0)
    df10["uch1_in_file"] = df10.groupby('file_id')['uch1'].transform('max')
    df10["rule14_ind"] = np.where((df10['rule2_ind']==0)&(df10['rule11_ind']==0)&(df10['rule8_ind']==0)&(df10['rule9_ind']==0)&(df10["rule13_ind"]==0)
                                  &(df10["uch1_in_file"]==1)&(df10["new_pred15"]=="Heading1")&(df10["upper_case"]!=1),1,0)
    df10["new_pred16"] = np.where(df10["rule14_ind"]==1,"Heading2",df10["new_pred15"])
    # rule indidicator
    df10 = df10.fillna({'rule14_ind':0, 'rule13_ind':0, 'rule12_ind':0, 'rule11_ind':0, 'rule10_ind':0, 'rule9_ind':0, 'rule8_ind':0, 'rule7_ind':0, 'rule6_ind':0, 'rule5_ind':0, 'rule4_ind':0, 'rule3_ind':0, 'rule2_ind':0, 'rule1_ind':0})
    temp = [x for x in list(df10.columns) if x.startswith('rule') and 'ind' in x]
    df10['ind_sum'] = df10['rule1_ind'] + df10['rule2_ind'] + df10['rule3_ind'] + df10['rule4_ind'] + df10['rule5_ind'] + df10['rule6_ind'] + df10['rule7_ind'] + df10['rule8_ind'] + df10['rule9_ind'] + df10['rule10_ind'] + df10['rule11_ind'] + df10['rule12_ind'] + df10['rule13_ind'] + df10['rule14_ind']
    df10['ruleind'] = np.argmax(df10[['rule14_ind', 'rule13_ind', 'rule12_ind', 'rule11_ind', 'rule10_ind', 'rule9_ind', 'rule8_ind', 'rule7_ind', 'rule6_ind', 'rule5_ind', 'rule4_ind', 'rule3_ind', 'rule2_ind', 'rule1_ind']].values,axis=1)
    df10['ruleind2'] = np.where((df10['ruleind']==0)&(df10['rule14_ind']==0),0,14-df10['ruleind'])
    return df10