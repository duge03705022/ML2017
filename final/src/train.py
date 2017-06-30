# -*- coding: utf-8 -*-
import re,os
import random,pickle
import time
import datetime 
import operator
import numpy as np
import pandas as pd 
import collections
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,StratifiedKFold
from xgboost import XGBClassifier
import xgboost as xgb

from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore") 

curTime = int(time.time())

def post_average(df,wins):
    total = 0.0
    df_list = df.total_cases.tolist()
    res = np.zeros((len(df_list,)),dtype="float32")
    for i in range(wins,len(df_list)):
        for j in range(wins):
            total = total + df_list[i-j-1]
        res[i] = float(total)/wins
        total = 0.0
    df["ma_%s" % str(wins)] = res
    return df

def post_label(df,wins):
    total = 0.0
    df_list = df.total_cases.tolist()
    res = np.zeros((len(df_list,)),dtype="float32")
    for j in range(wins):
        for i in range(j+1,len(df_list)):
            res[i] = df_list[i-j-1]
        df["post_%s" % str(j+1)] = res
        res = np.zeros((len(df_list,)),dtype="float32")
    return df

def processing_function(df):
    
    df.fillna(df.median(), inplace=True)
    #df.fillna(df.mode(), inplace=True)
    #df.fillna(df.mean(), inplace=True)
    #df.fillna(0.0, inplace=True)
    #df.fillna(method='ffill', inplace=True)
    
    df_sj = df.loc[df.city == 'sj']
    df_iq = df.loc[df.city == 'iq']
    
    return df_sj, df_iq

def add_groupby_feature(train_df, test_df): 
    train_df['month'] = [int(k[5:7]) for k in train_df.week_start_date]
    test_df['month'] = [int(k[5:7]) for k in test_df.week_start_date]
    
    cols = train_df.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    train_df = train_df[cols]
    cols.remove('total_cases')
    cols = test_df.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    test_df = test_df[cols]
    
    five_year_ago = train_df.year.max() - 5
    df_month = train_df[train_df.year >= five_year_ago].groupby('month', as_index = False)
    df_week = train_df[train_df.year >= five_year_ago].groupby('weekofyear', as_index = False)
    
    feature_name = 'mean_total_cases_by_month'
    
    total_cases_by_month = df_month['total_cases'].mean()
    total_cases_by_month.rename(columns={'total_cases': feature_name}, inplace=True)
    
    total_cases_by_week = df_week['total_cases'].mean()
    total_cases_by_week.rename(columns={'total_cases': 'mean_total_cases_by_week'}, inplace=True)
    
    train_merged_df = pd.merge(train_df, total_cases_by_month, how = 'left', on='month')
    train_merged_df = pd.merge(train_merged_df, total_cases_by_week, how = 'left', on='weekofyear')
    test_merged_df = pd.merge(test_df, total_cases_by_month, how = 'left', on='month')
    test_merged_df = pd.merge(test_merged_df, total_cases_by_week, how = 'left', on='weekofyear')
    
    return train_merged_df, test_merged_df

def add_neighbor_feature(train_df,wins):
    res = train_df.copy()
    cols = train_df.columns.tolist()
    cols.remove('city')
    cols.remove('year')
    cols.remove('weekofyear')
    cols.remove('week_start_date')
    if 'total_cases' in cols:
        cols.remove('total_cases')
    for i in range(wins):
        pre_new_cols = ["pre_%s_%s" % (str(i),s) for s in cols]
        after_new_cols = ["after_%s_%s" % (str(i),s) for s in cols]
                
        pre_list = np.concatenate([[[np.NaN for z in range(len(pre_new_cols))] for y in range(i+1)],train_df[cols].values[:-i-1]])
        pre_train_df = pd.DataFrame(pre_list,columns=pre_new_cols)
        pre_train_df = pre_train_df.fillna(pre_train_df.median(),inplace=True)
        pre_train_df["year"] = train_df["year"].values
        pre_train_df["weekofyear"] = train_df["weekofyear"].values

        after_list = np.concatenate([train_df[cols].values[i+1:],[[np.NaN for z in range(len(after_new_cols))] for y in range(i+1)]])
        after_train_df = pd.DataFrame(after_list,columns=after_new_cols)
        after_train_df = after_train_df.fillna(after_train_df.median(),inplace=True)
        after_train_df["year"] = train_df["year"].values
        after_train_df["weekofyear"] = train_df["weekofyear"].values
        
        new_df = pd.merge(pre_train_df, after_train_df, how = 'right', on=['year','weekofyear'])
        res = pd.merge(res, new_df, how = 'left', on=['year','weekofyear'])
    
    return res

def get_XGBmodel(dtrain,dvalue,dtest,dtestvalue,n_est,max_depth,n_feature):
    params = {"objective": "reg:linear",'eta':'0.02','subsample':'0.7','seed':42}
    params["max_depth"] = max_depth
    params["nthread"] = 8
    params['eval_metric'] = 'mae'
    # params["max_depth"] = max_depth
    # params["nthread"] = 8
    #evallist = [(dtest,'eval')]
    
    num_round = 600
    # params["n_estimators"] = n_est
    
    model_cols = random.sample(dtrain.columns.tolist(),n_feature)

    dtrain = xgb.DMatrix(dtrain[model_cols],label=dvalue)
    dtest = xgb.DMatrix(dtest[model_cols],label=dtestvalue)

    evallist = {(dtrain,'eval1'),(dtest, 'eval2')}
    
    xgbmodel = xgb.train(params,dtrain,num_round,evals=evallist,early_stopping_rounds= 100)
    
    return model_cols,xgbmodel

# load data
feature = pd.read_csv('data/dengue_features_train.csv', infer_datetime_format=True)
label = pd.read_csv('data/dengue_labels_train.csv')
test = pd.read_csv('data/dengue_features_test.csv')

df = pd.merge(feature, label, how='outer', on=label.columns.tolist()[:-1])
df = df.dropna(axis=0, thresh=10)
df_sj = df[df.city == 'sj'].copy()
df_iq = df[df.city == 'iq'].copy()

sj, iq = processing_function(df)
test_sj, test_iq = processing_function(test)

sj = add_neighbor_feature(sj,4)
iq = add_neighbor_feature(iq,4)
test_sj = add_neighbor_feature(test_sj,4)
test_iq = add_neighbor_feature(test_iq,4)

#sj = post_label(sj,4)
#iq = post_label(iq,4)
sj = post_label(sj,4)
iq = post_label(iq,4)
for i in range(2,5):
    sj = post_average(sj,i)
    iq = post_average(iq,i)

#sj = post_label2(sj,4)
#iq = post_label2(iq,4)


sj_merged_train, sj_merged_test = add_groupby_feature(sj, test_sj) 
iq_merged_train, iq_merged_test = add_groupby_feature(iq, test_iq) 

sj_value = sj_merged_train['total_cases']
iq_value = iq_merged_train['total_cases']

sj_wc = sj_value.tolist()
iq_wc = iq_value.tolist()

ignore_feature_list = ['city', 'week_start_date', 'total_cases','post_1','post_2','post_3','post_4','ma_2','ma_3','ma_4','mean_total_cases_by_month','mean_total_cases_by_week']
predictors = [feature for feature in sj_merged_train.columns.tolist() if feature not in ignore_feature_list]
predictors = predictors + ['mean_total_cases_by_month']
predictors = predictors + ['mean_total_cases_by_week']
t_predictors = predictors
#t_predictors = t_predictors+['ma_2','ma_3','ma_4']
#t_predictors = t_predictors+['ma_4']
t_predictors = t_predictors+['post_1','post_2']
t_predictors = t_predictors+['post_3','post_4']
target = 'total_cases'

#print(iq_merged_train['weekofyear'])

sj_train,sj_valid,sj_t_value,sj_v_value = train_test_split(sj_merged_train[t_predictors], sj_value, test_size=0.01)
iq_train,iq_valid,iq_t_value,iq_v_value = train_test_split(iq_merged_train[t_predictors], iq_value, test_size=0.01)

sj_test = sj_merged_test[predictors]
iq_test = iq_merged_test[predictors]

# sj_dtrain = xgb.DMatrix(sj_train,label=sj_t_value)
# sj_dvalid = xgb.DMatrix(sj_valid,label=sj_v_value)
# iq_dtrain = xgb.DMatrix(iq_train,label=iq_t_value)
# iq_dvalid = xgb.DMatrix(iq_valid,label=iq_v_value)

n_model = 1
sj_models=[]
iq_models=[]
for i in range(n_model):
    print(i)
    # sj_models.append(get_XGBmodel(sj_train,sj_t_value,sj_valid,sj_v_value,600,4,int(sj_train.shape[1]/2)+1))
    # iq_models.append(get_XGBmodel(iq_train,iq_t_value,iq_valid,iq_v_value,226,4,int(iq_train.shape[1]/2)+1))
    sj_models.append(get_XGBmodel(sj_train,sj_t_value,sj_valid,sj_v_value,600,4,int(sj_train.shape[1])))
    iq_models.append(get_XGBmodel(iq_train,iq_t_value,iq_valid,iq_v_value,226,4,int(iq_train.shape[1])))
    #print(sj_models[i][1].feature_importances_)
    #print(iq_models[i][1].feature_importances_)
    #print("\n")

sj_v_pred = []
iq_v_pred = []
for i in range(n_model):
    v_test = xgb.DMatrix(sj_valid[sj_models[i][0]])
    sj_v_pred.append(sj_models[i][1].predict(v_test))
    v_test = xgb.DMatrix(iq_valid[iq_models[i][0]])
    iq_v_pred.append(iq_models[i][1].predict(v_test))
    
sj_v_pred = np.array(sj_v_pred)
iq_v_pred = np.array(iq_v_pred)
sj_v_pred[np.where(sj_v_pred < 0)] = 0.0
iq_v_pred[np.where(iq_v_pred < 0)] = 0.0

#sj_v_pred = pd.ewma(sj_v_pred,span=4)
#iq_v_pred = pd.ewma(iq_v_pred,span=4)

sj_v_pred_1 = np.round(sj_v_pred.mean(axis=0)).astype(int)
iq_v_pred_1 = np.round(iq_v_pred.mean(axis=0)).astype(int)

sj_v_pred_2 = np.round(sj_v_pred).astype(int).transpose()
iq_v_pred_2 = np.round(iq_v_pred).astype(int).transpose()

sj_v_pred_2 = np.array([np.bincount(i).argmax() for i in sj_v_pred_2])
iq_v_pred_2 = np.array([np.bincount(i).argmax() for i in iq_v_pred_2])

sj_v = np.array([i for i in sj_v_value.tolist()])
iq_v = np.array([i for i in iq_v_value.tolist()])
print("average")
print((np.abs(sj_v - sj_v_pred_1)).mean())
print((np.abs(iq_v - iq_v_pred_1)).mean())

pred = np.concatenate((sj_v_pred_1 , iq_v_pred_1), axis=0)
v = np.concatenate((sj_v , iq_v), axis=0)
print((np.abs(v - pred)).mean())

#score = (np.abs(v - pred)).mean()

print("voting")
print((np.abs(sj_v - sj_v_pred_2)).mean())
print((np.abs(iq_v - iq_v_pred_2)).mean())

pred = np.concatenate((sj_v_pred_2 , iq_v_pred_2), axis=0)
v = np.concatenate((sj_v , iq_v), axis=0)
print((np.abs(v - pred)).mean())

score = (np.abs(v - pred)).mean()

sj_res = []
sj_test_wc = []
sj_test_wc = sj_wc
#print(sj_test_wc)
for x in range(sj_test.shape[0]):
    sj_pred = []
    test_entry = sj_test.loc[[x],:]
    test_entry['ma_2'] = np.array(sj_test_wc[-2:]).mean()
    test_entry['ma_3'] = np.array(sj_test_wc[-3:]).mean()
    test_entry['ma_4'] = np.array(sj_test_wc[-4:]).mean()
    test_entry['post_1'] = sj_test_wc[-1]
    test_entry['post_2'] = sj_test_wc[-2]
    test_entry['post_3'] = sj_test_wc[-3]
    test_entry['post_4'] = sj_test_wc[-4]
    
    #print(test_entry[['ma_4']])
    
    for i in range(n_model):
        test = xgb.DMatrix(test_entry[sj_models[i][0]])
        sj_pred.append(sj_models[i][1].predict(test))

    sj_pred = np.array(sj_pred)
    sj_pred[np.where(sj_pred < 0)] = 0.0
    #sj_pred = pd.ewma(sj_pred,span=4)

    #sj_pred = np.round(sj_pred.mean(axis=0)).astype(int)
    
    sj_pred = np.round(sj_pred).astype(int).transpose()
    #print(sj_pred)
    
    sj_pred = np.array([np.bincount(i).argmax() for i in sj_pred])

    sj_res.append(sj_pred[0])
    sj_test_wc.append(sj_pred[0])

iq_res = []
iq_test_wc = iq_wc
for x in range(iq_test.shape[0]):
    iq_pred = []
    test_entry = iq_test.loc[[x],:]
    test_entry['ma_2'] = np.array(iq_test_wc[-2:]).mean()
    test_entry['ma_3'] = np.array(iq_test_wc[-3:]).mean()
    test_entry['ma_4'] = np.array(iq_test_wc[-4:]).mean()
    test_entry['post_1'] = iq_test_wc[-1]
    test_entry['post_2'] = iq_test_wc[-2]
    test_entry['post_3'] = iq_test_wc[-3]
    test_entry['post_4'] = iq_test_wc[-4]
    
        
    for i in range(n_model):
        test = xgb.DMatrix(test_entry[iq_models[i][0]])
        iq_pred.append(iq_models[i][1].predict(test))

    iq_pred = np.array(iq_pred)
    iq_pred[np.where(iq_pred < 0)] = 0.0
    #iq_pred = pd.ewma(iq_pred,span=4)

    #iq_pred = np.round(iq_pred.mean(axis=0)).astype(int)
    
    iq_pred = np.round(iq_pred).astype(int).transpose()
    #print(iq_pred)
    
    iq_pred = np.array([np.bincount(i).argmax() for i in iq_pred])

    iq_res.append(iq_pred[0])
    iq_test_wc.append(iq_pred[0])

sj_merged_test['total_cases'] = sj_res
iq_merged_test['total_cases'] = iq_res
result = pd.concat([sj_merged_test, iq_merged_test])
result.total_cases = result.total_cases.astype(int)

os.makedirs(str(curTime))
for i in range(n_model):
    sj_models[i][1].save_model(str(curTime)+"/sj_%s.model" % (str(i))) 
    sj_models[i][1].dump_model( str(curTime)+"/sj_%s.raw.txt" % (str(i)))
    pickle.dump(sj_models[i][0], open(str(curTime) + "/sj_feature_%s.pkl" % (str(i)), "wb"))
    iq_models[i][1].save_model(str(curTime)+"/iq_%s.model" % (str(i))) 
    iq_models[i][1].dump_model( str(curTime)+"/iq_%s.raw.txt" % (str(i)))
    pickle.dump(iq_models[i][0], open(str(curTime) + "/iq_feature_%s.pkl" % (str(i)), "wb"))
result[['city','year','weekofyear','total_cases']].to_csv( str(curTime) + '/' + str(n_model)+'_'+str(int(sj_train.shape[1]/2)+1) +'_' + str(score) + '_submission.csv', index=False)
#, str(curTime)+"/sj_%s_featuremap.txt" % (str(i) ) 