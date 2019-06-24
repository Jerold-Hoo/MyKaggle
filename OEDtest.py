# -*- coding: utf-8 -*-

# @File       : OEDtest.py
# @Date       : 2019-05-16
# @Author     : Jerold
# @Description:

import pymssql
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import pydotplus
import matplotlib.pyplot as plt

SQL = "select Temp8,Temp25,Temp41,Rspeed400,Rspeed580,DownSw,Dateinterval" \
      ",[OutPut] from Bs_SinCapsule where OilCode like 'RT%'"

SQLsin = "select * from OED_Sin"

def LoadData():
    conn = pymssql.connect(server='118.123.7.142',user='sa',password='2019newpass.7142sql',database='AA_DATA')
    data = pd.read_sql(SQLsin,conn)
    return data

def Do_data(data):

    print('ori_Num:', len(data))
    data.dropna(inplace=True)
    print('aft_Num:', len(data))

    Xname = data.columns.drop('OutPut')
    Yname = 'OutPut'
    X = data[Xname]
    Y = data[Yname]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    # 特征处理
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='records'))
    X_test = vec.fit_transform(X_test.to_dict(orient='records'))
    feature_names = vec.feature_names_

    return X_train,X_test,Y_train,Y_test,feature_names

def get_mod(X_train,Y_train,mode_type='randomF_reg',paras=None):

    if mode_type == 'randomF_reg':
        mod = RandomForestRegressor(min_samples_split=30,min_samples_leaf=10,max_depth=70)
    if mode_type == 'xgboost_reg':
        cv_params = {'n_estimators': [100, 200, 400, 800]}
        other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

        mod = xgb.XGBRegressor(**other_params)

    return mod,cv_params,other_params

def para_select(mod,cv_params,other_params,X_train,Y_train):
    from sklearn.model_selection import GridSearchCV
    serchCV = GridSearchCV(estimator=mod, param_grid=cv_params)
    serchCV.fit(X_train, Y_train)
    #print(serchCV.grid_score_)
    print(serchCV.best_params_)
    print(serchCV.best_score_)
    print(serchCV.best_estimator_)
    print(serchCV.best_index_)
    return serchCV

def estimate_mod(mod,X_test,Y_test, feature_names):

    mae = mean_absolute_error(Y_test,mod.predict(X_test))
    print('test score:',mod.score(X_test,Y_test),1 - (mae / Y_test.mean()),mae)
    feature_import = pd.Series(mod.feature_importances_,index=feature_names)
    feature_import = feature_import[feature_import > 0]
    feature_import.sort_values(ascending=False,inplace=True)
    print(feature_import)

def export_tree(mod,feature_names):
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    export_graphviz(mod,out_file=dot_data,feature_names=feature_names,filled=True,rounded=True,
                     special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(r'E:\tree.pdf')

def program():
    data = LoadData()
    X_train, X_test, Y_train, Y_test, feature_names = Do_data(data)
    print(X_train.shape,X_test.shape)
    #mod,cv,para = get_mod(X_train,Y_train,'xgboost_reg')
    #CV = para_select(mod, cv, para, X_train, Y_train)

    mod = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.8, gamma=6,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=5, min_child_weight=4, missing=None, n_estimators=200,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=None,
       subsample=0.8, verbosity=1)
    mod.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_test,Y_test)], verbose=False)
    print("train Score", mod.score(X_train, Y_train))
    estimate_mod(mod,X_test,Y_test, feature_names)

    XX = range(100)
    plt.plot(XX, Y_test[0:100])
    plt.plot(XX, mod.predict(X_test)[0:100])
    plt.legend(['Actual Output','Predict Output'])
    plt.title('Use XGboost to predict single capsule output of OED')
    plt.show()
    return

if __name__ == '__main__':
    program()