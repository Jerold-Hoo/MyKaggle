# -*- coding: utf-8 -*-

# @File       : titanic.py
# @Date       : 2019-05-09
# @Author     : Jerold
# @Description: for kaggle titanic competition

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection as ms
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import pydotplus

def predo():
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(r'D:\pydata\titanic\train.csv')
    realTX = pd.read_csv(r'D:\pydata\titanic\test.csv')
    realTY = pd.read_csv(r'D:\pydata\titanic\gender_submission.csv')
    Xname = ['Pclass','Age','Sex']
    Yname = 'Survived'

    print('OriCount',len(data))
    data.dropna(inplace=True, subset=Xname+[Yname])
    print('clareCount', len(data))
    X = data[Xname]
    Y = data[Yname]
    # 切分数据
    X_train,X_test,Y_train,Y_test = ms.train_test_split(X,Y,test_size=0.25,random_state=33)
    realTX.fillna({'Fare':realTX['Fare'].median(),'Age':realTX['Age'].median()},inplace=True)
    # 特征处理
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='records'))
    X_test = vec.fit_transform(X_test.to_dict(orient='records'))
    RTX = vec.fit_transform(realTX[Xname].to_dict(orient='records'))
    feature_names = vec.feature_names_

    return X_train,X_test,Y_train,Y_test,feature_names,RTX,realTY[Yname]

def export_tree(tree,feature_names):
    from sklearn.externals.six import StringIO
    import os
    os.environ["PATH"] += os.pathsep + r'D:/Program Files(x86)/Graphviz2.38/bin/'

    dot_data = StringIO()
    export_graphviz(tree,out_file=dot_data,feature_names=feature_names,filled=True,rounded=True,
                     class_names=['s','Ns'],special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(r'D:\tree.pdf')

def paremeter_select(X_train,X_test,Y_train,Y_test):
    from sklearn.model_selection import GridSearchCV
    mod = RandomForestClassifier(min_samples_split=30,min_samples_leaf=10)
    parameters = {'n_estimators':(5,10,15,20,40,100)}
    GSCV = GridSearchCV(mod,parameters)
    GSCV.fit(X_train,Y_train)
    print(GSCV.best_estimator_)

def estimate_model(model,X_test,Y_test,feature_names):
    import sklearn.metrics as metr
    Y_pre = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    fpr,tpr,thresholds = metr.roc_curve(Y_test, y_score[:,1])
    auc_s = metr.auc(fpr,tpr)
    accuracy_s = metr.accuracy_score(Y_test,Y_pre)
    precision_s = metr.precision_score(Y_test, Y_pre)
    recall_s = metr.recall_score(Y_test, Y_pre)
    print('auc:',auc_s,'precision:',precision_s,'recall:',recall_s)
    print(feature_names)
    print(model.feature_importances_)
    return

def fit_mod(X_train,X_test,Y_train,Y_test,mod_type):
    if mod_type == 'tree':
        mod = DecisionTreeClassifier(max_depth=4,min_samples_split=30,min_samples_leaf=10)
    elif mod_type == 'randomF':
        mod = RandomForestClassifier(n_estimators=20, min_samples_split=30,min_samples_leaf=10)
    elif mod_type == 'adaboost':
        mod = AdaBoostClassifier()
    elif mod_type == 'GBDT':
        mod = GradientBoostingClassifier()

    mod.fit(X_train,Y_train)
    print('train score:',mod.score(X_train,Y_train),mod.score(X_test,Y_test))
    return mod

def program():
    X_train, X_test, Y_train, Y_test, feature_names, RTX, RTY = predo()
    #paremeter_select(X_train, X_test, Y_train, Y_test)
    mod = fit_mod(X_train, X_test, Y_train, Y_test, 'GBDT')
    #paremeter_select(X_train, X_test, Y_train, Y_test)
    #export_tree(mod,feature_names)
    estimate_model(mod, X_test, Y_test, feature_names)
    estimate_model(mod, RTX, RTY, feature_names)
    return

if __name__ == '__main__':
    #test_adb()
    program()
