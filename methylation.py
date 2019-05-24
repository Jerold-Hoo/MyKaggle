# -*- coding: utf-8 -*-

# @File       : methylation.py
# @Date       : 2019-05-14
# @Author     : Jerold
# @Description:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
import math

def trans_btoM(x):
        if x == 0:
            return 0
        else:
            return math.log(x/(1-x),2)

def ues_SkLearn():

    data = pd.read_csv(r'C:\PythonProj\Mykaggle\data\train\geneMpre.txt',sep='\t')
    data.set_index('ID_REF',inplace=True)
    data = pd.DataFrame(data.values.T,index=data.columns,columns=data.index)

    print(data.applymap(trans_btoM))
    return
    label = pd.read_csv(r'C:\PythonProj\Mykaggle\data\train\methyDesign.txt',sep='\t',names=['ID_REF','Label'])
    label.set_index('ID_REF',inplace=True)
    for index,n in data.count()[data.count() < 118].iteritems():
        data[index].fillna(data[index].median(),inplace=True)

    test1X = pd.read_csv(r'C:\PythonProj\Mykaggle\data\test2\geneMpre.txt', sep='\t')
    test1X.set_index('ID_REF', inplace=True)
    test1X = pd.DataFrame(test1X.values.T,index=test1X.columns,columns=test1X.index)
    test1Y = pd.read_csv(r'C:\PythonProj\Mykaggle\data\test2\methyDesign.txt', sep='\t', names=['ID_REF', 'Label'])

    for index,n in test1X.count()[test1X.count() < 118].iteritems():
        test1X[index].fillna(test1X[index].median(),inplace=True)

    lr = LogisticRegression(solver='liblinear')
    lr.fit(data,label['Label'])

    W = pd.DataFrame(lr.coef_.T,index=data.columns,columns=['W'])
    W['absW'] = abs(W['W'])
    print(W.sort_values(by=['absW'],ascending=False))
    #data['W'] = lr.coef_[0]

    return
    print(lr.score(test1X,test1Y['Label']))
    test1Y['predict'] = lr.predict(test1X)
    print((test1Y['Label'] != test1Y['predict']).count())
    print(test1Y)

if __name__ == '__main__':
    ues_SkLearn()