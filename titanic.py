# -*- coding: utf-8 -*-

# @File       : titanic.py
# @Date       : 2019-05-09
# @Author     : Jerold
# @Description: for kaggle titanic competition

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection as ms

def loadData(fpath):
    df = pd.read_csv(fpath)
    return df

def program():
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(r'G:\111MyData\titanic\train.csv')

    data['Age'].fillna(data['Age'].mean(),inplace=True)

    X_train,X_test,Y_train,Y_test = ms.train_test_split(data[['Pclass','Age','Fare','Parch','Sex']]
                                                        ,data[['Survived']],test_size=0.25,random_state=33)

    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='records'))
    X_test = vec.fit_transform(X_test.to_dict(orient='records'))

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,Y_train)
    y_hat = dtc.predict(X_test)
    print(dtc.score(X_train,Y_train),dtc.score(X_test,Y_test))

    #print(vec.feature_names_)
    #print(X_hat)
    #plt.scatter(data['Pclass'],data['Survived'])
    #plt.show()

if __name__ == '__main__':
    program()
