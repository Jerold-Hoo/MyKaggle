# -*- coding: utf-8 -*-

# @File       : titanic.py
# @Date       : 2019-05-09
# @Author     : Jerold
# @Description: for kaggle titanic competition

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection as ms
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pydotplus

def loadData(fpath):
    df = pd.read_csv(fpath)
    return df

def program():
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(r'G:\111MyData\titanic\train.csv')

    data['Age'].fillna(data['Age'].mean(),inplace=True)

    X_train,X_test,Y_train,Y_test = ms.train_test_split(data[['Pclass','Age','Fare','Parch','Sex']]
                                                        ,data['Survived'],test_size=0.25,random_state=33)

    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='records'))
    X_test = vec.fit_transform(X_test.to_dict(orient='records'))


    #dtc = DecisionTreeClassifier(max_depth=3,min_samples_split=50,min_samples_leaf=10)
    dtc = RandomForestClassifier(n_estimators=100,min_samples_split=40)
    dtc.fit(X_train,Y_train)
    y_hat = dtc.predict(X_test)
    print(dtc.score(X_train,Y_train),dtc.score(X_test,Y_test))

    return
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    export_graphviz(dtc,out_file=dot_data,feature_names=vec.feature_names_,filled=True,rounded=True,
                     class_names=['s','Ns'],special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(r'E:\tree.pdf')

    #print(vec.feature_names_)
    #print(X_hat)
    #plt.scatter(data['Pclass'],data['Survived'])
    #plt.show()

if __name__ == '__main__':
    program()
