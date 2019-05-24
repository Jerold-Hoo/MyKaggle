# -*- coding: utf-8 -*-

# @File       : OEDtest.py
# @Date       : 2019-05-16
# @Author     : Jerold
# @Description:

import pymssql
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
import pydotplus


def testCRAT():
    conn = pymssql.connect(server='118.123.7.142',user='sa',password='2019newpass.7142sql',database='OEDInUse')
    SQL = "select Temp8,Temp25,Temp41,Rspeed400,Rspeed580,DownSw,Dateinterval,[OutPut] from Bs_SinCapsule where OilCode like 'RT%'"
    data = pd.read_sql(SQL,conn)
    data.loc[data['Dateinterval'].isna(),'Dateinterval'] = data['Dateinterval'].median()
    data.dropna(inplace=True)

    X_train,X_test,Y_train,Y_test = train_test_split(data[['Temp8','Temp25','Temp41','Rspeed400','Rspeed580','DownSw','Dateinterval']]
                                                     ,data['OutPut'],test_size=0.2,random_state=1)

    dct = DecisionTreeRegressor(min_samples_split=20,max_depth=8)
    dct.fit(X_train,Y_train)
    print(dct.score(X_train,Y_train),dct.score(X_test,Y_test))

    ada = AdaBoostRegressor(n_estimators=100)
    ada.fit(X_train,Y_train)
    print(ada.score(X_train,Y_train),ada.score(X_test,Y_test))

    gda = GradientBoostingRegressor(n_estimators=100)
    gda.fit(X_train,Y_train)
    print(gda.score(X_train,Y_train),gda.score(X_test,Y_test))

    return
    from sklearn.externals.six import StringIO
    dot_data = StringIO()
    export_graphviz(dct,out_file=dot_data,feature_names=['Temp8','Temp25','Temp41','Rspeed400','Rspeed580','DownSw','Dateinterval'],filled=True,rounded=True,
                     special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(r'E:\tree.pdf')


if __name__ == '__main__':
    testCRAT()