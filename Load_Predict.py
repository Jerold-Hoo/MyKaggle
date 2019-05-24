# -*- coding: utf-8 -*-

# @File       : Load_Predict.py
# @Date       : 2019-05-17
# @Author     : Jerold
# @Description:

from numpy import *
import pandas as pd
import pymssql
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot

SQL = "select ID,optTime,PeriodValue,MeterID,HourScale from Bs_MeterData where meterID = 951 order by optTime"

def test(alldata):
    data = alldata[alldata['optTime'] < datetime(2018,2,10)]
    plt.plot(data['optTime'],data['PeriodValue'])
    plt.show()
    return

# 最近N天加权平均
def W_mean_lastN(alldata,day,lastN):
    W = range(lastN+1, 1, -1)
    W = W / sum(W)
    res = zeros((1,48))[0]
    for i in range(lastN):
        data = alldata[(alldata['optTime'] >= day - timedelta(days=i+1)) & (alldata['optTime'] < day - timedelta(days=i))]
        res += data['PeriodValue'].values * W[i]

    return res

# ARIMA
def use_ARIMA(alldata,day):
    series = alldata[alldata['optTime'] < day][['optTime', 'PeriodValue']]
    series.set_index('optTime', inplace=True)
    real = alldata[(alldata['optTime'] >= day) & (alldata['optTime'] < day + timedelta(days=1))]

    #autocorrelation_plot(series)
    #plt.show()
    model = ARIMA(series, order=(20, 1, 0))
    model_fit = model.fit(disp=0)
    a,b,c = model_fit.forecast(48)
    plt.plot(range(48), a)
    plt.plot(range(48), b)
    plt.plot(range(48), c.T[1])
    plt.plot(range(48), real['PeriodValue'])
    plt.legend(['a','b','c','real'])
    plt.show()
    #print(c)
    #print(model_fit.summary())

def parser(x):
    return datetime.strptime(x,'%Y-%m %H:%M')

if __name__ == "__main__":
    conn = pymssql.connect(server='118.123.7.142', user='sa', password='2019newpass.7142sql', database='Gird_EN')
    alldata = pd.read_sql(SQL,conn)
    day = datetime(2018, 2, 3)
    use_ARIMA(alldata,day)
    """
    real_value = alldata[(alldata['optTime'] >= day) & (alldata['optTime'] < day + timedelta(days=1))]

    for i in range(2,100,2):
        predict_value = W_mean_lastN(alldata,day,i)
        print(i,':',abs(real_value['PeriodValue'].values - predict_value).sum() / real_value['PeriodValue'].sum())
    """