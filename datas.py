# -*- coding: utf-8 -*-

# @File       : datas.py
# @Date       : 2019-05-31
# @Author     : Jerold
# @Description:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r'G:\111MyData\test.csv')
data= data.iloc[:,2:10].values  #取第3-10列

def get_train_data(batch_size=60,time_step=4,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化

    normalized_train_data = data_train
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):

        if i % batch_size == 0:
            batch_index.append(i)
        x=normalized_train_data[i:i+time_step,:7]
        y=normalized_train_data[i:i+time_step,7,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

batch_index,train_x,train_y = get_train_data()

def get_test_data(time_step=4,test_begin=0):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=data_test
    #normalized_test_data = (data_test - mean) / std  # 标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y

mean,std,test_x,test_y = get_test_data()

print(np.shape(train_x),np.shape(train_y),np.shape(test_x),np.shape(test_y))

for i in range(20):
    print(train_x[i])
    print(train_y[i])
    print('!!!')
    print(test_x[i])
    print(test_y[i])
    print('------------------')


