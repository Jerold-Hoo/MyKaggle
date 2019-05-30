# -*- coding: utf-8 -*-

# @File       : WYNYtest.py
# @Date       : 2019-05-28
# @Author     : Jerold
# @Description: for the test of WYNY

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,roc_curve,auc,precision_score,recall_score,accuracy_score
from datetime import datetime
import tensorflow as tf

MY_PATH = r"G:\111MyData\test_data201803.csv"

def check_data(data):
    train = data[data['time'] < datetime(2006,12,1)]
    to_predict = data[data['time'] >= datetime(2006,12,1)]

    data[['KWH']].boxplot()
    plt.show()

    train_zero = train[train['KWH'] == 0]
    train_Nonezero = train[train['KWH'] != 0]

    #zero_groupby_month = train_zero[['KWH','month']].groupby(by='month')

    #nonez_groupby_month.boxplot()
    #print(len(train_Nonezero),len(train_Nonezero[train_Nonezero['KWH']<2100]))
    #train_Nonezero.boxplot(column='KWH')
    #plt.show()

    #print(nonez_groupby_month.describe())
    #print(zero_groupby_month.count())

    #print(len(train_zero))
    #print(len(train))

def SearchCV(X_train, X_test, Y_train, Y_test):

    # 1 确定 n_estimators
    cv_params = {'n_estimators': [2000, 3000]}
    other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    mod = xgb.XGBRegressor(**other_params)
    g_serch = GridSearchCV(estimator=mod,param_grid=cv_params)
    g_serch.fit(X_train,Y_train)

    #mod = xgb.XGBRegressor(**{**other_params,**g_serch.best_params_})
    #mod.fit(X_train,Y_train)
    print(g_serch.best_score_)
    print(g_serch.best_params_)
    print(g_serch.best_estimator_.score(X_train,Y_train))
    print(g_serch.best_estimator_.score(X_test, Y_test))
    print(g_serch.best_estimator_.feature_importances_)
    return

def use_xgboost_judge_zeros(X_train, X_test, Y_train, Y_test):
    params = {'booster': 'gbtree',
              'n_estimators': 500,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 1,
              'eta': 0.025,
              'seed': 0 }

    mod = xgb.XGBClassifier(**params)
    mod.fit(X_train,Y_train)

    Y_pre = mod.predict(X_test)
    y_score = mod.predict_proba(X_test)
    fpr,tpr,thresholds = roc_curve(Y_test, y_score[:,1])
    auc_s = auc(fpr,tpr)
    accuracy_s = accuracy_score(Y_test,Y_pre)
    precision_s = precision_score(Y_test, Y_pre)
    recall_s = recall_score(Y_test, Y_pre)
    print('auc:',auc_s,'accuracy:',accuracy_s,'precision:',precision_s,'recall:',recall_s)
    print(pd.Series(mod.feature_importances_, index=X_train.columns).sort_values(ascending=False))
    return mod

def use_xgboost(X_train, X_test, Y_train, Y_test):

    params = {'n_estimators':500,
              'learning_rate': 0.1,
              'max_depth': 5,
              'min_child_weight': 1,
              'lambda': 10,
              'seed': 0,
              'subsample': 0.6,
              'colsample_bytree': 0.8,
              'gamma': 0.1,
              'reg_alpha': 0,
              'reg_lambda': 1}

    mod = xgb.XGBRegressor(**params)
    mod.fit(X_train,Y_train)
    print(mod.score(X_train,Y_train))
    print(mod.score(X_test, Y_test))
    print(pd.Series(mod.feature_importances_,index=X_train.columns).sort_values(ascending=False))
    return mod

# 预处理数据
def deal_data_forXgBoost(data,fronts=3):

    data.loc[:,'time'] = pd.to_datetime(data['time'])
    #data['month'] = [i.month for i in data['time']]
    #data['day'] = [i.day for i in data['time']]
    data['weekday'] = [i.dayofweek for i in data['time']]
    data['hour'] = [i.hour for i in data['time']]

    # 增加前序时间作为特征 fronts 参数确定增加多少个前序时间
    n_data = len(data)
    for i in range(1,fronts+1,1):
        front = data['KWH'][0:n_data-i]
        front.index = front.index + i
        new_c = pd.Series([0] * i).append(front)
        data['front'+str(i)] = new_c

    # 确定 X的名称和Y的名称
    x_names = data.columns.tolist().copy()
    x_names.remove('time')
    x_names.remove('KWH')
    y_name = 'KWH'

    # 处理用于 二分类是否为异常的数据集
    data_classf_z = data.copy()
    data_classf_z.loc[:, 'KWH'] = data_classf_z.loc[:,'KWH'].where(data_classf_z['KWH']>2100,1)
    data_classf_z.loc[:, 'KWH'] = data_classf_z.loc[:, 'KWH'].where(data_classf_z['KWH'] == 1,0)
    train_b = data_classf_z[data_classf_z['time'] < datetime(2006,12,1)]
    to_predict_b = data_classf_z[data_classf_z['time'] >= datetime(2006,12,1)]

    X_train_b, X_test_b, Y_train_b, Y_test_b = train_test_split(train_b[x_names], train_b[y_name], test_size=0.25, random_state=33)

    # 处理用于回归的数据
    # 小于 2000的异常数据，使用前一个非异常数据填充
    data.iloc[:, 4:] = data.iloc[:,4:].where(data.iloc[:,4:]>2100,pd.NaT)
    data = data.fillna(method='pad')
    data = data[16:]
    data.iloc[:, 4:] = data.iloc[:,4:].astype('float64')

    train = data[data['time'] < datetime(2006,12,1)]
    to_predict = data[data['time'] >= datetime(2006,12,1)]

    #剔除异常值
    train = train[train['KWH'] > 2100]
    X_train, X_test, Y_train, Y_test = train_test_split(train[x_names], train[y_name], test_size=0.25, random_state=33)

    return X_train, X_test, Y_train, Y_test, to_predict[x_names], to_predict[y_name],X_train_b, X_test_b, Y_train_b, Y_test_b, to_predict_b[x_names], to_predict_b[y_name]

# 评价二分类模型
def evaluate_mod_binary(mod,X_to_predict_b,Y_to_predict_b):
    """
    # 评价二分类（判断是否是否会出现异常）模型的数据
    :param mod: 模型实例
    :param X_to_predict_b: 需要预测数据的特征集合X
    :param Y_to_predict_b: 需要预测数据的真实结果 Y
    :return: 预测结果 Y_hat
    """
    Y_pre = mod.predict(X_to_predict_b)
    y_score = mod.predict_proba(X_to_predict_b)
    fpr,tpr,thresholds = roc_curve(Y_to_predict_b, y_score[:,1])
    auc_s = auc(fpr,tpr)
    accuracy_s = accuracy_score(Y_to_predict_b,Y_pre)
    precision_s = precision_score(Y_to_predict_b, Y_pre)
    recall_s = recall_score(Y_to_predict_b, Y_pre)
    print('auc:',auc_s,'accuracy:',accuracy_s,'precision:',precision_s,'recall:',recall_s)
    #show = pd.DataFrame(Y_to_predict_b)
    #show['predict'] = y_score[:,1]
    #print(show[show['KWH']==1])
    return Y_pre

# 评价回归模型
def evaluate_mod_reg(mod,X_to_predict,Y_to_predict):
    """
    评价回归模型的效果
    :param mod: 模型实例
    :param X_to_predict: 需要预测的X数据
    :param Y_to_predict: 需要预测的Y数据
    :return: 返回预测数据 Y_hat
    """
    predict_Y = mod.predict(X_to_predict)
    print(mean_absolute_error(Y_to_predict.values,predict_Y))
    return predict_Y

def program():
    """
    主运行函数
    """

    data = pd.read_csv(MY_PATH,index_col=0)

    X_train, X_test, Y_train, Y_test, X_to_predict, Y_to_predict,X_train_b, X_test_b, Y_train_b, Y_test_b, X_to_predict_b,Y_to_predict_b = deal_data_forXgBoost(data,10)

    mod_c = use_xgboost_judge_zeros(X_train_b, X_test_b, Y_train_b, Y_test_b)
    predict_y_b = evaluate_mod_binary(mod_c,X_to_predict_b,Y_to_predict_b)

    print("---------------------------------------------------------")
    mod = use_xgboost(X_train, X_test, Y_train, Y_test)
    predict_y = evaluate_mod_reg(mod,X_to_predict,Y_to_predict)
    print('res:')
    predict_y[predict_y_b == 1] = 0
    print(mean_absolute_error(Y_to_predict,predict_y))
    plt.figure()
    plt.plot(Y_to_predict.index, Y_to_predict)
    plt.plot(Y_to_predict.index, predict_y)
    plt.legend(['KWH','predict_KWH'])
    plt.title("Use XGboost to predict 'KWH' data in 2006.12")
    plt.show()

    #X_to_predict = X_to_predict[Y_to_predict > 2000]
    #Y_to_predict = Y_to_predict[Y_to_predict > 2000]

def deal_data_forLsTM(data,batch_size,time_step):
    data.loc[:,'time'] = pd.to_datetime(data['time'])
    data['month'] = [i.month for i in data['time']]
    data['day'] = [i.day for i in data['time']]
    data['weekday'] = [i.dayofweek for i in data['time']]
    data['hour'] = [i.hour for i in data['time']]

    # 只提取特征和均值
    #data = data[:,2:].values
    data_train = data[data['time'] < datetime(2006,11,1)]
    data_train = data_train.iloc[:,1:].values

    Y_index = 0 # label 列的列序号
    #print(data_train)

    # 标准化
    # normalized_train_data = data_train
    normalized_train_data = (data_train - np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    train_x, train_y = [], []
    batch_index = []
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,Y_index+1:]
       y=normalized_train_data[i:i+time_step,Y_index,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))

    data_test = data[(data['time'] >= datetime(2006,11,1)) & (data['time'] < datetime(2006,12,1))]

    data_test = data_test.iloc[:,1:].values
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    print(normalized_test_data)
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, Y_index+1:]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, Y_index]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, Y_index+1:]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, Y_index]).tolist())

    return batch_index,train_x,train_y,mean, std, test_x, test_y

# 定义网络结构
rnn_unit = 10  # 隐藏层 单元个数
input_size = 7
output_size = 1
lr = 0.0001 # 学习率

def use_LSTM(X):

    # 定义权重
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    }
    # 定义偏置
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]

    w_in = weights['in']
    b_in = biases['in']


    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入

    # cell 的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)

    #状态的初始化
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

def train_lstm(batch_index,train_x,train_y,time_step=15):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    # 训练样本中第2001 - 5785个样本，每次取15个

    #batch_index,train_x,train_y= get_train_data(batch_size,time_step,train_begin,train_end)

    print(np.array(train_x).shape)# 3785  15  7
    print(batch_index)
    #相当于总共3785句话，每句话15个字，每个字7个特征（embadding）,对于这些样本每次训练80句话

    pred,_= use_LSTM(X)

    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练200次
        for i in range(200):
            #每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==0:
                print("保存模型：",saver.save(sess,'model/stock2.model',global_step=i))


def prediction(mean, std, test_x, test_y,time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    #mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = use_LSTM(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


def LSTM_program():
    data = pd.read_csv(MY_PATH, index_col=0)
    patch_size = 80
    time_step = 15
    batch_index, train_x, train_y, mean, std, test_x, test_y = deal_data_forLsTM(data, patch_size, time_step)
    train_lstm(batch_index, train_x, train_y,time_step)
    prediction(mean, std, test_x, test_y,time_step)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns",None)
    #pd.set_option("display.max_rows", None)

    #program()
    LSTM_program()
