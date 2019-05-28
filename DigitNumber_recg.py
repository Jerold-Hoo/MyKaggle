# -*- coding: utf-8 -*-

# @File       : DigitNuber_recg.py
# @Date       : 2019-05-16
# @Author     : Jerold
# @Description:

import pandas as pd
import numpy as np
import tensorflow as tf

def loaddata():

    df1 = pd.read_csv(r"D:\pydata\ditgit_number\train.csv")
    df2 = pd.read_csv(r"D:\pydata\ditgit_number\test.csv")
    return df1,df2

def dense_to_onehot(label):
    label_num = len(np.unique(label))
    res = np.zeros((len(label),label_num)).astype(np.float64)

    for i in range(len(label)):
        res[i,int(label[i])] = 1.0

    return res

def do_data(data):

    X = data.iloc[:, 1:].values.astype(np.float64)
    Y = data.iloc[:, 0].values.astype(np.float64)

    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

    # 处理为0到1之间
    X_train = np.multiply(X_train, 1.0 / 255)
    X_test = np.multiply(X_test, 1.0 / 255)

    # 处理label
    Y_train = dense_to_onehot(Y_train)
    Y_test = dense_to_onehot(Y_test)

    return X_train,X_test,Y_train,Y_test

# 初始化权重
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev = 0.1,dtype=np.float64),dtype=np.float64)

# 初始化权重偏置
def bias_variable(shape):
    return tf.Variable(np.multiply(np.ones(shape,dtype=np.float64),0.1),dtype=np.float64)

# 卷积核函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def train(X_train,Y_train,batch_size = 128,step=100):
    batch_nums = int(len(X_train) / batch_size)

    #图片长宽
    image_size = X_train.shape[1]
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    # 设置x y
    x = tf.placeholder(tf.float64,[None,X_train.shape[1]])
    y = tf.placeholder(tf.float64,[None,Y_train.shape[1]])

    # 准备第一次卷积操作
    x_image = tf.reshape(x,[-1,image_size,image_width,1])

    # 11 计算32个特征，每3*3patch，第一二个参数指的是patch的size,第三个参数是输入的 channelss,第四个参数是输出的channels
    W_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二次卷积,之前生成了64个特征
    W_conv2 = weight_variable([6,6,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # 这一次池化之后变成 7*7

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])  # 展开回来，准备全连接NN

    # 准备全连接
    w_fc1 = weight_variable([7*7*64,1024])  # 1024个神经原
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

    # 进行dropout 操作
    keep_prob = tf.placeholder(tf.float64)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 准备全连接
    w_fc2 = weight_variable([1024,10])  # 1024个神经原
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.relu(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

    prediction = tf.nn.softmax(y_conv)

    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = y_conv))

    # 确定梯度下降方法
    train_step_1 = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

    # 计算准确度
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_conv,1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

    # 设置文件的名参数
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 初始化参数
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # 初始化参数
        sess.run(tf.global_variables_initializer())

        #开始训练
        for i in range(step):
            print('i:', i + 1)
            for batch in range(batch_nums):
                batch_x = X_train[(batch)*batch_size:(batch+1)*batch_size]
                batch_y = Y_train[(batch) * batch_size:(batch + 1) * batch_size]

                sess.run(train_step_1,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})

            # 训练剩下的数据
            batch_x = X_train[(batch) * batch_size:]
            batch_y = Y_train[(batch) * batch_size:]
            sess.run(train_step_1, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        saver.save(sess,r"D:\CNN.ckpt")

def progarm():
    data,RTdata = loaddata()
    X_train, X_test, Y_train, Y_test = do_data(data)
    train(X_train,Y_train,128,100)
    return

if __name__ == "__main__":
    progarm()