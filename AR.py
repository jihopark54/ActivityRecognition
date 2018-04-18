import tensorflow as tf
import numpy as np
import csv
from data import Activities, Sensors
from sklearn.utils import shuffle
import os

activity_num = len(Activities)
median_num = activity_num * 2 + 4
sensor_num = len(Sensors) +1
sequence_len = 156
drop = 1

print(sensor_num)
dataX = []
dataY = []
timeX = [] #오전 오후, 시간, 분,

with open('./data.csv', 'r') as csvfile:
    #reader = csv.DictReader(csvfile)
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        n_row = len(row)
        tmp = []
        for i in range(n_row):
            if i == 0:
                _index = Activities.index(row[i])
                _tmp = [0 for t in range(activity_num)]
                _tmp[_index] = 1
                dataY.append(_tmp)
            elif i == 1 or i == 3 or i == 4:
                continue
            elif i == 2:
                _tmp = [0 for _ in range(3)]
                noon = int(row[i])//43200
                time = (int(row[i])%43200)//3600
                minite = (int(row[i])%3600)//60
                sec = int(row[i])%60
                timeX.append([noon,time,minite,sec])
            else:
                _tmp = [0 for t in range(sensor_num)]

                if row[i] != '':
                    _index = Sensors.index(int(row[i]))
                    _tmp[_index] = 1
                else:
                    _tmp[sensor_num-1] = 1
                tmp.append(_tmp)
        dataX.append(tmp)

dataX, dataY, timeX = shuffle(dataX, dataY, timeX)

timesteps = sequence_len
hidden_dim = sensor_num * 3
learing_rate = 0.01
iterations = 100

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])
trainT, testT = np.array(timeX[0:train_size]), np.array(timeX[train_size:len(timeX)])
# input place holders
X = tf.placeholder(tf.float32, [None, sequence_len, sensor_num])
X_time = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, activity_num])
keep_prob = tf.placeholder(tf.float32)


wd1 = tf.Variable(tf.random_normal([median_num, median_num]))
#wdout = tf.Variable(tf.random_normal([activity_num + 4, activity_num + 4]))
wdout = tf.Variable(tf.random_normal([median_num, activity_num]))

bd1 = tf.Variable(tf.random_normal([median_num]))
bdout = tf.Variable(tf.random_normal([activity_num]))


# build a LSTM network
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(5)])
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], activity_num * 2, activation_fn=None)  # We use the last cell's output

median = tf.concat([Y_pred, X_time], axis=1)

dense1 = tf.nn.relu(tf.add(tf.matmul(median,wd1),bd1))
dense1 = tf.nn.dropout(dense1, keep_prob)
dense2 = tf.nn.relu(tf.add(tf.matmul(dense1,wdout),bdout))

# cost/loss
#loss = tf.reduce_sum(tf.square(dense2 - Y))  # sum of the squares
loss =  tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=dense2))
# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, activity_num])
predictions = tf.placeholder(tf.float32, [None, activity_num])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if os.path.exists("./model/AR.meta"):
        saver.restore(sess, "./model/AR")
    # Training step
    for i in range(iterations):
        trainX, trainY, trainT = shuffle(trainX, trainY, trainT)
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY, X_time:trainT, keep_prob:drop})
        print("[step: {}] loss: {}".format(i, step_loss))
    # Test step
    test_predict = sess.run(dense2, feed_dict={X: testX, X_time:testT, keep_prob:1.})
    rmse = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse))
    save_path = saver.save(sess, "./model/AR")

