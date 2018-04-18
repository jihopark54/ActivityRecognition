import tensorflow as tf
import numpy as np
import csv
from data import Activities, Sensors
from sklearn.utils import shuffle
import os

activity_num = len(Activities)
sensor_num = len(Sensors) +1
sequence_len = 156
drop = 0.5

print(sensor_num)
dataX = []
dataY = []

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
            elif i > 0 and i < 5:
                continue
            else:
                _tmp = [0 for t in range(sensor_num)]

                if row[i] != '':
                    _index = Sensors.index(int(row[i]))
                    _tmp[_index] = 1
                else:
                    _tmp[sensor_num-1] = 1
                tmp.append(_tmp)
        dataX.append(tmp)

dataX, dataY = shuffle(dataX, dataY)

timesteps = sequence_len
hidden_dim = sensor_num * 3
learing_rate = 0.01
iterations = 300

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, sequence_len, sensor_num])
Y = tf.placeholder(tf.float32, [None, activity_num])


# build a LSTM network
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(5)])
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], activity_num, activation_fn=None)  # We use the last cell's output


# cost/loss
#loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
loss =  tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))
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
    if os.path.exists("./model/AR2.meta"):
        saver.restore(sess, "./model/AR2")
    # Training step
    for i in range(iterations):
        trainX, trainY = shuffle(trainX, trainY)
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))
    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse))
    save_path = saver.save(sess, "./model/AR2")

