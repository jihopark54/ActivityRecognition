import tensorflow as tf
import numpy as np
import csv
from data import Activities, Sensors
from sklearn.utils import shuffle
import os

activity_num = len(Activities)
sensor_num = len(Sensors) +1
#sensor_num = len(Sensors)

sequence_len = 156
drop = 0.5

print(activity_num)
dataX = []
dataY = []
seq_list = []



## RNN input sturcture
with open('./data.csv', 'r') as csvfile:
    #reader = csv.DictReader(csvfile)
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        n_row = len(row)
        tmp = []
        count = 0
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
                    count += 1
                else:
                    _tmp[sensor_num - 1] = 1
                tmp.append(_tmp)
        dataX.append(tmp)
        seq_list.append(count)

"""
## ANN input structure
with open('./data.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        n_row = len(row)
        tmp = [0 for _ in range(sensor_num)]
        count =  0
        for i in range(n_row):
            if i == 0:
                _index = Activities.index(row[i])
                _tmp = [0 for t in range(activity_num)]
                _tmp[_index] = 1
                dataY.append(_tmp)
            elif i > 0 and i < 5:
                continue
            else:
                if row[i] != '':
                    _index = Sensors.index(int(row[i]))
                    tmp[_index] = 1
        dataX.append(tmp)
        seq_list.append(count)
"""

timesteps = sequence_len
hidden_dim = sensor_num
learing_rate = 0.01
iterations = 1000
n_hidden_1 = 256
n_hidden_2 = 256

# train/test split

# input place holders
X = tf.placeholder(tf.float32, [None, sequence_len, sensor_num]) #RNN input
#X = tf.placeholder(tf.float32, [None,  sensor_num]) #ANN input
Y = tf.placeholder(tf.float32, [None, activity_num])
L = tf.placeholder(tf.int32, [None])

# build a LSTM network
"""
X = tf.slice(X, [0], [50])
batched_X = tf.train.batch(tensors=[X], batch_size=50, dynamic_pad=True)
"""


## Bi-directional RNN1
######################################################################################################################
#cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(3)])
#cell2 = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(3)])

cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.tanh) for _ in range(3)])
cell2 = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.tanh) for _ in range(3)])

outputs, _states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell2, dtype=tf.float32, sequence_length=L, inputs=X)
print(outputs)

# With residual
n_X1 = tf.add(outputs[1], X)

# With concat
#n_X1 = tf.concat([outputs[1], X], axis=1)

# Witout concat
#n_X1 = outputs[1]

#cell2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh) for _ in range(3)])


cell2 = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(num_units=hidden_dim, activation=tf.tanh) for _ in range(3)])

outputs2, _states2 = tf.nn.dynamic_rnn(cell2, dtype=tf.float32, sequence_length=L + 33, inputs=n_X1, scope="rnn2")


## Fully-connected Layer
Y_pred2 = tf.contrib.layers.fully_connected(outputs2[:, -1], num_outputs= activity_num, activation_fn=None)  # We use the last cell's output
########################################################################################################################

"""
## LSTM
#########################################################################################################################
cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell1, dtype=tf.float32, sequence_length=L, inputs=X, scope="rnn1")

n_X1 = outputs

#n_X1 = tf.contrib.layers.fully_connected(n_X1, num_outputs=hidden_dim, activation_fn=tf.nn.relu)

cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs2, _state2 = tf.nn.dynamic_rnn(cell2, dtype=tf.float32, sequence_length=L, inputs=n_X1, scope="rnn2")

Y_pred2 = tf.contrib.layers.fully_connected(outputs2[:, -1], num_outputs=activity_num, activation_fn=None)
#########################################################################################################################
"""

"""
## GRU
#########################################################################################################################
cell1 = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell1, dtype=tf.float32, sequence_length=L, inputs=X, scope="rnn1")
#Y_pred2 = outputs
n_X1 = outputs

cell2 = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.tanh)
outputs2, _state2 = tf.nn.dynamic_rnn(cell2, dtype=tf.float32, sequence_length=L, inputs=n_X1, scope="rnn2")


Y_pred2 = tf.contrib.layers.fully_connected(outputs2[:, -1], num_outputs=activity_num, activation_fn=None)
#########################################################################################################################
"""

"""
## ANN 
#########################################################################################################################
def multilayer_perceptron(X, weights, biases):
    cell1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    cell1 = tf.nn.relu(cell1) 

    #Hidden layer with RELU activiation

    cell2 = tf.add(tf.matmul(cell1, weights['h2']), biases['b2'])
    cell2 = tf.nn.relu(cell2)

    #Output layer with linear activation
    out_layer = tf.matmul(cell2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([sensor_num, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, activity_num]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([activity_num]))
}

Y_pred2 = multilayer_perceptron(X, weights, biases)
###########################################################################################################################
"""
# cost/loss
#loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
loss =  tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred2))
# optimizer
optimizer = tf.train.AdamOptimizer(learing_rate)
#optimizer = tf.train.GradientDescentOptimizer(learing_rate)
train = optimizer.minimize(loss)

# RMSE
#targets = tf.placeholder(tf.float32, [None, activity_num])
#predictions = tf.placeholder(tf.float32, [None, activity_num])
#rmse = tf.sqrt(tf.reduce_mean(tf.square(Y - Y_pred2)))

correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(Y_pred2,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()



if __name__ == "__main__":
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(100):
            dataX, dataY, seq_list = shuffle(dataX, dataY, seq_list)
            train_size = int(len(dataY) * 0.1)
            test_size = len(dataY) - train_size
            trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
            trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])
            trainL, testL = np.array(seq_list[0:train_size]), np.array(seq_list[train_size:len(seq_list)])
            sess.run(train, {X: trainX, Y: trainY, L: trainL })

            cost = sess.run(accuracy, {X: testX, Y: testY, L: testL } )
            print(cost) 
