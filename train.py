'''
Created on 2017. 9. 21.

@author: em
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os
from prettytensor import xavier_init
import math

#tf.set_random_seed()  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

training_epochs = 23
learningRate = 0.005
batchSize = 1000

inputDim = 4
RNNLayers = 1
seq_length = 10
hiddenDim1 = 50
hiddenDim2 = 200
hiddenDim3 = 500
hiddenDim4 = 300
hiddenDim5 = 200
num_classes = 130

CHECK_POINT_DIR = TB_SUMMARY_DIR = "./log/L1_10-50-200-500-300-200-130"

xy = np.loadtxt('./emg/20171018-MovSD-12.txt', delimiter=' ')

x = xy[50:, :-1]
y = xy[50:, [-1]]

# x = xy[:, :-1]
# y = xy[:, [-1]]

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

trainSize = int(len(dataY) * 0.99)
testSize = len(dataY) - trainSize
trainX, testX = np.array(dataX[0:trainSize]), np.array(dataX[trainSize:len(dataX)])
trainY, testY = np.array(dataY[0:trainSize]), np.array(dataY[trainSize:len(dataY)])

X = tf.placeholder(tf.float32, shape=[None, seq_length, inputDim], name="EMGs")
Y = tf.placeholder(tf.int32, shape=[None, 1], name="Angle")
Y_one_hot = tf.one_hot(Y, num_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, num_classes])
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('LSTM_layers'):

    cell = tf.nn.rnn_cell.LSTMCell(num_units=inputDim, state_is_tuple=True, activation=tf.tanh)
    cells = tf.nn.rnn_cell.MultiRNNCell([cell] * RNNLayers, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
    X_FC = tf.contrib.layers.fully_connected(outputs[:,-1], hiddenDim1, activation_fn=None)

    tf.summary.histogram("X_FC", X_FC)

with tf.variable_scope('Fully_connected_layer1'):
    W1_FC = tf.get_variable("W1_FC", [hiddenDim1, hiddenDim2], tf.float32, initializer=xavier_init(hiddenDim1, hiddenDim2))
    B1_FC = tf.get_variable("B1_FC", [hiddenDim2], tf.float32, initializer=xavier_init(1, hiddenDim2))
    H1_FC = tf.nn.relu( tf.matmul(X_FC, W1_FC)+ B1_FC, name="H1_FC" )
    H1_FC = tf.nn.dropout(H1_FC, keep_prob=keep_prob)

    tf.summary.histogram("W1_FC", W1_FC)
    tf.summary.histogram("B1_FC", B1_FC)
    tf.summary.histogram("H1_FC", H1_FC)

with tf.variable_scope('Fully_connected_layer2'):
    W2_FC = tf.get_variable("W2_FC", [hiddenDim2, hiddenDim3], tf.float32,
                            initializer=xavier_init(hiddenDim2, hiddenDim3))
    B2_FC = tf.get_variable("B2_FC", [hiddenDim3], tf.float32,
                            initializer=xavier_init(1, hiddenDim3))
    H2_FC = tf.nn.relu(tf.matmul(H1_FC, W2_FC) + B2_FC, name="H1_FC")
    H2_FC = tf.nn.dropout(H2_FC, keep_prob=keep_prob)
    # Y_pred = tf.matmul(H1_FC, W2_FC) + B2_FC

    tf.summary.histogram("W2_FC", W2_FC)
    tf.summary.histogram("B2_FC", B2_FC)
    tf.summary.histogram("H2_FC", H2_FC)
    # tf.summary.histogram("hypothis", Y_pred)

with tf.variable_scope('Fully_connected_layer3'):
    W3_FC = tf.get_variable("W3_FC", [hiddenDim3, hiddenDim4], tf.float32,
                            initializer=xavier_init(hiddenDim3, hiddenDim4))
    B3_FC = tf.get_variable("B3_FC", [hiddenDim4], tf.float32,
                            initializer=xavier_init(1, hiddenDim4))
    H3_FC = tf.nn.relu(tf.matmul(H2_FC, W3_FC) + B3_FC, name="H3_FC")
    H3_FC = tf.nn.dropout(H3_FC, keep_prob=keep_prob)
    # Y_pred = tf.matmul(H2_FC, W3_FC) + B3_FC


    tf.summary.histogram("W3_FC", W3_FC)
    tf.summary.histogram("B3_FC", B3_FC)
    tf.summary.histogram("H3_FC", H3_FC)

with tf.variable_scope('Fully_connected_layer4'):
    W4_FC = tf.get_variable("W4_FC", [hiddenDim4, hiddenDim5], tf.float32,
                            initializer=xavier_init(hiddenDim4, hiddenDim5))
    B4_FC = tf.get_variable("B4_FC", [hiddenDim5], tf.float32,
                            initializer=xavier_init(num_classes, hiddenDim5))
    H4_FC = tf.nn.relu(tf.matmul(H3_FC, W4_FC) + B4_FC, name="H4_FC")
    H4_FC = tf.nn.dropout(H4_FC, keep_prob=keep_prob)
    # Y_pred = tf.matmul(H3_FC, W4_FC) + B4_FC

    tf.summary.histogram("W4_FC", W4_FC)
    tf.summary.histogram("B4_FC", B4_FC)
    tf.summary.histogram("H4_FC", H4_FC)

with tf.variable_scope('Fully_connected_layer5'):
    W5_FC = tf.get_variable("W5_FC", [hiddenDim5, num_classes], tf.float32,
                            initializer=xavier_init(hiddenDim5, num_classes))
    B5_FC = tf.get_variable("B5_FC", [num_classes], tf.float32,
                            initializer=xavier_init(num_classes, num_classes))
    Y_pred = tf.matmul(H4_FC, W5_FC)+ B5_FC

    tf.summary.histogram("W5_FC", W5_FC)
    tf.summary.histogram("B5_FC", B5_FC)
    tf.summary.histogram("hypothesis", Y_pred)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y_one_hot))
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

rmse2 = tf.square(tf.argmax(Y_pred, 1) - tf.argmax(Y_one_hot, 1))
rmse = tf.sqrt(tf.reduce_mean( tf.cast(rmse2, tf.float32)))

last_epoch = tf.Variable(0, name="last_epoch")
tf.summary.scalar("loss", cost)

with tf.Session() as sess:
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    global_step = 0

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
    ###############################################################################
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        except:
            print("Error on loading old network weights")
    else:
        print("Could not find old network weights")
    ###############################################################################
    start_from = sess.run(last_epoch)

    total_batch = math.ceil(trainSize/batchSize)

    for epoch in range(start_from, training_epochs):
        batch_cost = 0
        # Training step
        for i in range(total_batch):
            batchX, batchY = trainX[i*batchSize:(i+1)*batchSize], trainY[i*batchSize:(i+1)*batchSize]
            feed_dict = {X: batchX, Y: batchY, keep_prob:0.7}
            s, _ = sess.run([summary,optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1

            batch_cost += sess.run(cost, feed_dict=feed_dict) / batchSize

        test_rmse, acc = sess.run([rmse, accuracy], feed_dict={X: testX, Y: testY, keep_prob:1})

        print("epoch:{:5>}, loss:{:>10}, acc:{}, test_rmse:{}".format(epoch, batch_cost, acc, test_rmse))

        sess.run(last_epoch.assign(epoch+1))
        if not os.path.exists(CHECK_POINT_DIR):
            os.makdirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "/model", global_step=epoch+1)

    # Plot predictions
    y = tf.argmax(Y_one_hot, 1)
    y_ = tf.argmax(Y_pred, 1)

    yp, y_p = sess.run([y, y_], feed_dict={X: testX, Y: testY, keep_prob: 1})
    plt.plot(yp, 'g')
    plt.plot(y_p, 'r')
    plt.xlabel("Time Period")
    plt.ylabel("Elbow Angle")
    plt.show()