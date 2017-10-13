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

training_epochs = 1000
learningRate = 0.01
batchSize = 50

inputDim = 4
RNNLayers = 2
seq_length = 10
hiddenDim1 = 50
hiddenDim2 = 100
hiddenDim3 = 100
# hiddenDim4 = 100
# hiddenDim5 = 50
num_classes = 130

CHECK_POINT_DIR = TB_SUMMARY_DIR = "./log/L2-10seq-50-100-100-130"

xy = np.loadtxt('./emg/1n_angle_zeroADC_sd.txt', delimiter=' ')

# x = xy[50:, :-1]
# y = xy[50:, [-1]]

x = xy[:, :-1]
y = xy[:, [-1]]

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

trainSize = int(len(dataY) * 0.7)
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

    tf.summary.histogram("W2_FC", W2_FC)
    tf.summary.histogram("B2_FC", B2_FC)
    tf.summary.histogram("H2_FC", H2_FC)

with tf.variable_scope('Fully_connected_layer3'):
    W3_FC = tf.get_variable("W3_FC", [hiddenDim3, num_classes], tf.float32,
                            initializer=xavier_init(hiddenDim3, num_classes))
    B3_FC = tf.get_variable("B3_FC", [num_classes], tf.float32,
                            initializer=xavier_init(1, num_classes))
    # H3_FC = tf.nn.relu(tf.matmul(H2_FC, W3_FC) + B3_FC, name="H3_FC")
    # H3_FC = tf.nn.dropout(H3_FC, keep_prob=keep_prob)
    Y_pred = tf.matmul(H2_FC, W3_FC) + B3_FC


    tf.summary.histogram("W3_FC", W3_FC)
    tf.summary.histogram("B3_FC", B3_FC)
    tf.summary.histogram("H3_FC", Y_pred)

# with tf.variable_scope('Fully_connected_layer4'):
#     W4_FC = tf.get_variable("W4_FC", [hiddenDim4, hiddenDim5], tf.float32,
#                             initializer=xavier_init(hiddenDim4, hiddenDim5))
#     B4_FC = tf.get_variable("B4_FC", [hiddenDim5], tf.float32,
#                             initializer=xavier_init(hiddenDim5, hiddenDim5))
#     H4_FC = tf.nn.relu(tf.matmul(H3_FC, W4_FC) + B4_FC, name="H4_FC")
#     H4_FC = tf.nn.dropout(H4_FC, keep_prob=keep_prob)
#
#     tf.summary.histogram("W4_FC", W2_FC)
#     tf.summary.histogram("B4_FC", B2_FC)
#     tf.summary.histogram("H4_FC", H4_FC)
#
# with tf.variable_scope('Fully_connected_layer5'):
#     W5_FC = tf.get_variable("W5_FC", [hiddenDim5, outputDim], tf.float32,
#                             initializer=xavier_init(hiddenDim5, outputDim))
#     B5_FC = tf.get_variable("B5_FC", [outputDim], tf.float32,
#                             initializer=xavier_init(outputDim, outputDim))
#     Y_pred = tf.matmul(H4_FC, W5_FC)+ B5_FC
#
#     tf.summary.histogram("W5_FC", W5_FC)
#     tf.summary.histogram("B5_FC", B5_FC)
#     tf.summary.histogram("hypothesis", Y_pred)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y_one_hot))
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

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

        # Training step
        for i in range(total_batch):
            batchX, batchY = trainX[i*batchSize:(i+1)*batchSize], trainY[i*batchSize:(i+1)*batchSize]

            feed_dict = {X: batchX, Y: batchY, keep_prob:0.7}
            s, _ = sess.run([summary,optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1

        loss, acc = sess.run([cost, accuracy], feed_dict={X: testX, Y: testY, keep_prob:0.7})
        print("loss:{:>10}, acc:{:.3%}".format(loss, acc))

        sess.run(last_epoch.assign(epoch+1))
        if not os.path.exists(CHECK_POINT_DIR):
            os.makdirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "/model", global_step=epoch+1)

        pred = sess.run([cost, accuracy], feed_dict={X: testX, Y: testY, keep_prob:1.0})
        for p, y in zip(pred, testY.flatten()):
            print("[{}] Prediction: {:>5} True Y: {:>3}".format(p == int(y), p, int(y)))

    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_one_hot, 1))

    # Test step
    predction, real = sess.run([tf.argmax(Y_pred,1), tf.argmax(Y_one_hot,1)], feed_dict={X: testX, Y:testY, keep_prob:1})
    # test_predict = sess.run(Y_pred, feed_dict={X: testX})
    # rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    # print("RMSE: {}".format(rmse_val))

    # Plot predictions
    # plt.plot(testY, 'r')
    # plt.plot(test_predict, 'g')
    plt.plot(predction)
    plt.plot(real)
    plt.xlabel("Time Period")
    plt.ylabel("Elbow Angle")
    plt.show()