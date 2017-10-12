'''
Created on 2017. 9. 21.

@author: em
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os
from prettytensor import xavier_init

#tf.set_random_seed()  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

training_epochs = 1000000
learningRate = 0.01
batchSize = 1

RNNLayers = 2
seq_length = 10
inputDim = 4
hiddenDim1 = 40
hiddenDim2 = 60
hiddenDim3 = 70
hiddenDim4 = 50
hiddenDim5 = 30
outputDim = 1

CHECK_POINT_DIR = TB_SUMMARY_DIR = "./log/L2-10seq-40-60-70-50-30-1"

xy = np.loadtxt('./emg/1n_angle_zeroADC_sd.txt', delimiter=' ')

x = xy[50:, :-1]
y = xy[50:, [-1]]

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]

    dataX.append(_x)
    dataY.append(_y)

trainSize = int(len(dataY) * 0.8)
testSize = len(dataY) - trainSize
trainX, testX = np.array(dataX[0:trainSize]), np.array(dataX[trainSize:len(dataX)])
trainY, testY = np.array(dataY[0:trainSize]), np.array(dataY[trainSize:len(dataX)])

X = tf.placeholder(tf.float32, shape=[None, seq_length, inputDim], name="EMGs")
Y = tf.placeholder(tf.float32, shape=[None, 1], name="Angle")
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('LSTM_layer1'):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=inputDim, state_is_tuple=True, activation=tf.tanh)
    cells = tf.nn.rnn_cell.MultiRNNCell([cell] * RNNLayers, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
    X_FC = tf.contrib.layers.fully_connected(outputs[:,-1], hiddenDim1, activation_fn=None)

with tf.variable_scope('Fully_connected_layer1'):
    W1_FC = tf.get_variable("W1_FC", [hiddenDim1, hiddenDim2], tf.float32, initializer=xavier_init(hiddenDim1, hiddenDim2))
    B1_FC = tf.get_variable("B1_FC", [hiddenDim2], tf.float32, initializer=xavier_init(hiddenDim2, hiddenDim2))
    H1_FC = tf.nn.relu( tf.matmul(X_FC, W1_FC)+ B1_FC, name="H1_FC" )
    H1_FC = tf.nn.dropout(H1_FC, keep_prob=keep_prob)

    tf.summary.histogram("W1_FC", W1_FC)
    tf.summary.histogram("B1_FC", B1_FC)
    tf.summary.histogram("H1_FC", H1_FC)

with tf.variable_scope('Fully_connected_layer2'):
    W2_FC = tf.get_variable("W2_FC", [hiddenDim2, hiddenDim3], tf.float32,
                            initializer=xavier_init(hiddenDim2, hiddenDim3))
    B2_FC = tf.get_variable("B2_FC", [hiddenDim3], tf.float32,
                            initializer=xavier_init(hiddenDim3, hiddenDim3))
    H2_FC = tf.nn.relu(tf.matmul(H1_FC, W2_FC) + B2_FC, name="H1_FC")
    H2_FC = tf.nn.dropout(H2_FC, keep_prob=keep_prob)

    tf.summary.histogram("W2_FC", W2_FC)
    tf.summary.histogram("B2_FC", B2_FC)
    tf.summary.histogram("H2_FC", H2_FC)

with tf.variable_scope('Fully_connected_layer3'):
    W3_FC = tf.get_variable("W3_FC", [hiddenDim3, hiddenDim4], tf.float32,
                            initializer=xavier_init(hiddenDim3, hiddenDim4))
    B3_FC = tf.get_variable("B3_FC", [hiddenDim4], tf.float32,
                            initializer=xavier_init(hiddenDim4, hiddenDim4))
    H3_FC = tf.nn.relu(tf.matmul(H2_FC, W3_FC) + B3_FC, name="H3_FC")
    H3_FC = tf.nn.dropout(H3_FC, keep_prob=keep_prob)

    tf.summary.histogram("W3_FC", W3_FC)
    tf.summary.histogram("B3_FC", B3_FC)
    tf.summary.histogram("H3_FC", H3_FC)

with tf.variable_scope('Fully_connected_layer4'):
    W4_FC = tf.get_variable("W4_FC", [hiddenDim4, hiddenDim5], tf.float32,
                            initializer=xavier_init(hiddenDim4, hiddenDim5))
    B4_FC = tf.get_variable("B4_FC", [hiddenDim5], tf.float32,
                            initializer=xavier_init(hiddenDim5, hiddenDim5))
    H4_FC = tf.nn.relu(tf.matmul(H3_FC, W4_FC) + B4_FC, name="H4_FC")
    H4_FC = tf.nn.dropout(H4_FC, keep_prob=keep_prob)

    tf.summary.histogram("W4_FC", W4_FC)
    tf.summary.histogram("B4_FC", B4_FC)
    tf.summary.histogram("H4_FC", H4_FC)

with tf.variable_scope('Fully_connected_layer5'):
    W5_FC = tf.get_variable("W5_FC", [hiddenDim5, outputDim], tf.float32,
                            initializer=xavier_init(hiddenDim5, outputDim))
    B5_FC = tf.get_variable("B5_FC", [outputDim], tf.float32,
                            initializer=xavier_init(outputDim, outputDim))
    Y_pred = tf.matmul(H4_FC, W5_FC)+ B5_FC

    tf.summary.histogram("W5_FC", W5_FC)
    tf.summary.histogram("B5_FC", B5_FC)
    tf.summary.histogram("hypothesis", Y_pred)

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

last_epoch = tf.Variable(0, name="last_epoch")

    tf.summary.histogram("hypothesis", Y_pred)

loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

last_epoch = tf.Variable(0, name="last_epoch")

tf.summary.scalar("loss", loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

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

    for epoch in range(start_from, training_epochs):
        avg_cost = 0

        # Training step
        for i in range(batchSize):
            feed_dict = {X: trainX, Y: trainY, keep_prob:0.7}
            s, _ = sess.run([summary,optimizer], feed_dict=feed_dict)
            writer.add_summary(s, global_step=global_step)
            global_step += 1

            avg_cost += sess.run(loss, feed_dict=feed_dict)/(len(y)-seq_length) #/ total_batch


        sess.run(last_epoch.assign(epoch+1))
        if not os.path.exists(CHECK_POINT_DIR):
            os.makdirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "/model", global_step=epoch+1)

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testX, keep_prob:1})
        rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})

        print("Epoch= {:>5}, loss= {:>12}, RMSE: {}".format(epoch+1, avg_cost, rmse_val))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX, keep_prob:1})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY, 'r')
    plt.plot(test_predict, 'g')
    plt.xlabel("Time Period")
    plt.ylabel("Elbow Angle")
    plt.show()