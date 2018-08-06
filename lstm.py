# coding: utf-8

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from Alhena2.cn.cn_reader import (cn_reader)

def __read_data(mode='train', debug=False):

    if mode == 'train':
        (start, end) = ('2010-01-01', '2014-01-01')
    elif mode == 'test':
        (start, end) = ('2014-01-01', '2018-01-01')
    else:
        raise ValueError('%s not support' % mode)

    # Y
    # symbols = ['000651', '000750'] # tempz
    symbols = None
    ori_daily = cn_reader('.', symbols=symbols).daily(ex='backward')

    ori_daily = ori_daily.unstack(level=0)
    ori_daily = ori_daily.loc[start : end]
    ori_daily.dropna(axis=1, how='all', inplace=True)
    ori_daily.fillna(method='ffill', axis=1, inplace=True)

    mean = ori_daily.mean()
    
    percent = ori_daily.iloc[-1] / mean
    percent = percent > 1.2

    percent = percent.loc['close']

    labels = list(percent.index)

    if debug == True:
        percent.to_csv('t_labels.csv', encoding='gb2312')

    # X
    columns = ('ROE', 'CASH')
    reports = pd.read_hdf('all_cn.h5', key='report', mode='r')

    reports = reports.loc[(labels, slice(start, end)), columns]
    reports.fillna(0, inplace=True)  # tempz

    reports = reports.unstack(level=0).asfreq('A-DEC')

    X = np.zeros((len(labels), reports.index.size, len(columns)))

    reports = reports.stack(level=-1).swaplevel(axis=0)

    if debug == True:
        reports.to_csv('t_x.csv', encoding='gb2312')

    for (i, sym) in enumerate(labels):
        X[i,:,:] = reports.loc[sym].values

    return (X, percent.values, len(columns))

def __read_fake_data(n_samples, n_steps):

    import random as rd
    n_input = 2

    data = np.zeros((n_samples, n_steps, n_input))
    label = []
    for s in range(n_samples):

        b_rnd = rd.random() < 0.5

        for i in range(n_input):
            d = []
            if b_rnd:
                d = [rd.random() * 4.0 - 3.0 for i in range(n_steps)]
            else:
                if i == 0:
                    d = [rd.random() * 1.5 - 0.25 for i in range(n_steps)]
                elif i == 1:
                    d = [rd.random() * 1.5 - 0.25 for i in range(n_steps)]
                else:
                    pass

            data[s,:,i] = d

        label.append(int(b_rnd))

    return data, np.array(label), n_input

def standardize(train):
    """ Standardize data """

    x_mean = np.mean(train, axis=0)

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]

    return X_train

def one_hot(labels, n_class = 6):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def get_batches(X, y, batch_size = 100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

def __lstm(X, Y, chan_len, batch_size=100):

    from sklearn.model_selection import train_test_split
    #import matplotlib.pyplot as plt
    #get_ipython().magic('matplotlib inline')

    ### Prepare data
    # In[2]:
    X_train, labels_train, list_ch_train = (X, Y, chan_len)

    # In[3]:
    # Standardize
    # X_train = standardize(X_train)

    # Train/Validation Split
    # In[4]:
    X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                    stratify = labels_train, random_state = 123)
    # Fixed
    n_classes  = 2
    n_channels = chan_len

    #### Hyperparameters
    lstm_size     = 3*chan_len  # 3 times the amount of channels
    lstm_layers   = 2           # Number of layers
    seq_len       = 4           # Number of steps # tempz
    learning_rate = 0.0001      # Learning rate (default is 0.001)
    epochs        = 1000

    # One-hot encoding:
    # In[5]:
    y_tr  = one_hot(lab_tr,  n_class=n_classes)
    y_vld = one_hot(lab_vld, n_class=n_classes)

    # In[6]:
    # Imports

    # ### Construct the graph
    # Placeholders
    # In[8]:

    graph = tf.Graph()

    # Construct placeholders
    with graph.as_default():
        inputs_        = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels_        = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_     = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    # Build Convolutional Layer(s)
    # 
    # Questions: 
    # * Should we use a different activation? Like tf.nn.tanh?
    # * Should we use pooling? average or max?
    # In[9]:

    # Convolutional layers
    with graph.as_default():
        # (batch, samples, chan_len) --> (batch, samples, chan_len * 2)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=chan_len * 2, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        n_ch = n_channels * 2

    # Now, pass to LSTM cells
    # In[10]:

    with graph.as_default():
        # Construct the LSTM inputs and LSTM cells
        lstm_in = tf.transpose(conv1, [1,0,2])    # reshape into (seq_len, batch, channels)
        lstm_in = tf.reshape(lstm_in, [-1, n_ch]) # Now (seq_len*N, n_channels)
        
        # To cells
        lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None) # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
        
        # Open up the tensor into a list of seq_len pieces
        lstm_in = tf.split(lstm_in, seq_len, 0)
        
        # Add LSTM layers
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)


    # Define forward pass and cost function:
    # In[11]:

    with graph.as_default():
        outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                         initial_state = initial_state)
        
        # We only need the last output tensor to pass into a classifier
        logits = tf.layers.dense(outputs[-1], n_classes, name='logits')
        
        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        #optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping
        
        # Grad clipping
        train_op = tf.train.AdamOptimizer(learning_rate_)

        gradients = train_op.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        optimizer = train_op.apply_gradients(capped_gradients)
        
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    #### Train the network
    # In[12]:

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    # In[13]:

    validation_acc  = []
    validation_loss = []

    train_acc  = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
        
        for e in range(epochs):
            # Initialize 
            state = sess.run(initial_state)
            
            # Loop over batches
            for x,y in get_batches(X_tr, y_tr, batch_size):
                
                # Feed dictionary
                feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, 
                        initial_state : state, learning_rate_ : learning_rate}
                
                loss, _ , state, acc = sess.run([cost, optimizer, final_state, accuracy], 
                                                 feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)
                
                # Print at each 5 iters
                if (iteration % 5 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))
                
                # Compute validation loss at every 25 iterations
                if (iteration%25 == 0):
                    
                    # Initiate for validation set
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    
                    val_acc_ = []
                    val_loss_ = []
                    for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                        # Feed
                        feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0, initial_state : val_state}
                        
                        # Loss
                        loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict = feed)
                        
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)
                    
                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(np.mean(val_loss_)),
                          "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                    
                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))
                
                # Iterate 
                iteration += 1
        
        saver.save(sess,"checkpoints/lstm.ckpt")

def __lstm_test(X, Y, batch_size=100):

    y_vld = one_hot(Y, n_class=2)

    # TODO:
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.Saver()

    test_acc = []

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        
        for x_t, y_t in get_batches(X, y_vld, batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1}
            
            batch_acc = sess.run(accuracy, feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

def main(mode='train'):

    if mode == 'train':
        (x, y, len) = __read_fake_data(n_samples=10000, n_steps=4)
        # (x, y, len) = __read_data(mode='train', debug=True)

        __lstm(X=x, Y=y, chan_len=len, batch_size=100)

    elif mode == 'test':
        (x, y, len) = __read_fake_data(n_samples=1000, n_steps=4)
        # (x, y, len) = __read_data(mode='test', debug=True)

        __lstm_test(X=x, Y=y, batch_size=100)

if __name__ == '__main__':
    main(mode='train')