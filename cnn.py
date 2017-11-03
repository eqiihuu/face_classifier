import string, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
import pandas as pd
import tensorflow as tf


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 48, 48, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


proj_dir = '/Users/qihucn/Documents/EE576/Project/faceDetect'
train_path = os.path.join(proj_dir, 'train.csv')
test_path = os.path.join(proj_dir, 'test.csv')

print 'Load data ...'
train_data = np.asarray(pd.read_csv(train_path, skiprows=[0]))
test_data = np.asarray(pd.read_csv(test_path, skiprows=[0]))
train_label0 = train_data[:, 0]
train_img = train_data[:, 1:2305]
test_label0 = test_data[:, 0]
test_img = test_data[:, 1:2305]

N_TRAIN = train_label0.size
N_TEST = test_label0.size

# On-hot coding
train_label = np.zeros((N_TRAIN, 7), dtype=int)
test_label = np.zeros((N_TEST, 7), dtype=int)
for i in range(N_TRAIN):
    train_label[i, train_label0[i]] = 1
for i in range(N_TEST):
    test_label[i, test_label0[i]] = 1

# Mode parameters
BATCH_SIZE = 50
DROPOUT = 0.5
TRAIN_EPOCH = 100
LEARNING_RATE = 0.001

TRAIN_BATCH = N_TRAIN/BATCH_SIZE
TEST_BATCH = N_TEST/BATCH_SIZE
N_PIXEL = 2304
N_CLASS = 7

x = tf.placeholder(tf.float32, [None, N_PIXEL])
y = tf.placeholder(tf.float32, [None, N_CLASS])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128])),
    # 3x3 conv, 128 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    # 3x3 conv, 64 inputs, 32 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    # fully connected,
    'wd1': tf.Variable(tf.random_normal([6*6*32, 200])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([200, N_CLASS]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([128])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([200])),
    'out': tf.Variable(tf.random_normal([N_CLASS]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

Train_ind = np.arange(N_TRAIN)
Test_ind = np.arange(N_TEST)

with tf.Session() as sess:
    sess.run(init)
    print 'Train ...'
    for epoch in range(0, TRAIN_EPOCH):

        Total_test_loss = 0
        Total_test_acc = 0

        for train_batch in range(0, TRAIN_BATCH):
            sample_ind = Train_ind[train_batch * BATCH_SIZE:(train_batch + 1) * BATCH_SIZE]
            batch_x = train_img[sample_ind, :]
            batch_y = train_label[sample_ind, :]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: DROPOUT})

            if train_batch % BATCH_SIZE == 0:
                # Calculate loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})

                print("Epoch: " + str(epoch+1) + ", Batch: " + str(train_batch) + ", Loss= " + \
                      "{:.3f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

        # Calculate test loss and test accuracy
        print 'Test ...'
        for test_batch in range(0, TEST_BATCH):
            sample_ind = Test_ind[test_batch * BATCH_SIZE:(test_batch + 1) * BATCH_SIZE]
            batch_x = test_img[sample_ind, :]
            batch_y = test_label[sample_ind, :]
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 1.})
            Total_test_loss += test_loss
            Total_test_acc += test_acc

        Total_test_acc /= TEST_BATCH
        Total_test_loss /= TEST_BATCH

        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + \
              "{:.3f}".format(Total_test_loss) + ", Test Accuracy= " + \
              "{:.3f}".format(Total_test_acc))

plt.subplot(2, 1, 1)
plt.ylabel('Test loss')
plt.plot(Total_test_loss, 'r')
plt.subplot(2, 1, 2)
plt.ylabel('Test Accuracy')
plt.plot(Total_test_acc, 'r')


print "All is well"
plt.show()
