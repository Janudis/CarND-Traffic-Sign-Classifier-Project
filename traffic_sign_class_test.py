import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import random
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#import pandas as pd

with open('german-traffic-signs/train.p','rb') as f:
    train = pickle.load(f)
with open('german-traffic-signs/valid.p','rb') as f:
    val = pickle.load(f)
with open('german-traffic-signs/test.p','rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_test, y_test = test['features'], test['labels']


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.compat.v1.layers.flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.compat.v1.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


# Visualize images
f, ax = plt.subplots(2, 3, figsize=(15,8))
ax[0,0].imshow(x_test[0])
ax[0,1].imshow(x_test[1])
ax[0,2].imshow(x_test[2])
ax[1,0].imshow(x_test[3])
ax[1,1].imshow(x_test[4])
ax[1,2].imshow(x_test[5])
plt.show()

# Use the model to output the prediction for each image

# Declare TrafficNet logits
x = tf.compat.v1.placeholder(tf.float32, (None,32,32,3))
prob_keep = tf.compat.v1.placeholder(tf.float32)
logits = LeNet(x)

# Restore model
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()
saver.restore(sess, "./lenet")

# # Obtain CNN results for `imgs`
imgs = x_test[0:6]
logits_imgs = sess.run(logits, feed_dict={x: imgs, prob_keep: 1})
softmax_imgs = sess.run(tf.nn.softmax(logits_imgs))
des_sort_idx = np.argsort(-softmax_imgs)
num_imgs = des_sort_idx.shape[0]

preds = des_sort_idx[:,0]
print('Predictions (top-left to bottom-right):', preds)

# Print out the top-five softmax probabilities
# top_k = 5
# for k in range(num_imgs):
#     print('Image', k+1, '--------')
#     print('Class \t Softmax')
#     for i in range(top_k):
#         cur_class = des_sort_idx[0,i]
#         print(cur_class, '\t {:.5f}'.format(softmax_imgs[0,cur_class]))

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "./lenet")
    web_classes = sess.run(logits, feed_dict={x: x_test, prob_keep : 1.0})
    web_softmax = sess.run(tf.nn.softmax(logits), feed_dict={x: x_test, prob_keep : 1.0})

with tf.compat.v1.Session() as sess:
    predicts = sess.run(tf.nn.top_k(web_softmax, k=5, sorted=True))

for i in range(num_imgs):
    for j in range(0, len(predicts[0][i])):
        prob = predicts[0][i][j]
        index = predicts[1][i][j]

        print('   {:.6f} : {} '.format(prob, index))
    print()

