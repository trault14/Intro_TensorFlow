import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from skimage.data import astronaut
from scipy.misc import imresize
plt.style.use('ggplot')

# ==== CREATION OF THE DATA ====
img = imresize(astronaut(), (64, 64))
plt.imshow(img)

# We'll first collect all the positions in the image in our list, xs
xs = []

# And the corresponding colors for each of these positions
ys = []

# Now loop over the image
for row_i in range(img.shape[0]):
    for col_i in range(img.shape[1]):
        # And store the inputs
        xs.append([row_i, col_i])
        # And outputs that the network needs to learn to predict
        ys.append(img[row_i, col_i])

# we'll convert our lists to arrays
xs = np.array(xs)
ys = np.array(ys)

# Normalizing the input by the mean and standard deviation (standard score)
xs = (xs - np.mean(xs)) / np.std(xs)

# and print the shapes
print(xs.shape, ys.shape)


def distance(p1, p2):
    return tf.abs(p1 - p2)


def linear(X, n_input, n_output, activation=None, scope=None):
    """
    Creates the linear layer L+1 using the activation values of layer L.
    The weights from L to L+1 are generated randomly, as well as the
    biases of layer L+1.
    :param X: activation values from layer L
    :param n_input: number of activation values from L
    :param n_output: desired number of neurons in layer L+1
    :param activation: desired activation function to use in
    layer L+1 (e.g. tanh, sigmoid etc...)
    :param scope: the label under which the operations generated
    by the function will be placed in the TensorFlow graph
    :return: the activation values of layer L+1
    """
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        z = tf.matmul(X, W) + b
        if activation is not None:
            a = activation(z)
        else:
            a = z
        return a


X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

# ==== CREATION OF THE NETWORK ====
n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

current_input = X
for layer_i in range(1, len(n_neurons)):
    current_input = linear(
        X=current_input,
        n_input=n_neurons[layer_i - 1],
        n_output=n_neurons[layer_i],
        activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
        scope='layer_' + str(layer_i))
Y_pred = current_input

# ==== TRAINING THE NETWORK ====
cost = tf.reduce_mean(tf.reduce_sum(distance(Y_pred, Y), 1))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

n_iterations = 500
batch_size = 50
with tf.Session() as sess:
    # Here we tell TensorFlow that we want to initialize all
    # the variables in the graph so we can use them
    # This will set W and b to their initial random normal value.
    sess.run(tf.global_variables_initializer())

    # We now run a loop over epochs
    prev_training_cost = 0.0
    for it_i in range(n_iterations):
        idxs = np.random.permutation(range(len(xs)))
        n_batches = len(idxs) // batch_size
        for batch_i in range(n_batches):
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(it_i, training_cost)

        if (it_i + 1) % 20 == 0:
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            fig, ax = plt.subplots(1, 1)
            img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
            plt.imshow(img)
            #fig.canvas.draw()
            plt.show()
