import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from skimage.data import astronaut
from scipy.misc import imresize
plt.style.use('ggplot')

# ==== DESCRIPTION OF THE NETWORK ====
"""
This neural network implements a fun application : painting
and image. The input to the network is a position on the 
image, X = (row, col). The output is the color to paint,
Y = (R, G, B). The network is of type fully connected and is
composed of 6 hidden layers each containing 64 neurons.
The network relies on a supervised type of learning, given 
that each input position (x, y) is accompanied by its 
corresponding (r, g, b) label in the picture.
"""

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
    """
    Takes two matrices of the same dimensions and returns the matrix
    of abs(mat1(i,j) - mat2(i,j)) for every (i,j) in the matrices.
    :param p1: matrix of the same dimensions as p1
    :param p2: matrix of the same dimensions as p2
    :return: the matrix of abs(p1(i,j) - mp2(i,j)) for every (i,j) in p1 and p2
    """
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
            name='W',  # it's ok to give the same name to the weights of each layer thanks to the scope
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


# X is the input to the network, containing batch_size number of elements
# (x, y), representing the coordinates of the pixels for which we
# want to guess the corresponding RGB values. None can be replaced by any value
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

# ==== CREATION OF THE NETWORK ====
n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

# Generate the layers indicated in the n_neurons list.
# This generates the weights, biases and activation functions
# The activation values calculated by the linear function
# are stored in current_input which is used as input to create the next layer
current_input = X
for layer_i in range(1, len(n_neurons)):
    current_input = linear(
        X=current_input,
        n_input=n_neurons[layer_i - 1],
        n_output=n_neurons[layer_i],
        activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
        scope='layer_' + str(layer_i))
Y_pred = current_input  # Y_pred receives the activation values of the output layer

# ==== TRAINING THE NETWORK ====
# Sum the R, G and B error values for each row (i.e. each predicted-example's error)
# Then take the mean error over all the examples of the batch
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
        # Get a random permutation of indices to randomize the order of our input data
        indices = np.random.permutation(range(len(xs)))
        n_batches = len(indices) // batch_size
        for batch_i in range(n_batches):
            # Get the random indices corresponding to the current batch
            indices_i = indices[batch_i * batch_size: (batch_i + 1) * batch_size]
            # Calculate the gradients of the examples in the batch, take the mean
            # gradient, and modify the weights and biases of the network accordingly
            sess.run(optimizer, feed_dict={X: xs[indices_i], Y: ys[indices_i]})

        # Once all the batches have been processed, re-iterate the randomization
        # and train again
        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(it_i, training_cost)

        if (it_i + 1) % 20 == 0:
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            fig, ax = plt.subplots(1, 1)
            img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
            plt.imshow(img)
            #fig.canvas.draw()
            plt.ion()
            plt.show()
            #plt.draw()
            plt.pause(0.001)
