import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.style.use('ggplot')

# DESCRIPTION OF THE NETWORK
"""
This neural network is simply a deeper version of 
the network described in the file 'Nonlinearities_Activation.py'.
The network now contains three hidden layers, of 10 neurons each. 
"""

# Let's create some toy data

# We are going to say that we have seen 1000 values of some underlying representation that we aim to discover
n_observations = 1000

# Instead of having an image as our input, we're going to have values from -3 to 3.  This is going to be the
# input to our network.
xs = np.linspace(-3, 3, n_observations)

# From this input, we're going to teach our network to represent a function that looks like a sine wave.
#  To make it difficult, we are going to create a noisy representation of a sine wave by adding uniform noise.
# So our true representation is a sine wave, but we are going to make it difficult by adding some noise to the function,
# and try to have our algorithm discover the underlying cause of the data, which is the sine wave without any noise.
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
plt.scatter(xs, ys, alpha=0.15, marker='+')


def distance(p1, p2):
    return tf.abs(p1 - p2)


X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.nn.tanh(tf.matmul(X, W) + b, name='h')
        if activation is not None:
            h = activation(h)
        return h


h = linear(X, 2, 10, scope='layer1')

# Let's connect the output of our last network to another one!
# Wow we just made a Deep Network!  Let's make it deeper!
h2 = linear(h, 10, 10, scope='layer2')

# Again!
h3 = linear(h2, 10, 3, scope='layer3')
Y_pred = tf.reduce_sum(h, 1)


def train(X, Y, Y_pred, n_iterations=100, batch_size=200, learning_rate=0.02):
    cost = tf.reduce_mean(distance(Y_pred, Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, alpha=0.15, marker='+')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-2, 2])
    with tf.Session() as sess:
        # Here we tell TensorFlow that we want to initialize all
        # the variables in the graph so we can use them
        # This will set W and b to their initial random normal value.
        sess.run(tf.global_variables_initializer())

        # We now run a loop over epochs
        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            indexes = np.random.permutation(range(len(xs)))
            n_batches = len(indexes) // batch_size
            for batch_i in range(n_batches):
                # For every batch, calculate the gradient, and update the weight and bias
                indexes_i = indexes[batch_i * batch_size: (batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: xs[indexes_i], Y: ys[indexes_i]})

            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if it_i % 10 == 0:
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
                ax.plot(xs, ys_pred, 'k', alpha=it_i / n_iterations)
                print(training_cost)

            # Allow the training to quit if we've reached a minimum (if the new cost isn’t very different
            # from the previous, that means we’re not going to go anywhere else, we’re turning around the minimum)
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                break

            # Keep track of the training cost
            prev_training_cost = training_cost

    fig.show()
    plt.draw()


# Retrain with our new Y_pred
train(X, Y, Y_pred)
