import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipyd
from libs.datasets import MNIST
from libs import gif
from libs.utils import montage

# ==== DESCRIPTION OF THE NETWORK ====
"""
This Auto Encoder Neural Network takes a 28 by 28 (784 values) pixel 
image (or a batch of images) as its input. It learns to encode it 
down to 7 feature maps of 7x7 values, and to retrieve the original image 
as its output, only using the encoded values. This convolutional Neural 
Network learns in an unsupervised fashion as the input data is unlabeled.
"""

# ==== IMPORTING THE DATA ====
ds = MNIST()
print(ds.X.shape)
mean_img = np.mean(ds.X, axis=0)

# ==== ENCODER - CREATION OF THE FIRST HALF OF THE NETWORK ====
n_features = ds.X.shape[1]
X = tf.placeholder(tf.float32, [None, n_features])  # [batch, height*width]
# Reshape our input to 4D : N x H x W x C to use in a convolutional graph
# special value -1 is computed depending on the batch value
X_tensor = tf.reshape(X, [-1, 28, 28, 1])

# Define the number of layers, the number of filters per layer, and the sizes of the filters
n_filters = [16, 16, 16]
filter_sizes = [4, 4, 4]

current_input = X_tensor
# notice instead of having 784 as our input features, we're going to have
# just 1, corresponding to the number of channels in the image.
# We're going to use convolution to find 16 filters, or 16 channels of information
# in each spatial location we perform convolution at.
n_input = 1

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []
shapes = []

# We'll use a for loop to create each layer:
for layer_i, n_output in enumerate(n_filters):
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):
        # we'll keep track of the shapes of each layer
        # As we'll need these for the decoder
        shapes.append(current_input.get_shape().as_list())

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input
        W = tf.get_variable(
            name='W',
            shape=[filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02)
        )

        # Now we'll convolve our input by our newly created W matrix
        h = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')

        # And then use a ReLU activation function on its output
        current_input = tf.nn.relu(h)

        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output

# ==== DECODER - RETRIEVE THE ORIGINAL IMAGES FROM THE COMPRESSED DATA ====
# We'll first reverse the order of our weight matrices
Ws.reverse()
# and the shapes of each layer
shapes.reverse()
# and the number of filters
n_filters.reverse()
# and append the last filter size which is our input image's number of channels
n_filters = n_filters[1:] + [1]

print(n_filters, filter_sizes, shapes)

# and then loop through our convolution filters and get back our input image
# we'll enumerate the shapes list to get us there
for layer_i, shape in enumerate(shapes):
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):
        W = Ws[layer_i]

        # Now we'll convolve by the transpose of our previous convolution tensor
        h = tf.nn.conv2d_transpose(
            current_input,
            W,
            tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
            strides=[1, 2, 2, 1], padding='SAME'
        )

        # And then use a ReLU activation function on its output
        current_input = tf.nn.relu(h)

Y = current_input
Y = tf.reshape(Y, [-1, n_features])

# ==== TRAINING THE NETWORK ====
cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1))
learning_rate = 0.001

# pass learning rate and cost to optimize
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Session to manage vars/train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 100
n_epochs = 5

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
examples = ds.X[:100]

# We'll store the reconstructions in a list
images = []
fig, ax = plt.subplots(1, 1)
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_X - mean_img})
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
    img_i = montage(recon).astype(np.uint8)
    images.append(img_i)
    ax.imshow(img_i, cmap='gray')
    fig.canvas.draw()
    print(epoch_i, sess.run(cost, feed_dict={X: batch_X - mean_img}))
gif.build_gif(images, saveto='conv-ae.gif', cmap='gray')

ipyd.Image(url='conv-ae.gif?{}'.format(np.random.rand()), height=500, width=500)
