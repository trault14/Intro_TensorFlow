# %matplotlib inline
# %pylab osx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
# Some additional libraries which we'll use just
# to produce some visualizations of our training
from libs.utils import montage
from libs import gif
import IPython.display as ipyd
from libs.datasets import MNIST
# Bit of formatting of the default inline code style:
from IPython.core.display import HTML
HTML("""<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>""")
plt.style.use('ggplot')

# ==== IMPORT THE DATA ====
ds = MNIST()
print(ds.X.shape)
# re-shape the first image to a square so we can see all
# the pixels arranged in rows and columns instead of one giant vector
plt.imshow(ds.X[0].reshape((28, 28)))

# ==== LOOK AT THE MEAN AND STDDEV OF THE DATA SET ====
# Take the mean across all images
mean_img = np.mean(ds.X, axis=0)
# Then plot the mean image.
plt.figure()
plt.imshow(mean_img.reshape((28, 28)), cmap='gray')

# Take the std across all images
std_img = np.std(ds.X, axis=0)
# Then plot the std image.
plt.figure()
plt.imshow(std_img.reshape((28, 28)))

# ==== ENCODER - CREATION OF THE FIRST HALF OF THE NETWORK ====
# We are going to build a series of fully connected layers that get progressively smaller.
# The number of neurons in the input layer is going to be the number of pixels in an image
dimensions = [512, 256, 128, 64]

# The number of features is the second dimension of our inputs matrix, 784, 28 by 28
n_features = ds.X.shape[1]

# And we'll create a placeholder in the TensorFlow graph that will be able to get any number of n_feature inputs.
X = tf.placeholder(tf.float32, [None, n_features])

current_input = X
n_input = n_features

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []

# We'll use a for loop to create each layer:
for layer_i, n_output in enumerate(dimensions):

    # just like in the last session,
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        # Now we'll multiply our input by our newly created W matrix
        # We could also add a bias
        h = tf.matmul(current_input, W)

        # And then use a ReLu activation function on its output
        current_input = tf.nn.relu(h)

        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number of inputs is correct.
        n_input = n_output

print(current_input.get_shape())

# ==== DECODER - REVERSE THE INPUT IMAGES BACK TO THEIR ORIGINAL DIMENSIONS ====
# We'll first reverse the order of our weight matrices
Ws = Ws[::-1]

# Then reverse the order of our dimensions,
# appending the last layer's number of inputs.
dimensions = dimensions[::-1][1:] + [ds.X.shape[1]]
print(dimensions)

for layer_i, n_output in enumerate(dimensions):
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):

        # Now we'll grab the weight matrix we created before and transpose it
        # So a 3072 x 784 matrix would become 784 x 3072
        # or a 256 x 64 matrix, would become 64 x 256
        W = tf.transpose(Ws[layer_i])

        # Now we'll multiply our input by our transposed W matrix
        h = tf.matmul(current_input, W)
        current_input = tf.nn.relu(h)
        n_input = n_output

# The output of the network
Y = current_input

# ==== TRAINING THE NETWORK ====
# Now we need to define a training signal to train the network.
# We'll first measure the average difference across every pixel
cost = tf.reduce_mean(tf.squared_difference(X, Y), 1)
print(cost.get_shape())
# Then we take the mean again across the batch
# So, the cost is measuring the average pixel difference across our mini batch
cost = tf.reduce_mean(cost)

# We can now use an optimizer to train our network
learning_rate = 0.001
# The optimizer is going to apply back propagation to follow the gradient
# from the output of the network all the way back to the input and update
# all of the variables along the way.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Let's manage the training in mini batches
# We create a session to use the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 100
n_epochs = 5

# We'll try to reconstruct the same first 100 images and show how
# the network does over the course of training.
examples = ds.X[:100]

# We'll store the reconstructions in a list
images = []
fig, ax = plt.subplots(1, 1)
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_X - mean_img})
    # After every epoch, we try to reconstruct the first 100 images to get
    # an idea of how well the algorithm is learning over time.
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    # -1 is an inferred dimension. It represents the number of 28 by 28 images that
    # we are going to get back when doing the reshape
    recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
    img_i = montage(recon).astype(np.uint8)
    images.append(img_i)
    ax.imshow(img_i, cmap='gray')
    fig.canvas.draw()
    print(epoch_i, sess.run(cost, feed_dict={X: batch_X - mean_img}))
gif.build_gif(images, saveto='ae.gif', cmap='gray')

ipyd.Image(url='ae.gif?{}'.format(np.random.rand()), height=500, width=500)
