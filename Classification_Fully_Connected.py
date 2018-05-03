import tensorflow as tf
from libs import datasets
from libs import utils

"""
==== DESCRIPTION OF THE NETWORK ====
This fully connected neural network is trained to classify
images that each represent the drawing of a number from
0 to 9. The input to the network is an image of 28x28 pixels
(784 input neurons). The output is a 10-digit one-hot encoding
containing the probability that the network associates to 
each possible label for classifying the input image. 
Supervised learning is used here, as the cost function compares
the predicted output with the true output by means of a 
cross entropy function. 
"""

# ==== GATHERING THE DATA ====
ds = datasets.MNIST(split=[0.8, 0.1, 0.1])
# The input is the brightness of every pixel in the image
n_input = 28 * 28
# The output is the one-hot encoding of the label
n_output = 10

# ==== DEFINITION OF THE NETWORK ====
# First dimension is flexible to allow for both batches
# and singles images to be processed
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

# Connect the input to the output with a linear layer.
# We use the SoftMax activation to make sure the outputs
#  sum to 1, making them probabilities
Y_pred, W = utils.linear(
    x=X,
    n_output=n_output,
    activation=tf.nn.softmax,
    name='layer1')

# Cost function. We add 1e-12 (epsilon) to avoid reaching 0,
# where log is undefined.
cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred + 1e-12))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Output. The predicted class is the one weighted with the
# highest probability in our regression output :
predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)

# We can measure the accuracy of our network like so.
# Note that this is not used to train the network.
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# ==== TRAINING THE NETWORK ====
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 50
n_epochs = 5
previous_validation_accuracy = 0
for epoch_i in range(n_epochs):
    for batch_xs, batch_ys in ds.train.next_batch():
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
    # Get an estimation of the new accuracy of the network, using the validation data set :
    valid = ds.valid
    # If the validation accuracy reaches a threshold, or if it starts decreasing,
    # stop training to avoid over fitting
    validation_accuracy = sess.run(accuracy, feed_dict={X: valid.images, Y: valid.labels})
    print(validation_accuracy)
    if (validation_accuracy - previous_validation_accuracy) < 0:
        break
    previous_validation_accuracy = validation_accuracy

# Run a test with the test data set to get a confirmation of the accuracy
# of the network, and then start using it to classify unlabeled images
test = ds.test
print(sess.run(accuracy, feed_dict={X: test.images, Y: test.labels}))
