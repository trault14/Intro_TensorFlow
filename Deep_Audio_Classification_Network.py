import os
import numpy as np
import tensorflow as tf
from libs import dataset_utils
from libs import utils
from libs import dft
from libs import datasets
import matplotlib.pyplot as plt


# ==== DESCRIPTION ====
"""
This supervised convolutional neural network aims at classifying 
audio files into two categories : either music or speech. The data set
used to train the network (GTZAN Music and Speech) contains
64 music files and 64 speech files, each 30 seconds long and
at a sample rate of 22050Hz (22050 samplings of audio per sec.).
The input to the network is the audio data, and the output is
a one-hot encoding of the probabilities [music, speech].
The analysis of the audio files relies on the use of the 
Fourier transform. The network is capable of reaching a final
test accuracy on unseen data of about 97% after approximately
8 epochs of training.
"""

# ==== PREPARING THE DATA ====
# Download the Music and Speech data set inside a dst directory
dst = 'gtzan_music_speech'
if not os.path.exists(dst):
    dataset_utils.gtzan_music_speech_download(dst)


# Let's get the list of all the music and speech wave files
# Get the full path to the music directory
music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')
# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i) for file_i in os.listdir(music_dir) if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i) for file_i in os.listdir(speech_dir) if file_i.endswith('.wav')]

# Print the file names
print(music, speech)

# Store every magnitude frame and its label being music: 0 or speech: 1
Xs, ys = [], []

fft_size = 512
hop_size = 256

# The sample rate from our audio is 22050 Hz.
sr = 22050

# We can calculate how many hops there are in a second
# which will tell us how many frames of magnitudes
# we have per second
n_frames_per_second = sr // hop_size

# We want 500 milliseconds of audio in our window to determine
# whether the file contains speech or music
n_frames = n_frames_per_second // 2

# And we'll move our window by 250 ms at a time
frame_hops = n_frames_per_second // 4

# Let's start with the music files
for i in music:
    # Load the ith file:
    s = utils.load_audio(i)

    # Now take the Discrete Fourier Transform of it:
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)

    # And convert the complex representation (cartesian) to magnitudes/phases (polar representation):
    mag, phs = dft.ztoc(re, im)

    # This is how many sliding windows we have:
    n_hops = (len(mag) - n_frames) // frame_hops

    # Let's extract them all:
    for hop_i in range(n_hops):
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]

        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)

        # And store it:
        Xs.append(this_X)

        # And be sure that we store the correct label of this observation:
        ys.append(0)

# Now do the same thing with the speech files
for i in speech:
    s = utils.load_audio(i)
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)
    mag, phs = dft.ztoc(re, im)
    n_hops = (len(mag) - n_frames) // frame_hops

    for hop_i in range(n_hops):
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)
        Xs.append(this_X)
        ys.append(1)

# Convert them to an array:
Xs = np.array(Xs)
ys = np.array(ys)

print(Xs.shape, ys.shape)

# Just to make sure you've done it right.  If you've changed any of the
# parameters of the dft/hop size, then this will fail.  If that's what you
# wanted to do, then don't worry about this assertion.
assert (Xs.shape == (15360, 43, 256, 1) and ys.shape == (15360,))

# Let's plot the first magnitude matrix
plt.imshow(Xs[0][..., 0])
plt.title('label:{}'.format(ys[0]))
plt.show()

# The shape of our input to the network
n_observations, n_height, n_width, n_channels = Xs.shape

# Create a data set object from the data and the labels
ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True)

Xs_i, ys_i = next(ds.train.next_batch())
# Notice the shape this returns.  This will become the shape of our input and output of the network:
print(Xs_i.shape, ys_i.shape)
assert(ys_i.shape == (100, 2))

# Let's take a look at the first element of the randomized batch
plt.imshow(Xs_i[0, :, :, 0])
plt.title('label:{}'.format(ys_i[0]))
plt.show()

# And the second one
plt.imshow(Xs_i[1, :, :, 0])
plt.title('label:{}'.format(ys_i[1]))
plt.show()

# ==== CREATING THE NETWORK ====
tf.reset_default_graph()

# Create the input to the network. This is a 4-dimensional tensor
# Don't forget that we should use None as a shape for the first dimension
# Recall that we are using sliding windows of our magnitudes
X = tf.placeholder(name='X', shape=[None, n_height, n_width, n_channels], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values
Y = tf.placeholder(name='Y', shape=[None, 2], dtype=tf.float32)

# Define the number of layers and the number of filters in each layer
n_filters = [9, 9, 9, 9]

# Now let's loop over our n_filters and create the deep convolutional neural network
H = X
for layer_i, n_filters_i in enumerate(n_filters):
    # Let's use the helper function to create our connection to the next layer:
    H, W = utils.conv2d(H, n_filters_i, k_h=3, k_w=3, d_h=2, d_w=2, name=str(layer_i))

    # And use a non linearity
    H = tf.nn.relu(H)

    # Just to check what's happening:
    print(H.get_shape().as_list())

# Connect the last convolutional layer to a fully connected network
fc, W = utils.linear(H, 100, name="fully_connected_layer_1")

# And another fully connected layer, now with just 2 outputs, the number of outputs that our
# one hot encoding has
Y_predicted, W = utils.linear(fc, 2, activation=tf.nn.softmax, name="fully_connected_layer_2")

# ==== TRAINING THE NETWORK ====
# Cost function (measures the average loss of the batches)
loss = utils.binary_cross_entropy(Y_predicted, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

# Measure of accuracy to monitor training
predicted_y = tf.argmax(Y_predicted, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

n_epochs = 10
batch_size = 200

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Now iterate over our data set n_epoch times
for epoch_i in range(n_epochs):
    print('==== EPOCH ', epoch_i, ' ====')

    # Train
    this_accuracy = 0
    iterations = 0

    # Do our mini batches:
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        # Note here: we are running the optimizer so
        # that the network parameters train!
        this_accuracy += sess.run([accuracy, optimizer], feed_dict={X: Xs_i, Y: ys_i})[0]
        iterations += 1
        print('Epoch:', epoch_i, ' Batch:', iterations - 1, ' Accuracy:', this_accuracy / iterations)
    print('Training accuracy (mean of the accuracies on the batches): ', this_accuracy / iterations)

    # Validation : see how the network does on data that is not used for training
    this_accuracy = 0
    iterations = 0
    for Xs_i, ys_i in ds.valid.next_batch(batch_size):
        # Note here: we are NOT running the optimizer!
        # we only measure the accuracy!
        this_accuracy += sess.run(accuracy, feed_dict={X: Xs_i, Y: ys_i})
        iterations += 1
    print('Validation accuracy (data not used to train the network): ', this_accuracy / iterations)

# Test : estimate the network's accuracy to generalize to unseen data after training
this_accuracy = 0
iterations = 0
for Xs_i, ys_i in ds.test.next_batch(batch_size):
    this_accuracy += sess.run(accuracy, feed_dict={X: Xs_i, Y: ys_i})
    iterations += 1
print('==== TEST ACCURACY (on unseen data, after training): ', this_accuracy / iterations, ' ====')

# ==== INSPECTING THE NETWORK ====
# Let's show the convolution filters (W tensors) learned by the network in each convolution layer
g = tf.get_default_graph()
for layer_i in range(len(n_filters)):
    W = sess.run(g.get_tensor_by_name('{}/W:0'.format(layer_i)))
    plt.figure(figsize=(5, 5))
    plt.imshow(utils.montage_filters(W))
    plt.title('Layer {}\'s Learned Convolution Kernels'.format(layer_i))
    plt.show()

# ==== USING THE NETWORK ====
# Input a sound file and have the network classify it between music (0) or speech (1)


def classify_audio_file(wav_file):
    s = utils.load_audio(wav_file)
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)
    mag, phs = dft.ztoc(re, im)
    n_hops = (len(mag) - n_frames) // frame_hops
    Xs = []
    for hop_i in range(n_hops):
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)
        Xs.append(this_X)
    Xs = np.array(Xs)
    return sess.run(tf.argmax(tf.reduce_mean(Y_predicted, 0), 0), feed_dict={X: Xs})


unlabeled_dir = os.path.join(os.path.join(dst, 'music_speech'), 'unlabeled_wav')
wav_file = os.path.join(unlabeled_dir, 'EricJohnson-CliffsOfDover.wav')
print(classify_audio_file(wav_file))
