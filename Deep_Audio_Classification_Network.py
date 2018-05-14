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
10 epochs of training.
"""

# ==== PREPARING THE DATA ====
# Download the Music and Speech data set inside a dst directory
dst = 'gtzan_music_speech'
if not os.path.exists(dst):
    dataset_utils.gtzan_music_speech_download(dst)

print('==== Preparing audio files for training... ====')
# Let's get the list of all the music and speech wave files
# Get the full path to the music directory
music_dir = os.path.join(os.path.join(dst, 'music_speech'), 'music_wav')
# Now use list comprehension to combine the path of the directory with any wave files
music = [os.path.join(music_dir, file_i) for file_i in os.listdir(music_dir) if file_i.endswith('.wav')]

# Similarly, for the speech folder:
speech_dir = os.path.join(os.path.join(dst, 'music_speech'), 'speech_wav')
speech = [os.path.join(speech_dir, file_i) for file_i in os.listdir(speech_dir) if file_i.endswith('.wav')]


# FFT = Fast Fourier Transform (implementation algorithm of the Discrete Fourier Transform)
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


def prepare_audio_file(wav_file, label=None, fft_size=512, hop_size=256, n_frames=43, frame_hops=21):
    """
    Prepares and audio file by splitting it into frames of length fft_size,
    and by taking their Discrete Fourier Transforms. Frames are then grouped
    together in batches of n_frames (corresponding to 500ms of audio), sliding
    the window by 250ms (frame_hops) at a time. The label corresponding to
    each batch (music: 0 or speech: 1) is kept in the labels variable.
    :param wav_file: path to the .wav file to prepare
    :param label: the label associated to the .wav, if it is a labeled file (music: 0, speech: 1)
    :param fft_size: length of each frame to split the input and apply the DFT
    :param hop_size: by how many points we move every time we extract a frame from the input
    :param n_frames: length of the sliding window (i.e. the number of frames to take to
    make a batch of 500ms of audio)
    :param frame_hops: by how many ms we move our sliding window over the frames list every
    time we make a new 500ms batch
    :return: magnitudes array containing the magnitudes corresponding to the sliding windows
    (i.e. arrays containing 500ms worth of audio in the form of multiple frames of magnitudes),
    and labels array containing the labels corresponding to each sliding window.
    """
    s = utils.load_audio(wav_file)
    # Split into frames and take their Discrete Fourier Transforms:
    re, im = dft.dft_np(s, fft_size=fft_size, hop_size=hop_size)
    # Convert the complex representation (cartesian) to magnitudes/phases (polar representation):
    mag, phs = dft.ztoc(re, im)
    # This is how many sliding windows we have
    n_hops = (len(mag) - n_frames) // frame_hops
    # Let's extract them all:
    magnitudes, labels = [], []
    for hop_i in range(n_hops):
        # Get the current sliding window
        frames = mag[(hop_i * frame_hops):(hop_i * frame_hops + n_frames)]
        # We'll take the log magnitudes, as this is a nicer representation:
        this_X = np.log(np.abs(frames[..., np.newaxis]) + 1e-10)
        # Append this group of frames (corresponding to 500ms of audio) to the magnitudes matrix
        magnitudes.append(this_X)
        # And be sure that we store the correct label of this observation (0: music, 1: speech)
        labels.append(label)
    return magnitudes, labels


# Store every 500ms-magnitudes-group and its label being music: 0 or speech: 1
Xs, ys = [], []
audio_files = [music, speech]
for label, files in enumerate(audio_files):
    for i in files:
        magnitudes, labels = prepare_audio_file(i,
                                                label=label,
                                                fft_size=fft_size,
                                                hop_size=hop_size,
                                                n_frames=n_frames,
                                                frame_hops=frame_hops
                                                )
        Xs += magnitudes
        ys += labels

Xs = np.array(Xs)
ys = np.array(ys)

"""
# Let's plot the first magnitude matrix (43 horizontal lines of magnitudes, each corresponding
# to a 256-long frame. 43 frames, corresponds to 500ms). This is equivalent to 43 diagrams showing
# 256 values of magnitude, only here we use colored lines to plot all 43 diagrams at once
plt.imshow(Xs[0][..., 0])
plt.title('label:{}'.format(ys[0]))
plt.show()
"""

# The shape of our input to the network
n_observations, n_height, n_width, n_channels = Xs.shape

# Create a data set object from the data and the labels, to split the data into
# training, validation and testing subsets
ds = datasets.Dataset(Xs=Xs, ys=ys, split=[0.8, 0.1, 0.1], one_hot=True)

"""
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
"""

# ==== CREATING THE NETWORK ====
print('==== Creating the neural network... ====')
# Create the input to the network. This is a 4-dimensional tensor
# Use None as a shape for the first dimension (for use with different batch sizes)
# We are using sliding windows of our magnitudes as input
X = tf.placeholder(name='X', shape=[None, n_height, n_width, n_channels], dtype=tf.float32)

# Create the output to the network.  This is our one hot encoding of 2 possible values
Y = tf.placeholder(name='Y', shape=[None, 2], dtype=tf.float32)

# Define the number of layers and the number of filters in each layer
n_filters = [9, 9, 9, 9]

# Now let's loop over our n_filters and create the deep convolutional neural network
H = X
for layer_i, n_filters_i in enumerate(n_filters):
    # Create our connection to the next layer:
    H, W = utils.conv2d(H, n_filters_i, k_h=3, k_w=3, d_h=2, d_w=2, name=str(layer_i))

    # And use a non linearity
    H = tf.nn.relu(H)

# Connect the last convolutional layer to a fully connected layer
fc, _ = utils.linear(H, 100, name="fully_connected_layer_1")

# And another fully connected layer, now with just 2 outputs, the number of outputs that our
# one hot encoding has. A SoftMax activation is necessary to output a categorical probability distribution
Y_predicted, _ = utils.linear(fc, 2, activation=tf.nn.softmax, name="fully_connected_layer_2")


# ==== TRAINING THE NETWORK ====
print('==== Training the network ====')
# Cost function (measures the average loss of the batches)
loss = utils.binary_cross_entropy(Y_predicted, Y)
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

# Measure of accuracy to monitor training
predicted_y = tf.argmax(Y_predicted, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

n_epochs = 15
batch_size = 200

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Now iterate over our data set n_epoch times
for epoch_i in range(n_epochs):
    print('==== EPOCH ', epoch_i, ' ====')

    # Train with mini batches
    this_accuracy = 0
    iterations = 0
    for Xs_i, ys_i in ds.train.next_batch(batch_size):
        # Run the optimizer to train the network's parameters
        this_accuracy += sess.run([accuracy, optimizer], feed_dict={X: Xs_i, Y: ys_i})[0]
        iterations += 1
        print('Epoch:', epoch_i, ' Batch:', iterations - 1, ' Accuracy:', this_accuracy / iterations)
    print('Training accuracy (mean of the accuracies on the batches): ', this_accuracy / iterations)

    # Validation : see how the network does on data that is not used for training
    this_accuracy = 0
    iterations = 0
    for Xs_i, ys_i in ds.valid.next_batch(batch_size):
        # We are not running the optimizer here, only measuring the accuracy
        this_accuracy += sess.run(accuracy, feed_dict={X: Xs_i, Y: ys_i})
        iterations += 1
    print('Validation accuracy (data not used to train the network): ', this_accuracy / iterations)

# Test : estimate the network's accuracy when generalizing to unseen data after training
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
# Input an unlabeled sound file and have the network classify it between music (0) or speech (1)


def classify_unlabeled_audio_file(wav_file):
    """
    Takes an audio file as input, and uses the trained network
    to classify it (either containing music: 0, or speech: 1)
    :param wav_file: name of the .wav file to classify from the
    gtzan_music_speech/music_speech/unlabeled_wav folder.
    :return: the label that the network associates to the wav_file
    """
    wav_file = os.path.join(os.path.join(os.path.join(dst, 'music_speech'), 'unlabeled_wav'), wav_file)
    magnitudes, _ = prepare_audio_file(wav_file,
                                       fft_size=fft_size,
                                       hop_size=hop_size,
                                       n_frames=n_frames,
                                       frame_hops=frame_hops
                                       )
    return sess.run(tf.argmax(tf.reduce_mean(Y_predicted, 0), 0), feed_dict={X: magnitudes})


print(classify_unlabeled_audio_file('EricJohnson-CliffsOfDover.wav'))
