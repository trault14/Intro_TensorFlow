# Introduction to TensorFlow

The following deep learning projects were realized as homework assignments of the following tutorial : [Creative Applications of Deep Learning Using TensorFlow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info)
The main library used for developping these algorithms is Google's TensorFlow.
## Table of Contents
1. [Installing](#installing)
2. [Deep Convolutional Audio Classification Network](#deep-audio-classification-network)
3. [Convolutional Auto Encoder](#convolutional-auto-encoder)
4. [Fully Connected Auto Encoder](#fc-auto-encoder)
5. [Fully Connected Image Classification Network](#fc-image-classification)
6. [Fully Connected Network to paint an image](#fc-image-painting)
## Installing
The installation procedure will allow you to get the algorithms running on your machine, in order to run/tweak them.

Start by cloning the repository on your disk :
```
git clone https://github.com/trault14/Intro_TensorFlow
```
### Setting up a virtual environment
I recommend using a python virtual environment for managing the dependencies. We'll need to install virtualenv :
```
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
```
Let's now create a Python 3 virtual environment called 'venv' dedicated to the Intro_TensorFlow project. We'll create it inside the project's folder.
```
cd Intro_TensorFlow
virtualenv -p python3 venv
```
We can now activate the virtual environment :
```
source venv/bin/activate
```

Install the following dependencies inside the virtual environment:
```
sudo apt install ipython3
sudo apt-get install python3-tk
pip3 install IPython
pip3 install numpy
pip3 install tensorflow
pip3 install scipy
pip3 install matplotlib
pip3 install pillow
```
### Running an algorithm
We can now run any of the python scripts as follows :
Open an interactive Python3 shell
```
ipython3
```
Run the desired .py script :
```
%run ./Deep_Audio_Classification_Network.py
```
## Deep Convolutional Audio Classification Network <a name="deep-audio-classification-network"></a>
This supervised convolutional neural network aims at classifying audio files into two categories : either containing music, or speech. 
The data set used to train the network (GTZAN Music and Speech) contains 64 music files and 64 speech files, each 30 seconds long and at a sample rate of 22050Hz (22050 samplings of audio per sec.). 
The input to the network is the audio data, and the output is a one-hot encoding of the probability distribution [music, speech]. 
The analysis of the audio files relies on the use of the Fourier transform. 
The network is capable of reaching a final test accuracy on unseen data of about 97% after approximately 10 epochs of training.

In its current version the network contains 4 convolutional layers of 9 filters each, all using a ReLU non-linear activation function. The stride is 2x2, so no Max Pooling layers are used. This is then followed by two fully connected layers of respectively 100 and 2 neurons. The last layer uses a SoftMax activation function in order to output a categorical probability distribution [music, speech].

### Instructions
In order to start the training of the network, simply launch an ipython shell and run the Deep_Audio_Classification_Network.py file :
```
ipython3
%run ./Deep_Audio_Classification_Network.py
```
Once the training process is complete, you can use the network to classify an unlabeled audio file from the folder gtzan_music_speech/music_speech/unlabeled_wav. Simply use the following function :
```
result = classify_unlabeled_audio_file('EricJohnson-CliffsOfDover.wav')
print(result)
```
Label 0 corresponds to music, while label 1 corresponds to speech.

## Convolutional Auto Encoder <a name="convolutional-auto-encoder"></a>
This Auto Encoder Neural Network takes a 28 by 28 (784 values) pixel image (or a batch of images) as its input. 
It learns to encode it down to 7 feature maps of 7x7 values, and to retrieve the original image as its output, only using the  encoded values.
The network is composed of two sub-networks : the encoder, which is used to compress the image data, and the decoder, which allows to retrieve the original images from the compressed data.
This convolutional Neural Network learns in an unsupervised fashion as the input data is unlabeled.

In its current version, the encoder network uses three convolutional layers, each using 16 4x4 filters and a ReLU activation function. No pooling layer is used because the stride is 2x2.
The decoder network simply mirrors the encoder.

### Instructions
Start the training of the network by lauching an ipython shell and running the Autoencoder_Convolutional.py file :
```
ipython3
%run ./Autoencoder_Convolutional.py
```

## Fully Connected Auto Encoder <a name="fc-auto-encoder"></a>
This Auto Encoder Neural Network takes a 28 by 28 (784 values) pixel image (or a batch of images) as its input. It learns to encode it down to 64 values, and to retrieve the original image as its output, only using the encoded values. 
It learns in an unsupervised fashion as the input data is unlabeled.

The architecture of the encoder network relies on 4 Fully Connected layers that get progressively smaller : 512, 256, 128 and finally 64. The decoder network simply mirrors the encoder : 128, 256, 512 and finally 784 (output layer).
### Instructions
Start the training of the network by lauching an ipython shell and running the Autoencoder_Convolutional.py file :
```
ipython3
%run ./Autoencoder_Fully_Connected.py
```

## Fully Connected Image-Classification network <a name="fc-image-classification"></a>
This fully connected neural network is trained to classify images that each represent the drawing of a number from 0 to 9.
The input to the network is an image of 28x28 pixels (784 input neurons).
The output is a 10-digit one-hot encoding containing the probability that the network associates to each possible label for classifying the input image. 
Supervised learning is used here, as the cost function compares the predicted output with the true output by means of a cross entropy function.

In its current form, the network is composed of a single fully connected layer of 10 output neurons. The network reaches an accuracy level of about 93% in 4 epochs.
### Instructions
Start the training of the network by lauching an ipython shell and running the Autoencoder_Convolutional.py file :
```
ipython3
%run ./Classification_Fully_Connected.py
```
Use the network by feeding it a single image, and check whether or not the prediction is correct by plotting also the image :
```
print(sess.run(predicted_y, feed_dict={X: [ds.X[3]]})[0])
plt.imshow(ds.X[3].reshape((28, 28)))
plt.show()
```
## Fully Connected Network to paint an image <a name="fc-image-painting"></a>
This neural network implements a fun application : painting and image. The input to the network is a position on the 
image, X = (row, col). The output is the color to paint, Y = (R, G, B). The network is of type fully connected and is composed of 6 hidden layers each containing 64 neurons. 
The network relies on a supervised type of learning, given that each input position (x, y) is accompanied by its corresponding (r, g, b) label in the picture.
### Instructions
Start the training of the network for it to begin painting the input image :
```
ipython3
%run ./Painting.py
```
