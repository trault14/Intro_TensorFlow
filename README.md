# Introduction to TensorFlow

The following deep learning projects were realized as homework assignements of the following tutorial : [Creative Applications of Deep Learning Using TensorFlow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info)
The main library used for developping these algorithms is Google's TensorFlow.

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
## Deep Convolutional Audio Classification Network
This supervised convolutional neural network aims at classifying audio files into two categories : either containing music, or speech. 
The data set used to train the network (GTZAN Music and Speech) contains 64 music files and 64 speech files, each 30 seconds long and at a sample rate of 22050Hz (22050 samplings of audio per sec.). 
The input to the network is the audio data, and the output is a one-hot encoding of the probability distribution [music, speech]. 
The analysis of the audio files relies on the use of the Fourier transform. 
The network is capable of reaching a final test accuracy on unseen data of about 97% after approximately 10 epochs of training.

In its current version the network contains 4 convolutional layers of 9 filters each, all using a ReLU non-linear activation function. This is then followed by two fully connected layers of respectively 100 and 2 neurons. The last layer uses a SoftMax activation function in order to output a categorical probability distribution [music, speech]

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
