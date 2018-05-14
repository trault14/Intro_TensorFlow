# Introduction to TensorFlow

The following deep learning projects were realized as homework assignements of the following tutorial : [Creative Applications of Deep Learning Using TensorFlow](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info)
The main library used for developping these algorithms is Google's TensorFlow.

## Installing
The installation procedure will allow you to get the algorithms running on your machine, in order to run/tweak them.

Start by cloning the repository on your disk :
```
git clone https://github.com/trault14/Intro_TensorFlow
```
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
We can now run any of the python scripts as follows :
Open an interactive Python3 shell
```
ipython3
```
Run the desired .py script :
```
%run ./Deep_Audio_Classification_Network.py
```
