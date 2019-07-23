# PV_Panel_Classify

It is slightly simplified implementation of Convolutional Neural Networks for Image Classification
Image Classification via CNN neural network implemented in tensorflow

# Key Requirements: 
* Python 3.5.3 
* Tensorflow 1.13.1
* Numpy 1.16.3

Suggestion: Better to download Anaconda as it will take care of most of the other packages and easier to setup a virtual workspace to work with multiple versions of key packages like python, tensorflow etc.

# Directory structure description
*./resize_photoes.py   # resize image size to 100*100
*./tmp  # Save the model and model parameters to view the training information using the tensorboard
*./0510/N  # put training data(images)--abnormal pictures
*./0510/P  # put training data(images)--normal pictures
*./0510_resize/N  # put resized training data(images)--abnormal pictures
*./0510_resize/P  # put resized training data(images)--normal pictures
*./valid  # put validation data(images)
