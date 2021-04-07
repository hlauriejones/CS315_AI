"""
Author: Laurie Jones
File: neuralnetwork.py

max iterations= 50
alpha= 1e-5.
learning rate= 0.01
random state= 1
verbose = 10

stochastic gradient descent 
relu activation function

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
import mnist
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix

print("getting data....")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


# Flatten the images.
def flatten(set):
    return set.reshape((-1, 784))

print("reshaping...")
x_train = flatten(x_train)
x_test = flatten(x_test)

#Normalizing the data
print("normalizing....")
scaler = StandardScaler() 
scaler.fit(x_train) 
StandardScaler(copy=True, with_mean=True, with_std=True) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#changing the type because you need to fit it 
print("changing type...")
x_train = x_train.astype('int')
y_train = y_train.astype('int')
x_test = x_test.astype('int')
y_test = y_test.astype('int')

#create your network
print("making the network...")
network = MLPClassifier(solver = 'sgd', max_iter=50, activation = 'relu',alpha = 1e-5, 
learning_rate_init = 0.01, random_state=1, verbose = 10)
network.fit(x_train, y_train)
y_predict = network.predict(x_test)

#predict the accuracy
score = network.score(x_test, y_test)
print("Accuracy: ", score, "%")

#display a graph of the overall error of the model
plt.plot(network.loss_curve_)

#generate a confusion matrix of the test data
print("plotting...")
disp = plot_confusion_matrix(network, x_test, y_test,
                                 cmap=plt.cm.Blues,)
disp.ax_.set_title("MNIST Dataset")
print(disp.confusion_matrix)
plt.show()
