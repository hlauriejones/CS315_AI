"""
Author: Laurie Jones
File: homework4.py
"""

#imports
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

#fashion data is imported
print("getting data...")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#scale the values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#weights initialized 
initializer = tf.keras.initializers.HeNormal()
print("done")

#creating class names
class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

#we create our model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #input
    keras.layers.BatchNormalization(momentum = 0.9), #normalization
    keras.layers.Dense(300, activation = "elu", kernel_initializer = initializer), # 300 hidden layer
    keras.layers.BatchNormalization(momentum = 0.9), #normalization
    keras.layers.Dense(100, activation = "elu", kernel_initializer = initializer), # 100 hidden layer
    keras.layers.BatchNormalization(momentum = 0.9), #normalization
    keras.layers.Dense(10, activation = "softmax") #output
])

#showing the summary
model.summary()

#stochastic gradient descent
opt = SGD(lr = 0.01, clipnorm = 1)

#model is compiled
model.compile(optimizer = opt,
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

#model is fit
model.fit(train_images, train_labels, epochs = 200)

#model is evaluated
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Loss:", test_loss)
print("Accuracy:", test_acc)




