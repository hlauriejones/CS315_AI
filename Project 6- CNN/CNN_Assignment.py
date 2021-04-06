"""
File: CNN_Assignment.py
Authors: Laurie Jones and James Lawson

Links:
https://www.tensorflow.org/tutorials/images/cnn
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085

"""

import numpy as np
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
#added these imports
from tensorflow.keras import models
import tensorflow as tf
import matplotlib.pyplot as plt

## Load is Cifar 10 dataset
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

## Print ouf shapes of training and testing sets
print("Shape of training data: ", train_data.shape, train_labels.shape)
print("Shape of testing data: ", test_data.shape, test_labels.shape)

## Normalize x train and x test images
train_data = train_data / 255
test_data = test_data / 255

## Create one hot encoding vectors for y train and y test
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## Define the model
model = models.Sequential()

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape=(32, 32, 3)))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding 
model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D((2, 2)))

## Add dropout layer of 0.2
model.add(Dropout(0.2))

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D((2, 2)))

## Add dropout layer of 0.2
model.add(Dropout(0.2))

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
model.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add a max pooling 2d layer with 2x2 size
model.add(MaxPooling2D((2, 2)))

## Add dropout layer of 0.2
model.add(Dropout(0.2))

## Flatten the resulting data
model.add(Flatten())

## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))

## Add a batch normalization layer
model.add(BatchNormalization())

## Add dropout layer of 0.2
model.add(Dropout(0.2))

## Add a dense softmax layer
model.add(Dense(10, activation='softmax'))

## Set up early stop training with a patience of 3
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "loss",
    min_delta = 0.0001,
    patience = 3)

## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])

# Image Data Generator, we are shifting image accross width and height of 0.1 also we are flipping the image horizantally and rotating the images by 20 degrees
data_augmentation = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 20, horizontal_flip = True)

## Take data and label arrays to generate a batch of augmented data, default parameters are fine. 
batch = data_augmentation.flow(train_data, train_labels)

## Define the number of steps to take per epoch as training examples over 64
steps_per_epoch = len(train_data) / 64

## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined. 

model.summary()

final_model = model.fit(batch,
                        steps_per_epoch = steps_per_epoch,
                        epochs = 200,
                        validation_data = (test_data, test_labels),
                        callbacks = [early_stop])

test_results = model.evaluate(test_data, test_labels)

print("Accuracy: ", test_results[1])

#Show graph of loss over time for training data
plt.plot(final_model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data'], loc = 'upper left')
plt.show()

#Show graph of accuracy over time for training data
plt.plot(final_model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data'], loc = 'upper left')
plt.show()





