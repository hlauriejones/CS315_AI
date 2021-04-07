README.md

CSCI 315 – Assignment 6 CNNs
Laurie Jones and James Lawson
April 3rd 2021 


5 Elements of Learning:

1. Target function
The target function is the function that accurately assigns the important parts to the image and is able to differentiate the parts of an image as well as images from other images

2. Training Data
The dataset is the CIFAR10 dataset for image identification. It has 60,000 32x32 color images. And 10 different types of images to decide against. 

3. Hypothesis Set
In a multi-layered perception, the hypothesis set is any smooth, non-linear 
function.

4. Learning Algorithm
The learning algorithm that we are using is a convolutional neural network.

5. Final Hypothesis
The final hypothesis is the identification of the features of the images that is inputed and then the labeling of said image. 


Summary:
Create one hot encoding vectors for y train and y test. This is necessary because one-hot encoding can represent these images in a unique way that can then be referenced later. 

Then we normalize the images because currently the numbers are large and representative of RGB values and we need to put them as as a spectrum between 0 and 1. 

Then we create our model as a sequential model because an CNN looks at data in a sequential way. 

Then we create convolutional layers because we need to capture spatial and temporal dependencies in an image. A convolutional layer is a layer that applies a filter to the 
Input result in a feature map. 

The feature that we apply is batch normalization. Batch normalization is good because it regularizes that the input features. This is good so that we can figure out which features are the most important. 

Then after a set of convolutional layers and batch normalization we use a max pooling layer. This is helpful because it pulls together everything and synthesizes the features with down sampling the image. 

Then after all of this we use a dropout layer to make sure that we don't have overfitting in our model. And give each node a certain percentage of being included (ex 20%). 

We then do early stop training in order to make sure that we stop training our model at the best possible point. That way our model isn't over trained. 

Then we compile it and then make augmented data. 
We augment the data to create adversarial training on our model . 


Why we do the structure: 
Throughout the whole model the pooling layers get smaller kernels so that we can figure out which are the best features at each point and then draw conclusions about higher level features. 

We go up in number of filters to extract higher level features. 



Ideas from class: 
One-hot encoding -  takes the categorical variables (1,2,...9) and represents them as binary vectors. So this means that each image is represented as a binary vector where all the values are zero except the index corresponding to that integer, that is labled as one.

Dropout - We drop weights of the network out of contention. This can be done randomly with a probability on a node-to-node basis

Batch normalization - zero-centers and normalizes each input, then scales and shifts the result using two new parameter vectors per layer (One for scaling, One for shifting). Then
The model learns the optimal scale and mean for each layer's inputs. 

Convolutional layer - a layer that applies a filter to the input result in a feature map.  

Max pooling layer - is when we bring the "highest" feature from the convolutional layer before and put it in a new layer. 

Kernel - is the space that we use to represent the features

he_uniform - Draws samples from a uniform distribution within [-limit, limit] , where limit = sqrt(6 / fan_in) ( fan_in is the number of input units in the weight tensor).



Sources:
https://www.tensorflow.org/tutorials/images/cnn 
https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://towardsdatascience.com/image-augmentation-for-deep-learning-using-keras-and-histogram-equalization-9329f6ae5085

