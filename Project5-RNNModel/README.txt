README.txt
Laurie Jones and James Lawson
Saturday, March 27th 2021
Assignment 5: Deep Recurrent Neural Network - Model

5 Elements of Learning:

1. Target function

The target function maps input text to _______.

2. Training Data

The dataset is a csv file of companies’ responses regarding the financial services provided by the United States.

3. Hypothesis Set

In a multi-layered perception, we have non-linear functions.

4. Learning Algorithm

The learning algorithm that we are using is a recurrent neural network.

5. Final Hypothesis

The final hypothesis is a prediction of which words come before and after our target word????.


General Description:

First we had to preprocess the data......



We start by loading in the MNIST digits classification dataset. This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. This line of code returns a tuple of Numpy arrays: x_train | y_train and x_test | y_test. The x_train and x_test are arrays of grayscale image data. The y_train and y_test are arrays of digit labels (integers in range 0-9).

Next, we have to flatten each image. Each image in the MNIST dataset is 28x28 and contains a centered, grayscale digit. We flattened each 28x28 into a 784 dimensional vector. This is used as the input to the neural network.

Next, we get rid of all empty values. We substituted 0s in for any empty values.

Next, we used the standard scalar to normalize the training and testing data (Note: we only normalize the images, not the labels. Additionally, if you are going to normalize data, you must normalize both the training and the testing data). If the data is not normalized, the model may not be sufficiently trained by the time we reach the number of max iterations. A way to solve this is by scaling the data. More on this later.

Next, we create out model. We created out model using a multi-layer perceptron classifier. Our model in particular uses the relu activation function with stochastic gradient descent as the optimization algorithm. We use a learning rate of 0.01, we define the max iterations to be 50, we define the verbose parameter to be 10, we define the random state to be 1, and we define alpha to be 0.00001. More on this later.

Next, we fit the model using the training data and graphed the loss.

Finally, we printed a confusion matrix and a classification report of our model.


Detailed description of the machine learning concepts implemented in the project:

Feature Scaling: To accomplish feature scaling, we standardized the features by removing the mean and scaling to unit variance. The centering and scaling are calculated on each individual feature by using the mean and the standard deviation from the training set. These values are used when "scalar.transform" is used in the preceding 2 lines of code.

Neural Network Model: The Neural Network Model works by iteratively training at each time step. It uses the partial derivatives of the loss function to update the parameters of the model.

Resources:

https://keras.io/api/

https://www.tensorflow.org/guide/keras/rnn

https://www.kite.com/python/answers/how-to-copy-columns-to-a-new-pandas-dataframe-in-python

https://datatofish.com/count-nan-pandas-dataframe/

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html

https://www.kite.com/python/answers/how-to-drop-empty-rows-from-a-pandas-dataframe-in-python

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html

https://docs.python.org/3/library/re.html#re.sub

https://www.geeksforgeeks.org/regular-expression-python-examples-set-1/

https://docs.python.org/3/howto/regex.html

https://docs.python.org/3/library/re.html#re.compile

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences

https://www.geeksforgeeks.org/python-pandas-get_dummies-method/

https://keras.io/api/callbacks/early_stopping/

https://keras.io/api/layers/recurrent_layers/lstm/
