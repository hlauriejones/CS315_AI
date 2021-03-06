README.txt
Neuralnetwork.py
Laurie Jones

Elements of Learning: 
The target function is a function that accurately matches the images from the input
and classifies them under the lables of one of the 10 digits. 

The training data is x_train and y_train, a reshaped and normalized collection of
handwritten numbers in the size of (6000, 784) and (60000) respectively. 

Because this is a multilayer perceptron the hypothesis set is any smooth, non-linear 
function.

The learning algorithm is a feed forward neural network. 

The final hypothesis is an idnetificaiton of the number on the the image that is input. 


Summary: 
The data was imported as a grayscale. They are then reshaped from a three dimensional matrix
to a two dimensional vector. This helps to have an easier input into the network and to set 
up the labels to the values. 
They are then normalized using the Standard Scaler from scikit learn. This We fit the data with 
the x_train data to compute the average and standard deviation before calling it. We then use 
.fit to initialize the data so that it centers the data and scales it accordingly. Then that 
scaler is used to transform the x_train and x_test data because that has all of the values we 
are trying to match to the identifications. This does a standardization by centering and scaling 
the data we are actually using. It treats all NaN values as not a part of the data are not used
for fit and then held in transform. 
The type was then changed to integers because the y_train had blank spaces in the data therefore 
its type was object. This changes all of the blanks to zeroes. 
All of the intial conditions stipulated in the assignment were put into a scikit learn multi-layer 
perceptron classifier or MLP Classifier from sci-kit learn. The network was then trained on the x_train 
and y_train data with .fit. It was then used to create y_predict which test out the accuracy of the
data. 
Then the accuracy of the network is shown in network.score() which reveals it to be 96% affective. 
The loss curve is then plotted with netowrk.loss_curve_. This displays the over all error. This model 
optimizes the log-loss function using stochastic gradient descent.
A confusion matrix is then calculated with Confusion_matrix. It uses the y_test and the y_predict to 



Machine Learning Concepts: 
perceptron: an algorithm used for supervised learning of binary classifiers this works well here because 
    it either says that the image is correct or not. 
hidden layer: located between the input and output of the algorithm, in which the function applies weights 
    to the inputs and directs them through an activation function as the output.
relu activation function: a "rectified linear activation function".  Its output is the input directly if 
    it is positive, otherwise, it will output zero. This  overcomes the vanishing gradient problem, allowing 
    models to learn faster and perform better. This is particulary great to assure good results for a feed 
    forward neural network such as this one. 
stochastic gradient descent: this loss function works by calculating the derivative from each training data 
    instance and calculating the update immediately. This helps to minimize the valleys that the gradient
    descent might get caught in. Here it is specifically used to optimize the log-loss function for optimal 
    identification. 
the log-loss function: also known as the cross-entropy function measures classification model whose output 
    is a probability value between 0 and 1. This increases as the predicted probability diverges from the 
    actual label.
confusion matrix: This is used to create a visual of the quality of the output of a classifier. The diagonal 
    elements represent the number of points for which the predicted label is equal to the actual label, while 
    the other elements are mislabeled by the classifier. The higher the diagonal values of the confusion matrix 
    the more correct the network is. 
 

Websites: 
https://keras.io/api/datasets/mnist/ 
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html




