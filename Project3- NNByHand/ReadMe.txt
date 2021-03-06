ReadMe.txt
Laurie Jones

Elements of Learning: 
The target function is a function that accurately matches the images from the input
and classifies them under the lables of one of the 10 digits. 

The training data is train_imgs and train_labels.
    train_imgs is the grayscale images from the mnist_train.csv. They are then mapped 
    between 0.01 and 0.99 (the test_imgs are done the same way) They are also taken from
    a three dimensional matrix and reshaped into to a two dimensional vector.
    train_labels takes the corresponding labels from each images and puts them into one-
    hot representation. This takes the categorical variables (1,2,...9) and represents
    them as binary vectors. So this means that each integer value is represented as a 
    binary vector where all teh values are zero except the intex corresponding to that
    integer, that is labled as one. 

Because this is a multilayer perceptron the hypothesis set is any smooth, non-linear 
function.

The learning algorithm is a feed forward neural network. 

The final hypothesis is an idnetificaiton of the number on the the image that is input. 


Summary: 
First the data is imported and scaled: 
The images are imported from the mnist_train.csv and mnist_test.csv. They are then scaled
and placed into train_imgs and test_imgs, train_labels and test_labels. The images are 
the grayscale images from the mnist_train.csv. They are then mapped between 0.01 and 0.99 (
They are also taken from a three dimensional matrix and reshaped into to a two dimensional 
vector. The lables take the corresponding labels from each images and puts them into one-
hot representation. This takes the categorical variables (1,2,...9) and represents them as 
binary vectors. So this means that each integer value is represented as a binary vector where 
all the values are zero except the intex corresponding to that integer, that is labled as one.
Thes lables are then scaled from 0.01 to 0.99 to correspond with the data it wa smapped to.

The netowrk is created: 
the input to create the network is the input images, the number of nodes in the hidden layers
and the number of nodes in the output layer. I also initialized my matrix by calling a 
"createWeights" function where i create the weights that correspond between the respective layers. 
Here i just initialize them as 1 over the square root of the nodes that are put in. This works 
to set the weights within a normal distribution. 

the network is trained:
when the network is trained, it takes in the input, imposes the transform and then goes through
the network, taking the dot product of the weights and the output from the previous layer. Then
the train function calculates the error by comparing the output to the goal labels and
the ouput that was calculated. Then to update the weights, a temporary variable is created the dot product 
is taken of the transform and the errors from the previous layer. The 
weights are then updated, adjusted to equal the error of the past weights and the impact of the last layer,
and then it uses the learning rate and the dot product of itself and teh transform of the previous layer
to undo what was done so that it could iterate backwards. The weights between each layer are then 
updated. This backwards movement and multiplying by the transform of the previous layer is the
what is back propogation. Think of this as the "learning" of the network where the results are 
compared to the actual labels and the network is adjusted accordingly

The network is run: 
Querey is how the network is actually run. This is the iteration through the network. Here the input
is transformed and then the dot product is taken of the weights and the layers that are on the previous
layer. Then the activation function, a sigmoid function proposed through scipy.special and expit. This 
is equated to a neuron "firing" in a layer and allows the network to equate what connections are being
made between the input and the labels. 

Calculate the accuracy:
the accuracy is calculated through a series of functions that correspond to the function "evaluate". 
This function goes through all of the data that is input into it and runs the network, it then collects
a corresponding number of identities labled as "correct" or "wrong" these are notifications about whether
or not an output got the correct label on it or not. This outputs the total number of correct vs incorrect
labels that the network has output. 

Other functions: 
I created a confusion matrix so that one could visualize the accuracy of the code.
Also there is the ability to rotate the images and run the netowrk on the rotated images instead of the 
original MNIST images. This can be added in addendum to the other images by saving them under a different
set of vectors and then appending them to the dataset available. This can help to create a larger set to
train and/or test the network on. Mine does not work very well but it is possible

Other notes (hyper parameters): 
due to analysis and running of the network in various forms, the best hyper parameters are around 200 layers because
that allows for the best outcome in the shortest amount of time, 5 epochs, because the the over specification 
makes the network over complicated. and the learning rate fo 0.01. this allows for the most accurate assessment
of the error. 

Concepts from Class: 
one-hot representation: This takes the categorical variables (1,2,...9) and represents them as 
binary vectors. So this means that each integer value is represented as a binary vector where 
all the values are zero except the intex corresponding to that integer, that is labled as one.
back propogation: This is the undoing of the activation function and the dot product that allow the network
to propogate through the network and then 
hyper parameters: are the parts of the network that can't be determined from the network such as the number of nodes
in each layer, the learning rate, the epochs, the use of momentum, ect. 


websites:
https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/WorkingWithFiles.html 
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
https://www.kite.com/python/docs/scipy.stats.poisson.rvs
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html 
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ 
https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
https://pythonexamples.org/python-pillow-rotate-image-90-180-270-degrees/
https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
