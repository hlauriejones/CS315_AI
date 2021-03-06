import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.special import expit as activation_function
from scipy import ndimage, misc



#get data
print("getting data....")
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
train_data = np.loadtxt("mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist_test.csv", delimiter=",") 
test_data[:10]

test_data[test_data==255]
print("done")

#map the values between 0.01 and 0.99 - they are already in grayscale
print("scaling...")
fac = 0.98 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


#turning labels into one-hot repreesntation
lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(int)
    #print("label: ", label, " in one-hot representation: ", one_hot)

lr = np.arange(no_of_different_labels)
train_labels_one_hot = (lr==train_labels).astype(float)
test_labels_one_hot = (lr==test_labels).astype(float)

# scaling labels
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

print("done")


# def activation_function(x):
#     return 1 / (1 + np.e ** -x)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)




class neuralNetwork:
      
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.hiddenLone = hiddennodes
        self.hiddenLtwo = hiddennodes
        self.learningrate = learningrate
        self.createWeights()

    #initialize the weight matrix
    print("initializing weights...")
    def createWeights(self):
        rad = 1 / np.sqrt(self.inputnodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wIH = X.rvs((self.hiddenLone, self.inputnodes))
        #self.weights_matrices.append(self.wIH)

        #weights from hidden --> hidden
        rad = 1 / np.sqrt(self.hiddenLone)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wHH= X.rvs((self.hiddenLone, self.hiddenLtwo))
        #self.weights_matrices.append(self.wHH)


        #weights from hidden --> output
        rad = 1 / np.sqrt(self.hiddenLtwo)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wHO = X.rvs((self.outputnodes, self.hiddenLtwo))
        #self.weights_matrices.append(self.wHO)
    print("done")

    
    # train the neural network
    #print("training...")
    def train(self, inputs_list, targets_list):
        inputs_list = np.array(inputs_list, ndmin=2).T
        targets_list = np.array(targets_list, ndmin=2).T
        
        outputs_list1 = np.dot(self.wIH, inputs_list) #dot product b/w input and hidden
        output_hiddenOne = activation_function(outputs_list1) #weights after 1st hidden layer

        outputs_list2 = np.dot(self.wHH, output_hiddenOne) #dot product b/w hidden and hidden
        output_hiddenTwo = activation_function(outputs_list2) #output after 2nd hidden layer
        
        outputs_list3 = np.dot(self.wHO, output_hiddenTwo) #dot product b/w hidden and output
        output_network = activation_function(outputs_list3) #output after output
        
        #calculate output errors
        output_errors = targets_list - output_network

        # update the second hidden weights:
        #moving backwards
        tmp = output_errors * output_network * (1.0 - output_network)     
        tmp = self.learningrate  * np.dot(tmp, output_hiddenTwo.T)
        self.wHO += tmp

        # calculate HO errors:
        HH_errors = np.dot(self.wHO.T, output_errors)
        

        # update the first hidden weights:
        tmp = HH_errors * output_hiddenTwo * (1.0 - output_hiddenTwo)
        tmp = self.learningrate  * np.dot(tmp, output_hiddenOne.T)
        self.wHH += tmp

        #calculate HH errors
        # print("wHH.T shape: ", self.wHH.T.shape)
        # print("HH_errors shape: ", HH_errors.shape)
        HI_errors = np.dot(self.wHH.T, HH_errors)
        

        #update input weights
        tmp = HI_errors * output_hiddenOne * (1.0 - output_hiddenOne)
        tmp = self.learningrate * np.dot(tmp, inputs_list.T)
        self.wIH += tmp

      
    # query the neural network
    # goes through the network
    #print("running...")
    def query(self, inputs_list):
        inputs_list = np.array(inputs_list, ndmin=2).T

        hidden_list = np.dot(self.wIH, inputs_list)
        hidden_list = activation_function(hidden_list)

        hidden2_list = np.dot(self.wHH, hidden_list)
        hidden2_list = activation_function(hidden2_list)
        
        outputs_list = np.dot(self.wHO, hidden2_list)
        outputs_list = activation_function(outputs_list)
    
        return outputs_list


    #functions
    #creating confusion matrix
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.query(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm    

    #checking the precision
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
        
    #this returns the number of correct guesses and incorrect guesses
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.query(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


#calling the function

epochs = 5
# print("rotating...")
# for i in range(len(train_imgs)):
#     train_imgs[i] = train_imgs[i].reshape((28,28))
#     train_imgs[i]= ndimage.rotate(train_imgs[i], 45, reshape=False)

# for i in range(len(test_imgs)):
#     test_imgs[i] = test_imgs[i].reshape((28,28))
#     test_imgs[i]= ndimage.rotate(test_imgs[i], 45, reshape=False)


# print("done")

NN = neuralNetwork(inputnodes  = image_pixels, 
                   outputnodes = 10, 
                   hiddennodes = 200,
                   learningrate = 0.1)



for epoch in range(epochs):  
    print("epoch: ", epoch)

    print("training...")
    for i in range(len(train_imgs)):
        NN.train(train_imgs[i], train_labels_one_hot[i])
  
    corrects, wrongs = NN.evaluate(train_imgs, train_labels)
    accTrain = corrects / ( corrects + wrongs)
    print("accuracy train: ", accTrain)
    corrects, wrongs = NN.evaluate(test_imgs, test_labels)
    accTest = corrects / ( corrects + wrongs)
    print("accuracy: test", accTest)

#-----------------------------------------------------------------------------


#-------------- making graphs
epochs = 5

number_of_nodes_per_hidden_layer = [100, 200, 300, 400, 500, 600]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6]
epochsLyst = [1, 5, 10, 15, 20, 50, 100, 200]
accuracy_list = []

#------Hidden Layer Nodes
# for i in range(len(number_of_nodes_per_hidden_layer)):
#     NN = neuralNetwork(inputnodes = image_pixels, 
#                       outputnodes = 10, 
#                       hiddennodes = number_of_nodes_per_hidden_layer[i],
#                       learningrate = 0.01)
#    print("this is for learning rate", number_of_nodes_per_hidden_layer[i])


# #------Learning Rate
# for i in range(len(learning_rate)):
#     NN = neuralNetwork(inputnodes = image_pixels, 
#                         outputnodes = 10, 
#                         hiddennodes = 200,
#                         learningrate = learning_rate[i])
#     print("this is for learning rate", learning_rate[i])

# # ------unprint for both Hidden Layer and Learning Rate
    # for epoch in range(epochs):  
    #     print("epoch: ", epoch)

    #     print("training...")
    #     for i in range(len(train_imgs)):
    #         NN.train(train_imgs[i], train_labels_one_hot[i])


#------Epochs
# NN = neuralNetwork(inputnodes  = image_pixels, 
#                    outputnodes = 10, 
#                    hiddennodes = 200,
#                    learningrate = 0.1)

# for a in range(len(epochsLyst)):
#     #print which number of epochs we are looking at
#     print("this is the total number of epochs", epochsLyst[a] )
#     for b in range(epochsLyst[a]):  
#         print("epoch: ", b)

#         print("training...")
#         for i in range(len(train_imgs)):
#             NN.train(train_imgs[i], train_labels_one_hot[i])

#     #print the training accuracy
#     corrects, wrongs = NN.evaluate(train_imgs, train_labels)
#     accuracy_train = corrects / ( corrects + wrongs)
#     print("accuracy train: ", accuracy_train)
    
#     #print the testing accuracy
#     corrects, wrongs = NN.evaluate(test_imgs, test_labels)
#     accuracy_test = corrects / ( corrects + wrongs)
#     print("accuracy: test", accuracy_test)

#     #append the testing accuracy to the new list for accuracy
#     accuracy_list.append(accuracy_test)


# #print both of the new lists
# print("----------------")
# print("learning rate: ", learning_rate)
# print("accuracy: ", accuracy_list)

# #graphing
# plt.plot(learning_rate, accuracy_list)
# plt.title("Performance vs. Learning Rate (5 epochs and 200 hidden nodes)")
# plt.ylabel("Accuracy")
# plt.xlabel("Learning Rate")
# plt.show()

#-----------------------------------------------------------------------------