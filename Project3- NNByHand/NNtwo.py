#NNtwo.py
'''     takes the input matrix (flattened) and makes them a numpy arrawy with 2D
        he has one-hot encoding outside of train
         b/w input and hidden
            need to transpose the input, if not already doe
             teh activation function which returns a 200x1 vector then hidden layer goes through activation function and then it goes through other hidden layer
---- first pass
         calculate error, the one-hot encoded 
         then you need to take the error and the dot produce with that and the other layers to find the errors
----- figure otu how to adjust the weights b/w layers
         follow eq. on 
         slide 23 equation - wjk = alpha * (Ek * sigmoid(Ok) * (1 - sigmoid(Ok))) * Otj
         ^^^ this needs to be applied 3 times between the hideen layers
         transpose the inputs and the targers
---- caluclate errors between all 3 layers
         iterate through the number of epocs
         normalize the data between 0.01 and 0.99
         create a target vector
         query the function to get the outputs
         take the number of correct examples vs number of incorrect examples (we did this)
----- gimp images: adjust to 28, and normalize
         save photos of the graph
            take paramerters from the y axis (accuracy) and then x axis
             for each epoch
             plot that accuracy on teh graph
             then youre done for theat hyper parameters
                can also do SSE
                cna use other libs fro image roatation - PIL from python

        can use other libs after the netwrok is created%
'''


# number_of_nodes_per_hidden_layer = [100, 200, 300, 400, 500, 600]
# accuracy_list = [0.9619, 0.9689, 0.9694, 0.9697, 0.9693, 0.9701]

# plt.plot(number_of_nodes_per_hidden_layer, accuracy_list)
# plt.title("Performance vs. Nodes per Hidden Layer")
# plt.ylabel("Accuracy")
# plt.xlabel("Number of Nodes per Hidden Layer")
# plt.show()


# 1 = 

# 5 =  0.9336666666666666

# 10 = 0.9193333333333333
# 15 = 0.8838833333333334
# 20 =  0.90855

# 50 =  0.9229166666666667

# 100: 0.9194166666666667
import matplotlib.pyplot as plt

number_of_nodes_per_hidden_layer = [1, 5, 10, 15, 20, 50, 100]
accuracy_list = [0.9179, 0.9336666666666666, 0.9193333333333333, 0.8838833333333334, 0.90855, 0.9229166666666667, 0.9194166666666667]

plt.plot(number_of_nodes_per_hidden_layer, accuracy_list)
plt.title("Performance vs. Epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()




