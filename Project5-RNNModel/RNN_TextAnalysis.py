"""
Authors: Laurie Jones and James Lawson


Links:
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

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
import re

#downloads
# nltk.download('stopwords')
# nltk.download('punkt')

#Read in csv file into pandas data frame
data = pd.read_csv("consumer_complaints.csv")

#Extract complaints and products into a new dataframe
new_data = data[["consumer_complaint_narrative","product"]]

#Check how many values are null and count them
nullVals = new_data["consumer_complaint_narrative"].isnull()
nullCounts = nullVals.sum()
print("We have", str(nullCounts), "null values.")

#Check how many product values we have
productValue = new_data["product"].count()
print("We have", productValue, "product values.")

#Drop all empty values, reset the index, and drop them
new_data = new_data.dropna()
new_data = new_data.reset_index(drop = True)

#drop the indexes 
new_data.reset_index(drop=True, inplace=True)

#Recount product values
total_product_values = len(new_data)
print("We have", total_product_values, "product values.")


#Create a clean text function
def clean_text(data):
    
    #Lower case the text
    new_data = data.lower()
    
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,'_.\"!@#$%^&*(){}?/;`~:<>+=-]")  #added an underscore and apostrophy, might want to get rid of that

    #Sub the regular expresion with a "" character.
    new_data = re.sub(pattern, "", str(new_data))

    #Remove repetative x's from the text characters with a "" character.
    new_data = re.sub(r"xx+", "", str(new_data))   #had weird space in it now
    
    #Split the text
    new_data = new_data.split()    #split and loss ALOT of numbers

    # # #For each word check if its a word and its an alphanumeric
    # print("checking for alphanumeric")
    # for i in range(len(splitData)):
    #     #print("hi")
    #     #print(splitData[i])
    #     if splitData[i].isalnum() == False:
    #         print("Problem")
    #         print(splitData[i])
    #         #can just remove them here 
    #         splitData.remove(splitData[i]) #it runs out range??

    new_data[:] = [x for x in new_data if x.isalnum()]
    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))  
    
    #Check if each word in the text and add the ones not in stop words
    data = []
    for w in new_data:  
        if w not in stop_words:  
            data.append(w)
    
    #Join all the text by " "
    seperator = " "
    new_data = seperator.join(data)
    
    #Return the clean text
    return new_data

#Apply clean text to the complaints
print("cleaning...")
new_data["consumer_complaint_narrative"] = new_data["consumer_complaint_narrative"].apply(clean_text)
print("done cleaning...")

#Define maximum number of words in our vocabulary to 50000
vocab_size = 50000

#Define maximum number of words within each complaint document to 250
max_complaint_size = 250

#Define maximum number of words within each embedding to 100
embedding_dim = 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level
tokenizer = Tokenizer(
    num_words = vocab_size, #the maximum number of words to keep, based on word frequency.
    filters ='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower = True,        # Whether to convert the texts to lowercase.
    split = " ",         # Separator for word splitting. 
    char_level = False, # every character will be treated as a token.
)

#Fit Tokenizer object on the text
tokenizer.fit_on_texts(new_data["consumer_complaint_narrative"])

#Get the word index from tokenizer object
word_index = tokenizer.word_index #tokens for complaint

#Print number of unique tokens found
print("The number of unique tokens found is: ", len(word_index))

#Get a text to sequences representation of the complaints
sequences = tokenizer.texts_to_sequences(new_data["consumer_complaint_narrative"])

#Pad the sequences with the max length
padded = pad_sequences(sequences, maxlen = max_complaint_size)

#Print the shape of the data
print("The shape of our data is: ", padded.shape)

#Print the first example of the tokenizer object to the sequences to text
print("The first example of a text to sequences representation of the complaints is: ", sequences[0])

################################################################################################

print("----------------------------")


#Get dummy representation of complaints into their one hot encoded vector categories
dummy = pd.get_dummies(new_data["product"])

#Split data into training and testing data with a test size of 10%
X_train, X_test, y_train, y_test = train_test_split(padded,
                                                    dummy,
                                                    test_size = 0.1)

#Create your RNN model
model = tf.keras.Sequential()

#Add your embedding layers  ----- these numbers are wrong
model.add(tf.keras.layers.Embedding(input_dim = vocab_size,
                                    output_dim = embedding_dim))

#Add your spatial dropout (20%)
model.add(tf.keras.layers.Dropout(0.2))

#Add your LSTM layer
model.add(tf.keras.layers.LSTM(100,
                               dropout = 0.2,
                               recurrent_dropout = 0.2))

#Add your dense layer
model.add(tf.keras.layers.Dense(11, activation = "softmax"))

model.summary()

#Compile your model
model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False),
    metrics=['accuracy'])

#Define epochs
epochs = 5

#Define batch size
batch_size = 64

#Fit the model with early stop training
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "loss",
    min_delta = 0.0001,
    patience = 3)

final_model = model.fit(
    X_train, y_train, validation_data = (X_test, y_test),
    batch_size = batch_size,
    epochs = epochs,
    callbacks = [early_stop]
)

#Evaluate the model for accuracy (index 0 is loss and index 1 is accuracy)
test_results = model.evaluate(X_train, y_train)

#Print the accuracy
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
    



