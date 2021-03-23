"""
RNN_TextAnalysis.py
Authors: Laurie Jones and James Lawson

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import nltk
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
    pattern = re.compile(r"[,'_.\"!@#$%^&*(){}?/;`~:<>+=-]") 

    #Sub the regular expresion with a "" character.
    new_data = re.sub(pattern, "", str(new_data))

    #Remove repetative x's from the text characters with a "" character.
    new_data = re.sub(r"xx+", "", str(new_data)) 
    
    #Split the text
    new_data = new_data.split()

    #For each word check if its a word and its an alphanumeric
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
print("done cleaning")

#Define maximum number of words in our vocabulary to 50000
vocab_size = 50000

#Define maximum number of words within each complaint document to 250
max_complaint_size = 250

#Define maximum number of words within each embedding to 100
embedding_dim = 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level
tokenizer = Tokenizer(
    num_words = vocab_size, 
    filters ='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower = True,        
    split = " ",         
    char_level = False, 
)

#Fit Tokenizer object on the text
tokenizer.fit_on_texts(new_data["consumer_complaint_narrative"])

#Get the word index from tokenizer object
word_index = tokenizer.word_index

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

