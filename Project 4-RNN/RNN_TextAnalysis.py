"""
Authors: Laurie Jones and James Lawson


Links:
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

Get function working


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
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

#we need to somehow pass in new_data["consumer_complaint_narrative"] into the function below
column = "consumer_complaint_narrative"
data = "new_data[column]"

#print("before the function: ---------- ", new_data.head())

#Create a clean text function
def clean_text(data):
    
    #Lower case the text
    new_data = data.str.lower()
    print("lower")
    #print(new_data)
    
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,_.\"!@#$%^&*(){}?/;`~:<>+=-]")  #added an underscore, might want to get rid of that

    #Sub the regular expresion with a "" character.
    new_data = re.sub(pattern, "", str(new_data))
    print("characters")
    #print(new_data)

    #Remove x from the text characters with a "" character.
    new_data = re.sub(r"x", "", str(new_data))   #had weird space in it now
    print("delete x")
    #print(len(new_data))
    #print(new_data)
    
    #Split the text
    splitData = new_data.split()    #split and loss ALOT of numbers
    #print(len(splitData))
    print("split")
    # #print(splitData)
    # # print(type(splitData))

    # #For each word check if its a word and its an alphanumeric
    print("checking for alphanumeric")
    for i in range(len(splitData)):
        #print("hi")
        #print(splitData[i])
        if splitData[i].isalnum() == False:
            print("Problem")
            print(splitData[i])
    print("done")

    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))  
    
    #Check if each word in the text and add the ones not in stop words
    new_data = []

    for w in splitData:  
        if w not in stop_words:  
            new_data.append(w)
    
    print("Stop words")
    
    #Join all the text by " "
    seperator = " "
    new_data = seperator.join(new_data)
    print("joined")
    

    
    #Return the clean text
    return new_data






#Apply clean text to the complaints
new_data["consumer_complaint_narrative"] = clean_text(new_data["consumer_complaint_narrative"])
print("after the function: ---------- ", new_data.head())


#Define maximum number of words in our vocabulary to 50000
vocab_size = 50000

#Define maximum number of words within each complaint document to 250
max_complaint_size = 250

#Define maximum number of words within each embedding to 100
embedding_dim = 100


#Implement Tokenizer object with num_words, filters, lower, split, and char_level
#token = (word_token(new_data["consumer_complaint_narrative"]))


#Fit Tokenizer object on the text


#Get the word index from tokenizer object

#Print number of unique tokens found

#Get a text to sequences representation of the complaints

#Pad the sequences with the max length

#Print the shape of the data

#Print the first example of the tokenizer object to the sequences to text


