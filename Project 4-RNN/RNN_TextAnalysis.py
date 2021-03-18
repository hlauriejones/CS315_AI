"""
Authors: Laurie Jones and James Lawson


Links:
https://www.kite.com/python/answers/how-to-copy-columns-to-a-new-pandas-dataframe-in-python
https://datatofish.com/count-nan-pandas-dataframe/
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html
https://www.kite.com/python/answers/how-to-drop-empty-rows-from-a-pandas-dataframe-in-python
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
https://pypi.org/project/series/
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
#imported series seperatly
import series
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
#idk how to do this atm ---------------------------------------

#Recount product values
total_product_values = len(new_data)
print("We have", total_product_values, "product values.")


#currently working on converting the text to lower case


new_data = new_data.str.lower()


print(new_data.head)

#Create a clean text function
def clean_text(new_data):
    
    #Lower case the text
    
    
    
    #Compile pattern to remove all other characters
    
    #Sub the regular expresion with a "" character.
    
    #Remove x from the text characters with a "" character.
    
    #Split the text
    
    #For each word check if its a word and its an alphanumeric
    
    #Remove all english stop words
    
    #Check if each word in the text and add the ones not in stop words
    
    #Join all the text by " "
    
    #Return the clean text
    pass

#Apply clean text to the complaints


#Define maximum number of words in our vocabulary to 50000

#Define maximum number of words within each complaint document to 250

#Define maximum number of words within each embedding to 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level

#Fit Tokenizer object on the text

#Get the word index from tokenizer object

#Print number of unique tokens found

#Get a text to sequences representation of the complaints

#Pad the sequences with the max length

#Print the shape of the data

#Print the first example of the tokenizer object to the sequences to text

