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
#new import
import series

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

#we need to somehow pass in new_data["consumer_complaint_narrative"] into the function below
column = "consumer_complaint_narrative"
data = "new_data[column]"


#print("before the function: ---------- ", new_data.head())

print("hello  ")

#Create a clean text function
def clean_text(data):
    
    #Lower case the text
    new_data["consumer_complaint_narrative"] = new_data["consumer_complaint_narrative"].str.lower()
    new_data["product"] = new_data["product"].str.lower()
    
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")    
    new_data = new_data.replace("[,.\"!@#$%^&*(){}?/;`~:<>+=-]","",regex=True).astype(str)
    
    #Sub the regular expresion with a "" character.
    #I dont really know what a regular expression is or what pattern we are looking for

    #re.sub(pattern, repl, string, count=0, flags=0)


    
    #Remove x from the text characters with a "" character.
    new_data["consumer_complaint_narrative"] = new_data["consumer_complaint_narrative"].str.replace("x", "")
    new_data["product"] = new_data["product"].str.replace("x", "")


    
    #Split the text
    #how are we suspose to split the text?
    
    
    #For each word check if its a word and its an alphanumeric
    new_data["consumer_complaint_narrative"].str.isalnum()
    new_data["product"].str.isalnum()

    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))

    
    #Check if each word in the text and add the ones not in stop words
    

    
    #Join all the text by " "
    

    
    #Return the clean text
    pass



#Apply clean text to the complaints
#new_data["consumer_complaint_narrative"] = clean_text(new_data["consumer_complaint_narrative"])

#new_data["consumer_complaint_narrative"] = new_data["consumer_complaint_narrative"].str.lower()

#print("after the function: ---------- ", new_data.head())

# ----- we just need to apply this to the complaints column



#Define maximum number of words in our vocabulary to 50000
vocab_size = 50000

#Define maximum number of words within each complaint document to 250


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


