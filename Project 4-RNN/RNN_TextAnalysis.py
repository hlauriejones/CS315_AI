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
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

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

#we need to somehow pass in new_data["consumer_complaint_narrative"] into the function below
column = "consumer_complaint_narrative"
data = "new_data[column]"

#print("before the function: ---------- ", new_data.head())

#Create a clean text function
def clean_text(data):
    
    #Lower case the text
    new_data = data.lower()
    # print("lower")
    #print(new_data)
    
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,'_.\"!@#$%^&*(){}?/;`~:<>+=-]")  #added an underscore and apostrophy, might want to get rid of that

    #Sub the regular expresion with a "" character.
    new_data = re.sub(pattern, "", str(new_data))
    # print("characters")
    #print(new_data)

    #Remove repetative x's from the text characters with a "" character.
    new_data = re.sub(r"xx+", "", str(new_data))   #had weird space in it now
    # print("delete x")
    # print(len(new_data))
    #print(new_data)
    
    #Split the text
    new_data = new_data.split()    #split and loss ALOT of numbers
    # print(len(splitData))
    # print("split")
    # print(splitData)
    # print(type(splitData))

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
    #print(new_data)
    # print("done")

    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))  
    
    #Check if each word in the text and add the ones not in stop words
    data = []
    for w in new_data:  
        if w not in stop_words:  
            data.append(w)
    
    # print(new_data)
    # print("data")
    # print(data)
    # print("Stop words")
    
    #Join all the text by " "
    seperator = " "
    new_data = seperator.join(data)
    # print(new_data)
    # print("joined")
    

    
    #Return the clean text
    return new_data



#Apply clean text to the complaints
print("cleaning...")
new_data["consumer_complaint_narrative"] = new_data["consumer_complaint_narrative"].apply(clean_text)
print("after the function: ---------- ", new_data.head())


#Define maximum number of words in our vocabulary to 50000
vocab_size = 50000

#Define maximum number of words within each complaint document to 250
max_complaint_size = 250

#Define maximum number of words within each embedding to 100
embedding_dim = 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level
tokenizer = Tokenizer(
    num_words = vocab_size, #the maximum number of words to keep, based on word frequency.
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower= True,        # Whether to convert the texts to lowercase.
    split= " ",         # Separator for word splitting. 
    char_level=False, # every character will be treated as a token.
)

#Fit Tokenizer object on the text
tokenizer.fit_on_texts(new_data)

#Get the word index from tokenizer object
word_index = tokenizer.word_index #complaint and narritive are two different tokens
#print("word index", word_index) #word index {'consumer': 1, 'complaint': 2, 'narrative': 3, 'product': 4}

#Print number of unique tokens found
#uniqueWords = set(text_to_word_sequence(str(new_data))) #this is 80 words but it seemed sketch
uniqueWords = text_to_word_sequence(str(new_data))
vocab_size = len(uniqueWords)
#print(uniqueWords)
# print(str(new_data))
print(vocab_size)

#Get a text to sequences representation of the complaints

#Pad the sequences with the max length

#Print the shape of the data

#Print the first example of the tokenizer object to the sequences to text


