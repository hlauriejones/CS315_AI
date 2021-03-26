README.txt
Laurie Jones and James Lawson
Saturday, March 27th 2021
Assignment 5: Deep Recurrent Neural Network - Model

5 Elements of Learning:

1. Target function

The target function maps input text to _______.

2. Training Data

The dataset is a csv file of companies’ responses regarding the financial services provided by the United States.

3. Hypothesis Set

In a multi-layered perception, we have non-linear functions.

4. Learning Algorithm

The learning algorithm that we are using is a recurrent neural network.

5. Final Hypothesis

The final hypothesis is a prediction of which words come before and after our target word????.


General Description:

First we had to preprocess the data. To do that, we did the following. First, we had to load in our dataset and extract the "consumer_complaint_narrative" and the "product" columns. Then we removed null or meaningless datapoints. We checked if we had any null values. Then we dropped all empty values, having to reset how everything is set, and then dropped the indexes to clean it up further. We then had to remove all null and meaningless datapoints as they are not going to contribute to our model. For example, having a blankspace should not get a token as it is not going to help predict what word comes before or after it. The same can be said about meaingless datapoints. Next, we made all the data all lower case. We had to make all of the data lower case becuase we do not a capitalized version of a word and the non capitalized version of that same word to have different tokens. Words have the same meaning as their capitalized/non capitalized versions, so they should have the same token. To accomplish this, we simpily made all text lower case. Next, we removed symbols, punctuation, brackets, parenthesis and special characters. To do this, we compiled all of the alphanumeric values we could think of and then substituted it with nothing. We had to remove it all because they do not contribute to our model and could confuse the network that it was working on. Sumbols such as these are mostly for gramatical reasons which does not hold weight for network analysis. For example, periods at the end of sentences do not help to predict the word before or after it. Next, we removed all repetitive and meaningless characters. Everytime a number in our dataset was edited out (possibly for privacy reasons), that number was replaced with "x"s. That sequence of characters does not have any relation to the words before or after it, so we removed it. However, by removing it we did not want to remove an x in "box" so we only removed it when there were two or more x's in sucession. Next, we removed stop words. Before doing this we made sure that we got rid of all of non alphaneumeric symbols in our text. We then retrived the stopwords and after going through the text and removing them all, we appended "data" with the non stopword, words. After doing this we had to join the words back together becuase we had to have the text as a list in order to easily take out the stop words. We had to remove all stop words because they are so commonly found in english that they do not contribute to our model. Additionally, they increase our training time while providing no real value. Next, we tokenized the remaining natural language. First, we created a tokenizer, that had the maximum number or words, as the maximum in our vocabulary (50000), filtered out all of the non-alphaneumeric symbols we could think of, made sure that it was lower, split the words on spaces and then made sure that not every character was treated as a token but instead every word. We basically went through and created another filter for our text to ensure that it was compatible with the tokenizer. Then we fit it to our consumer complaint narrative. We had to tokenize all of the remaining natural language because our model cannot understand all of the text at once as sentences. If it is in one long string, it will also be impossible for the network to know when a word ends or begins. Next, we transformed the language into a numerical representation. In order to do this, we indexted the tokenizer object and then changed it into a sequence representation so that we could represent full sentences and complex narratives. We had to transform the language into a numerical representation because our model cannot understand the sentiment behind individual words. Therefore, we had to transform each word into an integer value in order to proced with our model. Finally, we padded those embeddings so that all input is the same length. To pad the embeddings we looked at the sequences and then filled in the blank space of the space the size of the maximum complaint size. We filled them with 0s, because 0s is not mapped to any word embedding. We had to pad the embeddings because not all sentences are going to be the same length. For our inputs to our model, we need them all to be the same length.

After all of the preprocessing steps, we got a dummy representation of the "product" category (complaints) into their one hot encoded vector categories. We needed to do that because our model cannot understand the text data in the column, so we needed to have it one hot encoded. Next, we split our padded data and our new dummy data into training and testing data. We used a test size of 10% for our testing data. Next, we created our model. Next, we added our embedding layer. In that layer, we added the number of unique words and the embedding dimensions. Next, we added our spatial dropout with a 20% likelihood. Next, we added our LSTM layer, in our LSTM layer we had 100 nodes with a dropout of 20% and a recurrent dropout of 20%. Finally, we added our dense layer with 11 nodes and a softmax activation function.

We compiled our model with a categorical cross entropy loss function and used the Adam optimizer. We set our number of epochs to be 5. We then set the batch size to be 64. We then implemented the regularization technique of early stop training with loss to monitor when to stop training. Our patience was 3 and our minimum delta was 0.0001. We then fit our model with our training and validation data, our batch size, our epochs, and our early stop training. We then evaluvated the models accuracy and printed it. Finallly, we graphed our loss over time and our accuracy over time for the training data.


Detailed description of the machine learning concepts implemented in the project: ---------







Resources:

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
