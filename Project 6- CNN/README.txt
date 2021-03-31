README.md

CSCI 315 – Assignment 6 CNNs

Laurie Jones and James Lawson

April 3rd 2021 

Summary:
We completed the following 8 steps that make up the preprocessing process:

1: Removing null or meaningless datapoints.
  First we check if we have any null values. Then we drop all empty values, have to reset how everything is set, and then drop the indexes to clean it up further. 
  We had to remove all null and meaningless datapoints as they are not going to contribute to our model. For example, having a blankspace should not get a token as it is not going to help predict what word comes before or after it. The same can be said about meaingless datapoints.

2: Making all the data all lower case.
  We had to make all of the data lower case becuase we do not a capitalized version of a word and the non capitalized version of that same word to have different tokens. Words have the same meaning as their capitalized/non capitalized versions, so they should have the same token. To accomplish this, we simpily make all text lower case.

3: Removing symbols, punctuation, brackets, parenthesis and special characters.
  First we do this by compiling all of the alphanumeric values we could think of and then substituted it with nothing. 
  We had to remove it all because they do not contribute to our model and could confuse the network that it was working on. Sumbols such as these are mostly for gramatical reasons which does not hold weight for network analysis. For example, periods at the end of sentences do not help to predict the word before or after it.

4: Removing repetitive and meaningless characters.
 Everytime a number in our dataset was edited out (possibly for privacy reasons), that number was replaced with "x"s. That sequence of characters does not have any relation to the words before or after it, so we remove it. However by removing it we did not want to remove an x in "box" so we only removed it when there were two or more x's in sucession. 

5: Removing stop words.
  Before doing this we made sure that we got rid of all of non alphaneumeric symbols in our text. We then retrived the stopwords and after going through the text and removing them all, we appended "data" with the non stopword, words. After doing this we had to join the words back together becuase we had to have the text as a list in order to easily take out the stop words.  
  We had to remove all stop words because they are so commonly found in english that they do not contribute to our model. Additionally, they increase our training time while providing no real value.


6: Tokenize the remaining natural language.
  First we created a tokenizer, that had the maximum number or words, as the maximum in our vocabulary (50000), filtered out all of the non-alphaneumeric symbols we could think of, made sure that it was lower, split the words on spaces and then made sure that not every character was treated as a token but instead every word. We basically went through and created another filter for our text to ensure that it was compatible with the tokenizer. Then we fit it to our consumer complaint narrative. 
  We had to tokenize all of the remaining natural language because our model cannot understand all of the text at once as sentences. If it is in one long string, it will also be impossible for the network to know when a word ends or begins. 

7: Transform the language into a numerical representation.
  In order to do this we indexted the tokenizer object and then changed it into a sequence representation so that we could represent full sentences and complex narratives. 
  We had to transform the language into a numerical representation because our model cannot understand the sentiment behind individual words. Therefore, we have to transform each word into an integer value in order to proced with our model.

8: Pad those embeddings so that all input is the same length.
  To pad the embeddings we looked at the sequences and then filled in the blank space of the space the size of the maximum complaint size. We fill them with 0s, because 0s is not mapped to any word embedding.
  We had to pad the embeddings because not all sentences are going to be the same length. For our inputs to our model, we need them all to be the same length.

Sources:
