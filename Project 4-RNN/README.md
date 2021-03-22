CSCI 315 – Assignment 4

Laurie Jones and James Lawson

Monday, March 22nd 2021

Deep Recurrent Neural Network

---------------------------------------------------------------------------

We completed the following 8 steps that make up the preprocessing process:

1: Removing null or meaningless datapoints.

We had to remove all null and meaningless datapoints as they are not going to contribute to our model. For example, having a blankspace should not get a token as it is not going to help predict what word comes before or after it. The same can be said about meaingless datapoints.

2: Making all the data all lower case.

We had to make all of the data lower case becuase we do not a capitalized version of a word and the non capitalized version of that same word to have different tokens. Words have the same meaning as their capitalized/non capitalized versions, so they should have the same token. To accomplish this, we simpily make all text lower case.

3: Removing symbols, punctuation, brackets, parenthesis and special characters.

We had to remove all of the symbols, punctuation, brackets, parenthesis and special characters as these are not going to contribute to our model. For example, periods at the end of sentences do not help to predict the word before or after it.

4: Removing repetitive and meaningless characters.

We had to remove all repetitive and meaningless characters. For example, in our dataset we had some characters that were repetitive. Everytime a number in our dataset was edited out (possibly for privacy reasons), that number was replaced with "x"s. That sequence of characters does not have any relation to the words before or after it, so we remove it.

5: Removing stop words.

We had to remove all stop words because they are so commonly found in english that they do not contribute to our model. Additionally, they increase our training time while providing no real value.

6: Tokenize the remaining natural language.

We had to tokenize all of the remaining natural language because our model cannot understand all of the text at once. If the model can only see the entire dataframe as one unbroken string, then it will not be able to see all of the individual words.

7: Transform the language into a numerical representation.

We had to transform the language into a numerical representation because our model cannot understand text. It can only understand integer values. Therefore, we have to transform each word into an integer value in order to proced with our model.

8: Pad those embeddings so that all input is the same length.

We had to pad the embeddings because not all sentences are going to be the same length. For our inputs, we need them all to be the same length. The process of padding is how we get all of the sentences to be the same length with a pre-set max length.



