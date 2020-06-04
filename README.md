# next_word_predictor

Using transfer learning with Google's Universal Sentence Encoder to perform next word prediction. The new model is trained on a US blogs dataset
containing millions of lines in English. From this, I am using a sliding window approach to get training samples and labels for the model. 

The architecture of the model is fairly simple being a fully connected network stacked on top of the Universal Sentence Encoder model - so the 
accuracy can definitely be improved in the future by using recurrent layers such as RNN and LSTM.

The model is wrapped into an API using Flask