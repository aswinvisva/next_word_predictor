# next_word_predictor

Using transfer learning with Google's Universal Sentence Encoder to perform next word prediction. The new model is trained on a US blogs dataset
containing millions of lines in English. From this, I am using a sliding window approach to get training samples and labels for the model. 

The architecture of the model consists of Recurrent layers (LSTM) stacked on top of an Embedding layer and finally a fully connected layer 
to produce a probability distribution of the next word. Class imbalance was faced using oversampling methods such as SMOTE.

The model is wrapped into an API using Flask