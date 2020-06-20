# Next Word Prediction Using TensorFlow & Universal Sentence Encoder

This project is making use of a transfer learning approach, using Google's Universal Sentence Encoder to predict the next word given a string containing 5 words. 

The Universal Sentence Encoder is wrapped in a Lambda layer with Tensorflow and stacked on top of reccurent layers, followed by a softmax layer to provide a probability distribution for the next word. The model was trained on a dataset of US blogs containing millions of lines in English. From this, I am using a sliding window approach to get training samples and labels for the model. Class imbalance was faced using oversampling methods such as SMOTE.

The model is wrapped into an API using Flask and can be run on local host. 
Ex.1 | Ex. 2
- | - 
![alt](https://github.com/aswinvisva/next_word_predictor/blob/master/next_word_prediction.png) | ![alt](https://github.com/aswinvisva/next_word_predictor/blob/master/next_word_prediction2.png)

## Usage for Linux/macOS

1. Create a virtual environment 
```console
python3 -m venv env
```

2. Activate the virtual environment:
```console
source env/bin/activate
```

3. Install requirements:
```console
pip install -r requirements.txt
```

4. Serve the API locally:
```console
python3 api.py
```

5. Pass a sentence in the form
