from collections import Counter

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras_preprocessing.text import hashing_trick
import itertools as it
import json
from imblearn.over_sampling import SMOTE


def window(arr, window=3):
    if window > len(arr): return None

    windows = []
    labels = []

    for idx in range(len(arr) - window - 1):
        windows.append(arr[idx:idx + window])
        labels.append(arr[idx + window])

    return windows, labels


def n_gram_generate(path):
    f = open(path, 'r')
    text = f.read(1000000)
    f.close()
    sentences = sent_tokenize(text)

    threegrams = ngrams(text.split(), 2)

    count = Counter(threegrams)

    print(count)


def generate(path):
    f = open(path, 'r')
    text = f.read(35000)
    f.close()

    words = word_tokenize(text)

    words = [word.lower() for word in words if word.isalnum()]

    c = Counter(words)
    vocab_size = len(c.keys())
    dictionary = c.most_common(vocab_size)
    print(c)
    print(dictionary)

    encoded_dictionary = {}
    embed_dictionary = {}

    for idx, word in enumerate(dictionary):
        encoded_dictionary[word[0]] = idx

    for word in encoded_dictionary.keys():
        embed_dictionary[encoded_dictionary[word]] = c[word]

    windows, data = window(words, window=5)

    window_list = []
    encoded_targets = []

    print(embed_dictionary)

    for index, word in enumerate(data):
        if word in encoded_dictionary.keys():
            encoded_vec = np.zeros(vocab_size)
            encoded_vec[encoded_dictionary[word]] = 1

            encoded_targets.append(encoded_dictionary[word])
            win = [encoded_dictionary[w] for w in windows[index]]
            window_list.append(win)
    x = []
    y = []

    # Remove all words which occur less than 6 times so that SMOTE can be used
    for index, word in enumerate(encoded_targets):
        if embed_dictionary[word] > 6:
            x.append(window_list[index])
            y.append(encoded_targets[index])

    # Oversample less frequent classes using SMOTE
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(np.array(x), np.array(y))

    encoded_targets = np.array(y_res).reshape(len(y_res), 1)

    print("There are", len(X_res), "samples.")
    print("There are", len(y_res), "targets.")

    print("Shape", encoded_targets.shape)

    return np.array(X_res), y_res, encoded_dictionary
