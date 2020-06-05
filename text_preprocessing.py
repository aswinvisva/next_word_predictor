from collections import Counter

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from keras_preprocessing.text import hashing_trick
import itertools as it
import json

def window(arr, window=3):
    if (window > len(arr)): return None

    windows = []
    labels = []

    for idx in range(len(arr) - window - 1):
        windows.append(arr[idx:idx + window])
        labels.append(arr[idx + window])

    return windows, labels

def generate(path):

    vocab_size = 50

    f = open(path, 'r')
    text = f.read(100000)
    f.close()

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    words = [word.lower() for word in words]
    c = Counter(words)
    dictionary = c.most_common(vocab_size)
    print(c)
    print(dictionary)

    encoded_dictionary = {}

    for idx, word in enumerate(dictionary):
        encoded_dictionary[word[0]] = idx

    windows, data = window(words, window=5)

    window_list = []
    encoded_targets = []

    print(encoded_dictionary)

    for index, word in enumerate(data):
        if word in encoded_dictionary.keys():
            encoded_vec = np.zeros(vocab_size)
            encoded_vec[encoded_dictionary[word]] = 1

            encoded_targets.append(encoded_vec)
            # window_list.append(' '.join(windows[index]))
            window_list.append(windows[index])

    encoded_targets = np.array(encoded_targets).reshape(len(encoded_targets), vocab_size)

    print(words[0:20])
    print(window_list[0:20])
    print(encoded_targets[0:20])

    print("There are", len(window_list), "samples.")
    print("There are", len(encoded_targets), "samples.")

    print("Shape", encoded_targets.shape)


    return np.array(window_list), encoded_targets, encoded_dictionary
