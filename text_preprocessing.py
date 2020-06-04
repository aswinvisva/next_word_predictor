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

    f = open(path, 'r')
    text = f.read(25000)
    f.close()

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    words = [word.lower() for word in words]
    c = Counter(words)
    dictionary = c.most_common(100)
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
            encoded_targets.append(encoded_dictionary[word])
            window_list.append(' '.join(windows[index]))

    print(words[0:20])
    print(window_list)
    print(encoded_targets[0:20])

    print("There are", len(window_list), "samples.")
    print("There are", len(encoded_targets), "samples.")

    return np.array(window_list), np.array(encoded_targets), encoded_dictionary
