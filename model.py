import math

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from tensorflow.python.keras import Input
from tensorflow.python.keras.activations import swish
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Lambda, Concatenate, Dense, BatchNormalization, Dropout, Bidirectional, LSTM, \
    Reshape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, SGD

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(module_url)


def UniversalEmbedding(x):
    results = embed(tf.squeeze(tf.cast(x, tf.string)))
    return keras.backend.concatenate([results])


class PredictorModel:

    def __init__(self):
        self.model = None

    def build_model(self, embed_size=5, vocab_size=1000):

        input_text = Input(shape=(1,), dtype=tf.string)
        x = Lambda(UniversalEmbedding, output_shape=(embed_size, 1, 1))(input_text)
        x = Reshape((-1, 1))(x)

        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(256, return_sequences=False)(x)
        x = BatchNormalization()(x)

        x = Dense(512, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        output = Dense(vocab_size, activation='softmax', name='output')(x)

        self.model = Model(inputs=input_text, outputs=output)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])
        self.model.summary()

    def fit(self, X, Y):
        self.model.fit(X, Y, epochs=1000, batch_size=112)
        save_model(self.model, "model.h5")

    def load(self):
        self.model = load_model("model.h5")

    def predict(self, X):
        return self.model.predict(X)
