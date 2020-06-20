import math

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.python.keras.layers import Lambda, Concatenate, Dense, BatchNormalization, Dropout, Bidirectional, LSTM, \
    Reshape, Embedding
from tensorflow.python.keras.models import Model, save_model, load_model, Sequential

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embed = hub.load(module_url)

def UniversalEmbedding(x):
    import tensorflow as tf
    import tensorflow_hub as hub

    results = embed(tf.squeeze(tf.cast(x, tf.string)))
    return keras.backend.concatenate([results])


class PredictorModel:

    def __init__(self):
        self.model = None

    def build_model(self, embed_size=5, vocab_size=1000):

        model = Sequential()
        model.add(Embedding(vocab_size, 2, input_length=embed_size))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.4))
        model.add(Dense(vocab_size, activation='softmax', name='output'))

        self.model = model

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])
        self.model.summary()

    def fit(self, X, Y):
        self.model.fit(X, Y, epochs=250, batch_size=32)
        save_model(self.model, "model.h5")

    def load(self):
        self.model = load_model("model.h5")

    def predict(self, X):
        return self.model.predict(X)
