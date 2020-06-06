import json

import flask
from flask import jsonify, request, Response, render_template, redirect

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from model import PredictorModel

app = flask.Flask(__name__)
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(module_url)

def UniversalEmbedding(x):
    results = embed(tf.squeeze(tf.cast(x, tf.string)))
    return keras.backend.concatenate([results])

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/v1/predictor/get_next_word', methods=['POST'])
def get_next_word():
    # initialize the data dictionary that will be returned from the
    # view

    data = {"success": False}
    p = PredictorModel()
    p.build_model(vocab_size=75)
    p.load()

    with open('data.json', 'r') as fp:
        index_map = json.load(fp)

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        if flask.request.form['fname'] != None:
            print(flask.request.form['fname'])
            sentence_split = flask.request.form['fname'].split()
            print(sentence_split)

            prediction = p.predict([sentence_split])
            prediction = prediction[0]

            # loop over the results and add them to the list of
            # returned predictions
            top_n = sorted(range(len(prediction)), key=lambda i: prediction[i])[-3:]

            first_word = index_map[str(top_n[2])]
            second_word = index_map[str(top_n[1])]
            third_word = index_map[str(top_n[0])]

            data["first_word"] = str(first_word)
            data["second_word"] = str(second_word)
            data["third_word"] = str(third_word)

            # indicate that the request was a success
            data["success"] = True

        else:
            print("No sentence passed!")

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
#
# if __name__ == '__main__':
#     app.run()
