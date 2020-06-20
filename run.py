from model import PredictorModel
import numpy as np
import text_preprocessing
import json

if __name__ == '__main__':
    w, l, dictionary = text_preprocessing.generate("en_US/en_US.blogs.txt")
    p = PredictorModel()
    p.build_model(vocab_size=len(dictionary.keys()))
    # p.fit(w, l)

    p.load()

    words = ['let', 'me', 'see', 'how', 'i']
    num = [dictionary[w] for w in words]

    prediction = p.predict([num])
    prediction = prediction[0]

    top_n = sorted(range(len(prediction)), key=lambda i: prediction[i])[-3:]
    print(top_n)

    inv_map = {v: k for k, v in dictionary.items()}

    with open('data.json', 'w') as fp:
        json.dump(inv_map, fp)

    for index in top_n:
        print("prediction: ", inv_map[index])
