from flask import Flask,request,jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import nltk
import json
from nltk import LancasterStemmer, WordNetLemmatizer
from Translation.translator import translate
nltk.download('wordnet')
nltk.download('punkt')
import requests

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
words = []
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
labels = []
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)
answer_for_tag = {}
with open('answers_for_tag.pkl', 'rb') as f:
    answer_for_tag = pickle.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input']).to_numpy()
    results = model.predict([input_data])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((labels[r[0]], str(r[1])))

    return return_list

model = load_model("chatbot_model.hdf5")
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    sentence = request.form.get('sentence')
    result = classify_local(sentence)
    tag_predicted = result[0][0]
    answer = answer_for_tag[tag_predicted]
    translated_answer = translate(answer, 'en', 'en')
    return json.dumps(translated_answer, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
    app.run(debug=True)