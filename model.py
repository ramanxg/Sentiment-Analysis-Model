import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle, re, nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing import sequence
import tweepy
import numpy as np

class Model:

    def __init__(self):
        self.model = load_model('sentimentmodel4.h5')
        self.model._make_predict_function()
        with open('tokenizer1.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_sentence_length = self.model.layers[0].input_shape[1]
        self.lemmatizer = WordNetLemmatizer()

    def process_text(self, texts):
        sentences = []
        for s in texts:
            replaced = re.sub(r'[^a-zA-z0-9\s]','', s.lower())

            #tokenize words
            words = word_tokenize(replaced)
            
            #lemmatize words
            #lexicon = filtered_sentence
            lexicon = [self.lemmatizer.lemmatize(i) for i in words]
            sentences.append(lexicon)
            
        t = self.tokenizer.texts_to_sequences(sentences)
        p =  sequence.pad_sequences(t, maxlen=self.max_sentence_length)
        return p

    def process_prediction(self, prediction):
        if prediction <= 1:
            return "Very Negative"
        elif prediction <= 1.9:
            return "Somewhat Negative"
        elif 1.9 <= prediction <= 2.1:
            return "Neutral"
        elif prediction <= 3:
            return "Somewhat Positive"
        else:
            return "Very Positive"

    def predict(self, inputs):
        return self.model.predict_classes(inputs)

