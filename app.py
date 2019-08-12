from flask import Flask
import pickle
from model import Model
from flask_restful import reqparse, abort, Api, Resource
import tweepy
import os
from dotenv import load_dotenv

app = Flask(__name__)
api = Api(app)
load_dotenv()

CONSUMER_KEY = os.environ.get('C_KEY')
CONSUMER_SECRET = os.environ.get('CS_KEY')
ACCESS_TOKEN = os.environ.get('A_KEY')
ACCESS_TOKEN_SECRET = os.environ.get('AS_KEY')

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
twitter_api = tweepy.API(auth)

def get_dataset(query):
    results = twitter_api.search(query, count=100, lang='en')
    texts = [tweet.text for tweet in results]
    print(len(texts))
    return texts

model = Model()

parser = reqparse.RequestParser()
parser.add_argument('query')

class Predict(Resource):
    def get(self):
        #parse arguments
        args = parser.parse_args()
        user_query = args['query']
        
        #get data from twitter
        data = get_dataset(user_query)

        #process data and predict
        sentences = model.process_text(data)
        predictions = model.predict(sentences)

        #calculate average of all predictions
        average = sum(predictions) / len(predictions)
        #generate text for prediction
        prediction_text = model.process_prediction(float(average))

        #create JSON
        output = {'prediction': prediction_text, 'sentiment': average}

        return output

api.add_resource(Predict, '/')




if __name__ == "__main__":
    app.run()