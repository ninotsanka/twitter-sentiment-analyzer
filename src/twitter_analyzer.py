# from textblob import TextBlob
from dotenv import load_dotenv
import tweepy
# import sys
import os

load_dotenv()

api_key = os.getenv("API_KEY")
api_sectret_key = os.getenv("API_SECRET_KEY")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(consumer_key=api_key,
                           consumer_secret=api_sectret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# tweets = api.home_timeline()
# for tweet in tweets:
#     print(tweet.text)

search_topics = 'cybersecurity'
number_of_tweets = 15

tweets = tweepy.Cursor(api.search, q=search_topics,
                       lang='en').items(number_of_tweets)
for tweet in tweets:
    print(tweet.created_at)
    print(tweet.text)
    # print(tweet.place)
    # print(tweet.source)
    # print(tweet.geo)
    print(tweet.metadata)
    # print(tweet.truncated)
    # print(dir(tweet))
    print("--------------------------")
