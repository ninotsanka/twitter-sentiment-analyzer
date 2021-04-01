from textblob import TextBlob
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

search_topics = 'cybersecurity'
number_of_tweets = 100

tweets = tweepy.Cursor(api.search,
                       q=search_topics,
                       lang='en',
                       tweet_mode="extended").items(number_of_tweets)
for tweet in tweets:
    # print(tweet.source_url)
    # print(dir(tweet))
    if not hasattr(tweet, "retweeted_status"):
        # print("RETWEET")
        # print(tweet.retweeted_status.full_text)
        # else:
        print(tweet.full_text)

        analysis = TextBlob(tweet.full_text)
        print(analysis.sentiment)
        print(analysis.polarity)
        # polarity = analysis.polarity
        print("-------------------")

    # if polarity > 0
    # clean_tweet = tweet.text.replace('RT', '')
    # if clean_tweet.startswith(' @'):
    #     position = clean_tweet.index(': ')
    #     clean_tweet = clean_tweet[position + 2:]
    # if clean_tweet.startswith('@'):
    #     position = clean_tweet.index(' ')
    #     clean_tweet = clean_tweet[position + 2:]
    # if clean_tweet.startswith('#'):
    #     position = clean_tweet.index(' ')
    #     clean_tweet = clean_tweet[position + 2:]
    # print(clean_tweet)
