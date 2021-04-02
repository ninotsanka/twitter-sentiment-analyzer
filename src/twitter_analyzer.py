from textblob import TextBlob
from dotenv import load_dotenv
import tweepy
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

load_dotenv()
lemmatizer = WordNetLemmatizer()

api_key = os.getenv("API_KEY")
api_sectret_key = os.getenv("API_SECRET_KEY")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(consumer_key=api_key,
                           consumer_secret=api_sectret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

search_topics = ['privacy', 'trust']
number_of_tweets = 100

query = tweepy.Cursor(api.search,
                      q=search_topics,
                      lang='en',
                      tweet_mode="extended").items(number_of_tweets)

# create an empty list for tweets
tweets = []
for tweet in query:
    # weed out retweets
    if hasattr(tweet, "retweeted_status"):
        # print(tweet.retweeted_status.full_text)
        tweets.append({
            'Tweet': tweet.retweeted_status.full_text,
            'Timestamp': tweet.created_at
        })
    else:
        tweets.append({
            'Tweet': tweet.full_text,
            'Timestamp': tweet.created_at
        })

df = pd.DataFrame.from_dict(tweets)
# print(df.head())
# print(df.shape)

technology_context_keywords = ["tech", "technology"]
policy_context_keywords = ["policy", "legislature", "law"]
government_context_keywords = ["government", "administration", "FBI"]


def identify_context(tweet: str, contexts: str) -> int:
    flag = 0
    for context in contexts:
        if tweet.find(context) != -1:
            flag = 1
    return flag


df['technology'] = df['Tweet'].apply(
    lambda x: identify_context(x, technology_context_keywords))
df['policy'] = df['Tweet'].apply(
    lambda x: identify_context(x, policy_context_keywords))
df['government'] = df['Tweet'].apply(
    lambda x: identify_context(x, government_context_keywords))

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

# print(stop_words)


def preprocess_tweets(tweet):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]', '')
    preprocessed_tweet = ' '.join(word for word in preprocessed_tweet.split()
                                  if word not in stop_words)
    preprocessed_tweet = ' '.join(
        lemmatizer.lemmatize(word) for word in preprocessed_tweet.split())
    return (preprocessed_tweet)


df['Processed Tweet'] = df["Tweet"].apply(lambda x: preprocess_tweets(x))

df['polarity'] = df['Processed Tweet'].apply(
    lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['Processed Tweet'].apply(
    lambda x: TextBlob(x).sentiment.subjectivity)

print(df[df['technology'] == 1][['technology', 'polarity',
                                 'subjectivity']].groupby('technology').agg(
                                     [np.mean, np.max, np.min, np.median]))

# print(df.head(50))
