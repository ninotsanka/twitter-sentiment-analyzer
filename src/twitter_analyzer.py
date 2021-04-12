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

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

# search terms (topics)
search_topics = ['privacy', 'trust']
number_of_tweets = 100

# search terms to identify the context of the tweets
technology_context_keywords = ['tech', 'technology']
policy_context_keywords = ['policy', 'legislature', 'law']
government_context_keywords = ['government', 'administration', 'FBI']


def initialize_api():
    api_key = os.getenv('API_KEY')
    api_sectret_key = os.getenv('API_SECRET_KEY')
    access_token = os.getenv('ACCESS_TOKEN')
    access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

    # connect to Twitter using keys
    auth = tweepy.OAuthHandler(consumer_key=api_key,
                               consumer_secret=api_sectret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def get_tweets():
    # search the Twitter datasets using the search terms
    api = initialize_api()
    query = tweepy.Cursor(api.search,
                          q=search_topics,
                          lang='en',
                          tweet_mode='extended').items(number_of_tweets)

    # create an empty list for tweets
    tweets = []
    for tweet in query:
        # weed out retweets
        if hasattr(tweet, 'retweeted_status'):
            # print(tweet.retweeted_status.full_text)
            tweets.append({
                'Tweet': tweet.retweeted_status.full_text,
                'Timestamp': tweet.created_at
            })
        else:
            # add tweets to the tweets list
            tweets.append({
                'Tweet': tweet.full_text,
                'Timestamp': tweet.created_at
            })

    # create dataframe
    df = pd.DataFrame.from_dict(tweets)
    # remove duplicate tweets from the dataset
    df.drop_duplicates(subset="Tweet", keep=False, inplace=True)
    return df


def identify_context(tweet: str, contexts: str) -> int:
    flag = 0
    for context in contexts:
        if tweet.find(context) != -1:
            flag = 1
    return flag


def context_filter(df):
    df['technology'] = df['Tweet'].apply(
        lambda x: identify_context(x, technology_context_keywords))
    df['policy'] = df['Tweet'].apply(
        lambda x: identify_context(x, policy_context_keywords))
    df['government'] = df['Tweet'].apply(
        lambda x: identify_context(x, government_context_keywords))
    return df


def preprocess_tweets(tweet):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace('[^\w\s]', '')
    preprocessed_tweet = ' '.join(word for word in preprocessed_tweet.split()
                                  if word not in stop_words)
    preprocessed_tweet = ' '.join(
        lemmatizer.lemmatize(word) for word in preprocessed_tweet.split())
    return (preprocessed_tweet)


def clean_tweets(df):
    df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x))
    return df


def get_polarity_subjectivity(df):
    df['polarity'] = df['Processed Tweet'].apply(
        lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['Processed Tweet'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)

    print(df.head())

    print(df[df['technology'] == 1][[
        'technology', 'polarity', 'subjectivity'
    ]].groupby('technology').agg([np.mean, np.max, np.min, np.median]))
    print(df[df['policy'] == 1][['policy', 'polarity',
                                 'subjectivity']].groupby('policy').agg(
                                     [np.mean, np.max, np.min, np.median]))
    print(df[df['government'] == 1][[
        'government', 'polarity', 'subjectivity'
    ]].groupby('government').agg([np.mean, np.max, np.min, np.median]))

    technology = df[df['technology'] == 1][['Timestamp', 'polarity']]
    technology = technology.sort_values(by='Timestamp', ascending=True)
    technology['Mean Average Polarity'] = technology.rolling(
        10, min_periods=3).mean()

    policy = df[df['policy'] == 1][['Timestamp', 'polarity']]
    policy = policy.sort_values(by='Timestamp', ascending=True)
    policy['Mean Average Polarity'] = policy.rolling(10, min_periods=3).mean()

    government = df[df['government'] == 1][['Timestamp', 'polarity']]
    government = government.sort_values(by='Timestamp', ascending=True)
    government['Mean Average Polarity'] = government.rolling(
        10, min_periods=3).mean()

    print(technology)
    print(policy)
    print(government)


if __name__ == "__main__":
    raw_tweets = get_tweets()
    filtered_tweets = context_filter(raw_tweets)
    ready_tweets = clean_tweets(filtered_tweets)
    get_polarity_subjectivity(ready_tweets)
