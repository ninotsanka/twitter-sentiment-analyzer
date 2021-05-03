import tweepy
import plotly.graph_objects as go
import pandas as pd
import os
import nltk
# import json
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dotenv import load_dotenv
from datetime import datetime
# import time
# import numpy as np

load_dotenv()
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

# RUN_MODE = "Offline"
RUN_MODE = "Online"

OFFLINE_JSON_FILE = "data/offline_tweets_" + datetime.now().strftime(
    "%H-%M-%S") + ".json"

# search terms (topics)
# search_topics = ['privacy', 'trust']
search_topics = [
    'privacy', 'trust', 'security', 'cybersecurity', 'technology',
    'dataprotection', 'hacking', 'infosec', 'tech', 'hacker', 'datasecurity',
    'cybercrime', 'dataprivacy', 'linux', 'data', 'gdpr',
    'informationsecurity', 'encryption', 'iot', 'cyber', 'internet',
    'business', 'cyberattack', 'windowtint', 'protection', 'programming',
    'vpn', 'hackers', 'malware'
]
number_of_tweets = 10

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


def get_tweets_online():
    # search the Twitter datasets using the search terms
    api = initialize_api()
    # create an empty list for tweets
    tweets = []
    # https://docs.tweepy.org/en/latest/api.html?highlight=lang#tweepy.API.search
    for search_topic in search_topics:
        search_key = search_topic
        # print(api.rate_limit_status())
        print(search_key)
        query = tweepy.Cursor(api.search,
                              q=search_key,
                              lang='en',
                              tweet_mode='extended',
                              result_type='recent').items(number_of_tweets)

        for tweet in query:
            # weed out retweets
            if hasattr(tweet, 'retweeted_status'):
                # print(tweet.retweeted_status.full_text)
                # "2021-05-03 02:09:24"
                # print(tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"))
                tweets.append({
                    'Tweet':
                    tweet.retweeted_status.full_text,
                    'Timestamp':
                    tweet.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # add tweets to the tweets list
                tweets.append({
                    'Tweet':
                    tweet.full_text,
                    'Timestamp':
                    tweet.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })

    # save tweets to csv file format for offline use
    # with open("data/offline_tweets.json", "w",
    #           encoding="utf-8") as output_file:
    #     json.dump(tweets, output_file, ensure_ascii=False)
    # for tweet in tweets:
    #     output_file.write(",\n")

    # create dataframe
    df = pd.DataFrame.from_dict(tweets)
    # remove duplicate tweets from the dataset
    df.drop_duplicates(subset="Tweet", keep=False, inplace=True)

    df.to_json(OFFLINE_JSON_FILE, orient="records", force_ascii=False)

    print(df.count())
    # print(df.head)
    # print(df.shape)
    return df


def get_tweets_offline():
    import glob
    list_of_files = glob.glob("data/*.json")
    # print(list_of_files)
    df = pd.DataFrame()
    for json_file in list_of_files:
        # print(json_file)
        df = df.append(pd.read_json(json_file))
        # print(df.count())

    if df.empty:
        print("No offline data to load")
        exit(1)

    df.drop_duplicates(subset="Tweet", keep=False, inplace=True)
    # print(df.head)
    print(df.count())
    # print(df.head)
    # print(df.shape)
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

    # print(df.head())

    # print(df[df['technology'] == 1][[
    #     'technology', 'polarity', 'subjectivity'
    # ]].groupby('technology').agg([np.mean, np.max, np.min, np.median]))
    # print(df[df['policy'] == 1][['policy', 'polarity',
    #                              'subjectivity']].groupby('policy').agg(
    #                                  [np.mean, np.max, np.min, np.median]))
    # print(df[df['government'] == 1][[
    #     'government', 'polarity', 'subjectivity'
    # ]].groupby('government').agg([np.mean, np.max, np.min, np.median]))
    # Wrong number of items passed 2, placement implies 1
    technology = df[df['technology'] == 1][['Timestamp', 'polarity']]
    technology = technology.sort_values(by='Timestamp', ascending=True)
    polarity = technology.rolling(5, min_periods=2).mean()
    technology['Polarity'] = polarity if len(polarity) > 0 else []

    policy = df[df['policy'] == 1][['Timestamp', 'polarity']]
    policy = policy.sort_values(by='Timestamp', ascending=True)
    polarity = policy.rolling(5, min_periods=2).mean()
    policy['Polarity'] = polarity if len(polarity) > 0 else []

    government = df[df['government'] == 1][['Timestamp', 'polarity']]
    government = government.sort_values(by='Timestamp', ascending=True)
    polarity = government.rolling(5, min_periods=2).mean()
    government['Polarity'] = polarity if len(polarity) > 0 else []

    print("technology")
    print(technology['Timestamp'])
    print(technology['Polarity'])
    print("policy")
    print(policy['Timestamp'])
    print(policy['Polarity'])
    print("government")
    print(government['Timestamp'])
    print(government['Polarity'])

    return technology, policy, government


def visualize_data(technology, policy, government):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=technology['Timestamp'],
                   y=technology['Polarity'],
                   mode='lines',
                   name='lines'))
    fig.add_trace(
        go.Scatter(x=policy['Timestamp'],
                   y=policy['Polarity'],
                   mode='lines',
                   name='lines'))
    fig.add_trace(
        go.Scatter(x=government['Timestamp'],
                   y=government['Polarity'],
                   mode='lines',
                   name='lines'))
    fig.show()


if __name__ == "__main__":
    if RUN_MODE == "Online":
        raw_tweets = get_tweets_online()
    else:
        raw_tweets = get_tweets_offline()

    filtered_tweets = context_filter(raw_tweets)
    ready_tweets = clean_tweets(filtered_tweets)
    t, p, g = get_polarity_subjectivity(ready_tweets)

    visualize_data(t, p, g)
