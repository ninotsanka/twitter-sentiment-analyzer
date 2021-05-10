import tweepy
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
import nltk
from textblob import TextBlob
from plotly.subplots import make_subplots
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
lemmatizer = WordNetLemmatizer()

nltk.download("stopwords")
nltk.download("wordnet")
stop_words = stopwords.words("english")

RUN_MODE = "Offline"
# RUN_MODE = "Online"

OFFLINE_JSON_FILE = (
    "data/offline_tweets_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +
    ".json"
)

NUMBER_OF_TWEETS = 100

# search terms (topics)
SEARCH_TOPICS = [
    "privacy",
    "trust",
    "security",
    "cybersecurity",
    "technology",
    "dataprotection",
    "hacking",
    "infosec",
    "tech",
    "hacker",
    "datasecurity",
    "cybercrime",
    "dataprivacy",
    "linux",
    "data",
    "gdpr",
    "informationsecurity",
    "encryption",
    "iot",
    "cyber",
    "internet",
    "cyberattack",
    "windowtint",
    "protection",
    "programming",
    "vpn",
    "hackers",
    "malware",
]

# search terms to identify the context of the tweets
TECHNOLOGY_CONTEXT_KEYWORDS = [
    "tech",
    "technology",
    "gadgets",
    "android",
    "smartphone",
    "electronics",
    "computer",
    "apple",
    "iphone",
    "google",
    "facebook",
    "windows",
]
POLICY_CONTEXT_KEYWORDS = [
    "policy",
    "legislature",
    "law",
    "election",
    "judiciary",
    "constitution",
    "supremecourt",
]
GOVERNMENT_CONTEXT_KEYWORDS = ["government", "administration", "FBI", "CIA"]


def initialize_api():
    api_key = os.getenv("API_KEY")
    api_sectret_key = os.getenv("API_SECRET_KEY")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    # connect to Twitter using keys
    auth = tweepy.OAuthHandler(
        consumer_key=api_key, consumer_secret=api_sectret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    return api


def get_tweets_online():
    # search the Twitter datasets using the search terms
    api = initialize_api()
    # create an empty list for tweets
    tweets = []
    for search_topic in SEARCH_TOPICS:
        search_key = search_topic
        print(search_key)
        query = tweepy.Cursor(
            api.search,
            q=search_key,
            lang="en",
            tweet_mode="extended",
            result_type="popular",
        ).items(NUMBER_OF_TWEETS)

        for tweet in query:
            # weed out retweets
            if hasattr(tweet, "retweeted_status"):
                tweets.append(
                    {
                        "Tweet": tweet.retweeted_status.full_text,
                        "Timestamp": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            else:
                # add tweets to the tweets list
                tweets.append(
                    {
                        "Tweet": tweet.full_text,
                        "Timestamp": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
        print("Number of tweets added:", len(tweets))

    # create dataframe
    df = pd.DataFrame.from_dict(tweets)
    # remove duplicate tweets from the dataset
    df.drop_duplicates(subset="Tweet", keep=False, inplace=True)
    # output dataframe for offline analysis
    df.to_json(OFFLINE_JSON_FILE, orient="records", force_ascii=False)

    print("Number of tweets loaded:\n", df.count())
    return df


def get_tweets_offline():
    import glob

    list_of_files = glob.glob("data/*.json")

    df = pd.DataFrame()
    for json_file in list_of_files:
        df = df.append(pd.read_json(json_file))

    if df.empty:
        print("No offline data to load")
        exit(1)

    df.drop_duplicates(subset="Tweet", keep=False, inplace=True)

    print("Number of tweets loaded:\n", df.count())
    return df


def identify_context(tweet, contexts):
    flag = 0
    for context in contexts:
        if tweet.find(context) != -1:
            flag = 1
    return flag


def context_filter(df):
    df["technology"] = df["Tweet"].apply(
        lambda x: identify_context(x, TECHNOLOGY_CONTEXT_KEYWORDS)
    )
    df["policy"] = df["Tweet"].apply(
        lambda x: identify_context(x, POLICY_CONTEXT_KEYWORDS)
    )
    df["government"] = df["Tweet"].apply(
        lambda x: identify_context(x, GOVERNMENT_CONTEXT_KEYWORDS)
    )
    return df


def preprocess_tweets(tweet):
    preprocessed_tweet = tweet
    preprocessed_tweet.replace("[^\w\s]", "")
    preprocessed_tweet = " ".join(
        word for word in preprocessed_tweet.split() if word not in stop_words
    )
    preprocessed_tweet = " ".join(
        lemmatizer.lemmatize(word) for word in preprocessed_tweet.split()
    )
    return preprocessed_tweet


def clean_tweets(df):
    df["Processed Tweet"] = df["Tweet"].apply(lambda x: preprocess_tweets(x))
    return df


def get_polarity_subjectivity(df):
    df["polarity"] = df["Processed Tweet"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df["subjectivity"] = df["Processed Tweet"].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )

    technology_info = (
        df[df["technology"] == 1][["technology", "polarity", "subjectivity"]]
        .groupby("technology")
        .agg([np.mean, np.max, np.min, np.median])
        .to_dict()
    )

    policy_info = (
        df[df["policy"] == 1][["policy", "polarity", "subjectivity"]]
        .groupby("policy")
        .agg([np.mean, np.max, np.min, np.median])
        .to_dict()
    )

    government_info = (
        df[df["government"] == 1][["government", "polarity", "subjectivity"]]
        .groupby("government")
        .agg([np.mean, np.max, np.min, np.median])
        .to_dict()
    )

    technology_stats = ["technology"]
    for value in technology_info.values():
        technology_stats.append(round(value[1], 2))

    policy_stats = ["policy"]
    for value in policy_info.values():
        policy_stats.append(round(value[1], 2))

    government_stats = ["government"]
    for value in government_info.values():
        government_stats.append(round(value[1], 2))

    technology = df[df["technology"] == 1][["Timestamp", "polarity"]]
    technology = technology.sort_values(by="Timestamp", ascending=True)
    polarity = technology.rolling(10, min_periods=3).mean()
    technology["Polarity"] = polarity if len(polarity) > 0 else []

    policy = df[df["policy"] == 1][["Timestamp", "polarity"]]
    policy = policy.sort_values(by="Timestamp", ascending=True)
    polarity = policy.rolling(10, min_periods=3).mean()
    policy["Polarity"] = polarity if len(polarity) > 0 else []

    government = df[df["government"] == 1][["Timestamp", "polarity"]]
    government = government.sort_values(by="Timestamp", ascending=True)
    polarity = government.rolling(10, min_periods=3).mean()
    government["Polarity"] = polarity if len(polarity) > 0 else []

    return (
        [technology, technology_stats],
        [policy, policy_stats],
        [government, government_stats],
    )


def visualize_data(technology, policy, government):
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(['Context Stats', 'Sentiment Trends']),
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
    )

    table_headers = [
        "Context",
        "Polarity: Mean",
        "Polarity: Max",
        "Polarity: Min",
        "Polarity: Median",
        "Subjectivity: Mean",
        "Subjectivity: Max",
        "Subjectivity: Min",
        "Subjectivity: Median",
    ]

    technology_color = 'lightsalmon'
    policy_color = 'lightgreen'
    government_color = 'skyblue'

    fig.add_trace(
        go.Table(
            header=dict(values=table_headers),
            cells=dict(values=np.rot90(
                [government[1], policy[1], technology[1]], 3),
                line_color='white',
                fill_color=[
                    [technology_color, policy_color, government_color] * 3],
                font=dict(color='white')),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=technology[0]["Timestamp"],
            y=technology[0]["Polarity"],
            mode="lines+markers",
            name="technology",
            line=dict(color=technology_color)
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=policy[0]["Timestamp"],
            y=policy[0]["Polarity"],
            mode="lines+markers",
            name="policy",
            line=dict(color=policy_color)
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=government[0]["Timestamp"],
            y=government[0]["Polarity"],
            mode="lines+markers",
            name="government",
            line=dict(color=government_color)
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Twitter Sentiment Analysis",
        title_font_color='rgb(29, 161, 242)',
        legend_title_text="Context",
        xaxis_title="Date",
        yaxis_title="Sentiment",
    )
    fig.show()


if __name__ == "__main__":
    if RUN_MODE == "Online":
        print("Online mode")
        raw_tweets = get_tweets_online()
    else:
        print("Offline mode")
        raw_tweets = get_tweets_offline()

    filtered_tweets = context_filter(raw_tweets)
    ready_tweets = clean_tweets(filtered_tweets)

    technology, policy, government = get_polarity_subjectivity(
        ready_tweets
    )

    visualize_data(technology, policy, government)
