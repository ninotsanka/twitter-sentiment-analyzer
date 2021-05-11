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

# RUN_MODE = "Online"
RUN_MODE = "Offline"
"""str: offline or online modes
"""

OFFLINE_JSON_FILE = (
    "data/offline_tweets_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +
    ".json"
)

NUMBER_OF_TWEETS = 100
"""int: number of tweets to pull from Twitter for each topic
"""

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
"""list(str): search terms (topics)
"""

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
"""list: reference array to identify the context of the tweet
"""

POLICY_CONTEXT_KEYWORDS = [
    "policy",
    "legislature",
    "law",
    "election",
    "judiciary",
    "constitution",
    "supremecourt",
]
"""list(str): reference array to identify the context of the tweet
"""

GOVERNMENT_CONTEXT_KEYWORDS = ["government", "administration", "FBI", "CIA"]
"""list(str): reference array to identify the context of the tweet
"""


def initialize_api():
    """Initialize twitter api variables to access Twitter developer account.
    Expects: `API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET`
    environment variables to be set.

    Return: API instance

    Rtype: tweepy.API

    """
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
    """Search the Twitter datasets using search terms: ``Returns`` pandas
    Dataframe. searches only English tweets and sets result type to popular,
    iterates on the list of search topics and pulls predefined number of tweets
    `NUMBER_OF_TWEETS` for each search topic defined in `SEARCH_TOPICS` list

    Note:
        Outputs dataframe to a json file using pandas to_json method for
        offline analysis

    Returns:
        Dataframe: pandas Dataframe
    """
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
    """Reads json files from `/data` into pandas DataFrame and removes
    duplicates

    Note: If `/data` is empty program will exit with
    ``No offline data to load`` message

    Returns:
        Dataframe: pandas Dataframe
    """
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
    """Determines if a tweet contains a context.

    Return: `True` if context was found, `False` otherwise

    Rtype: bool
    """
    flag = 0
    for context in contexts:
        if tweet.find(context) != -1:
            flag = 1
    return flag


def context_filter(df):
    """Identifies the context of the tweet using identify_context(),
    adds a new column to the dataframe to indicate the context,
    and ``returns`` the updated dataframe.

    Args:
        df (Dataframe): De-duplicated dataframe of tweets

    Returns:
        Dataframe: pandas Dataframe
    """
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
    """This method essentially filters out tweets with useless words. Creates a
    new variable preprocessed_tweet which is assigned a passed in tweet.
    Replaces punctuation marks with a blank value. Removes stopwords. A stop
    word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search
    engine has been programmed to ignore, both when indexing entries for
    searching and when retrieving them as the result of a search query and
    lemmatizes the remaining words (reduces words to its base form).
    Lemmatization considers the context and converts the word to its meaningful
    base form. For example, lemmatization would correctly identify the base
    form of ‘caring’ to ‘care’

    Args:
        tweet (str): single tweet

    Returns:
        str: pre-processed tweet
    """
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
    """Loop through the dataframe and apply preproocess_tweets() method to
    clean the tweets.

    Args:
        df (DataFrame): tweets DataFrame

    Returns:
        DataFrame: pandas DataFrame
    """
    df["Processed Tweet"] = df["Tweet"].apply(lambda x: preprocess_tweets(x))
    return df


def get_polarity_subjectivity(df):
    """Calculates sentiment using TextBlob and sentiment stats using Pandas
    dataframe. Gives two values: polarity which describes how positive or
    negative the tweet is and ranges from -1 to 1, and subjectivity which
    describes how objective or subjective the tweet is and ranges from 0 to 1.
    Loops through each context dataframe and groups them by context to
    calculate, sentiment and polarity mean, max, min and median. Loops through
    the preprocessed tweets and stores values of polarity and subjectivity in
    the dataframe with new columns. Applies a filter on the dataframe to
    identify the flags, returns the columns polarity and subjectivity and
    groups by the context identified and applies aggregate function to
    calculate sentiment statistics and returns mean, max, min, and median.
    Creates new moving average dataframe to get polarity for a given window
    size in our case 10 with minimum set to 3 and return stats and sentiment
    results for each context.

    Args:
        df (DataFrame): pre-processed DataFrame of tweets

    Returns:
        [[DataFrame], [list]]: [[context sentiments],
        [context stats]] for each context
    """
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
    """Function receives context keywords as parameters. Then creates a graph
    and a table using make_subplots(). The table displays information about
    polarity and subjectivity for each context, and the graph displays
    sentiment for each context over time. go.Scatter: Creates a figure and adds
    three lines for each context with tweet’s timestamp as its x axis and
    polarity as its y axis. go.Table: creates a table and displays each
    context stats


    Args:
        technology ([[DataFrame], [list]]): Predefined context
        policy ([[DataFrame], [list]]): Predefined context
        government ([[DataFrame], [list]]): Predefined context
    """
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
