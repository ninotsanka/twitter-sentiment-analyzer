# Twitter Sentiment Analyzer

## What is it?

The application to explore existing twitter datasets. Detect tweets related to privacy and computer
security. Analyze tweets using sentiment analysis. Explore the relationship between the number of
retweets and the tweet context and visualize the data.

## Installation

### Clone the repository:

```
git clone https://github.com/ninotsanka/twitter-sentiment-analyzer.git
```

### Install pip dependencies

```bash
pip install -r requirements.txt
```
### Twitter API

To pull tweets from twitter, you'll need to first obtain twitter developer api tokens. Offline mode doesn't require Twitter API tokens. For online mode you can request them from here: [Twitter Developer: Use Cases, Tutorials, & Documentation](https://developer.twitter.com/en), you'll need to create a developer account.

Once you get api tokens create a file named **.env** in the root directory of the project with the following variables set:

```
API_KEY=<your_unique_key>
API_SECRET_KEY=<your_unique_key>
BEARER_TOKEN=<your_unique_key>
ACCESS_TOKEN=<your_unique_key>
ACCESS_TOKEN_SECRET=<your_unique_key>
```

## Run

To run the script, simply run the following commands in the cmd or bash

```bash
python src/twitter_analyzer.py
```

### Change configurations: run modes
Global variables can be modified to adjust src/twitter_analyzer.py run modes: Set RUN_MODE variable to either "Offline" or "Online".
