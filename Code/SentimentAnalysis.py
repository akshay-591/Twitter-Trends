_author_ = 'Akshay Kumar'
_Git_ = 'https://github.com/akshay-591'

"""
This File Contains Code for Sentiment Analysis for Twitter Tweets.
"""
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk import word_tokenize

import string
import random

stop_words = stopwords.words('english')


class Sentiment(object):
    def __init__(self):
        self.classifier = None

    def fit(self, dataset=None):
        """
        This Method will initialize the Training for model using data sample present in nltk libraries
        :param dataset:
        :return:model object
        """
        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(cleanData(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(cleanData(tokens, stop_words))

        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)
        self.classifier = nltk.NaiveBayesClassifier.train(dataset)
        return self

    def get_sentiment(self, data):
        """
        This Method will Predict sentiment on external tweet data using trained model
        :param data: external tweet
        :return: sentiment
        """
        custom_tokens = cleanData(word_tokenize(data)) # Tokenize the tweet
        new_tokens = get_tweets_for_model(custom_tokens)
        return self.classifier.classify(new_tokens)


def cleanData(tweet_tokens, stop_words=()):
    """
    This method will perform cleaning on the on the tweets like stemming and removing unneccesory things
    like hyperlink and symbols and performing cleaned tokenizing the tweets
    :param tweet_tokens:
    :param stop_words:
    :return: cleaned tokens
    """
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = nltk.re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                            token)

        token = nltk.re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        stemming = nltk.WordNetLemmatizer()
        token = stemming.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Determine Word density

def get_tweets_for_model(cleaned_tokens_list):
    """
    as the Naive Bayes classifier need the model requires,Python dictionary with words as keys and True as values.
    This Method generator function to change the format of the cleaned data and yield the data.

    :param cleaned_tokens_list:
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
