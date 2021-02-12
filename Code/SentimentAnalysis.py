_author_ = 'Akshay Kumar'
_Git_ = 'https://github.com/akshay-591'

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

        :param dataset:
        :return:
        """
        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(cleandata(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(cleandata(tokens, stop_words))

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

        :param data:
        :return:
        """
        custom_tokens = cleandata(word_tokenize(data))
        return self.classifier.classify(dict([token, True] for token in custom_tokens))


def cleandata(tweet_tokens, stop_words=()):
    """
    :param tweet_tokens:
    :param stop_words:
    :return:
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

        lemmatizer = nltk.WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Determine Word density

def get_tweets_for_model(cleaned_tokens_list):
    """

    :param cleaned_tokens_list:
    :return:
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
