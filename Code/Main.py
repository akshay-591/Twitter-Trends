_author_ = 'Akshay Kumar'
_Git_ = 'https://github.com/akshay-591'

import tweepy
import pandas as pd
import sys
import SentimentAnalysis


consumer_key = 'Key'
consumer_secret = 'Key'
access_token_key = 'Key'
access_token_secret = 'Key'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)

# getting the trending #hashtags from India
# first get the location woeid (where on earth id)

woeid_ind = 23424848  # woeid (where on earth id) of India

trends = api.trends_place(id=woeid_ind)
print("Problem a). Latest Trending Topics for India (#tag and No. of tweets)", "\n")
print("======== The top trends for the India are : ===============")

val = []
for value in trends:
    print(value)
    for trends in value['trends']:
        val.append([trends['name'], trends['tweet_volume']])

tweets_df = pd.DataFrame(val,
                         columns=['Hashtag', 'No. of Tweets'])
sys.stdout = open("../Output/Problem Result.txt", "w")

print("The top trends for the India are :")
print('\n\n')
print(tweets_df)
print("=======================================================================")
print()

# get the first 100 tweets from #joibiden
tweets = tweepy.Cursor(api.search, q='#JoeBiden', tweet_mode='extended', lang='en').items(100)
list_tweets = [tweet for tweet in tweets]
i = 1
db = pd.DataFrame(columns=['username', 'location', 'text', 'hashtags'])
model = SentimentAnalysis.Sentiment().fit()

print("Problem b) and c) - Extract first 100 Tweets for #JoeBiden and Find the Sentiment of these Tweets","\n")

print(" ====================================== ==")

for tweet in list_tweets:
    username = tweet.user.screen_name
    location = tweet.user.location
    hashtags = tweet.entities['hashtags']
    try:
        text = tweet.retweeted_status.full_text
    except AttributeError:
        text = tweet.full_text
    hashtext = list()
    for j in range(0, len(hashtags)):
        hashtext.append(hashtags[j]['text'])
    ith_tweet = [username, location, text, hashtext]
    print()
    print(f"Tweet {i}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Location:{ith_tweet[1]}")
    print(f"Tweet Text:{ith_tweet[2]}")
    print(f"Hashtags Used:{ith_tweet[3]}")

    # get sentiment
    sentiment = model.get_sentiment(text)
    print('Tweet Sentiment = ', sentiment)
    print("=======================================================================")

sys.stdout.close()
