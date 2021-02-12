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
sys.stdout = open("../Output/Problem Result.txt", "w") # open the text file
print("Problem a). Latest Trending Topics for India (#tag and No. of tweets)")
print("======== The top trends for the India are : ===============")

val = []
for value in trends:
    for trends in value['trends']:
        val.append([trends['name'], trends['tweet_volume']])

tweets_df = pd.DataFrame(val,
                         columns=['Hashtag', 'No. of Tweets'])

print('\n')
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
    try:
        text = tweet.retweeted_status.full_text
    except AttributeError:
        text = tweet.full_text
    ith_tweet = [username, location, text,]
    print()
    print(f"Tweet {i}:") # Tweet number
    print(f"Username:{ith_tweet[0]}")
    print(f"Location:{ith_tweet[1]}")
    print(f"\nTweet Text:{ith_tweet[2]}")
    # get sentiment
    sentiment = model.get_sentiment(text)
    print(f'\nTweet Sentiment = ', sentiment)
    print("=======================================================================")
    i += 1

sys.stdout.close() # close the files
