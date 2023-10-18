import os
import pandas as pd
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import matplotlib.pyplot as plt
import seaborn as sns


#dat = pd.read_csv('C:/Users/Marion/OneDrive/Documents/cours/strasbourg/M2/Machine learning/transform\Assignment\data_tweet_sample_challenge.csv')    
dat = pd.read_csv('C:/Users/epcmic/OneDrive/Documents/GitHub/Transformer/Challenge/data_tweet_sample_challenge.csv')

# Keeping important columns, ID, time, content, author id et username, number of followers,  number of likes, number of rt, language, country code
# • id: Unique identifier for each tweet
# • created at: Time at which the tweet was posted
# • text: The actual content of the tweet
# • author.id: Unique identifier for each user
# • public metrics.like count: Number of likes
# • public metrics.retweet count: Number of retweets
# • label: Country code ISO-2

var = dat.loc[:,["id","created_at", "text", "author.id", "author.username", "author.public_metrics.followers_count","public_metrics.like_count","public_metrics.retweet_count","lang", "label"]]

#check if there are na inside the dataset

print("Number of missing values in 'id':", var['id'].isna().sum())
print("Number of missing values in 'created_at':", var['created_at'].isna().sum())
print("Number of missing values in 'text':", var['text'].isna().sum())
print("Number of missing values in 'author.id':", var['author.id'].isna().sum())
print("Number of missing values in 'author.username':", var['author.username'].isna().sum())
print("Number of missing values in 'author.public_metrics.followers_count':", var['author.public_metrics.followers_count'].isna().sum())
print("Number of missing values in 'public_metrics.like_count':", var['public_metrics.like_count'].isna().sum())
print("Number of missing values in 'public_metrics.retweet_count':", var['public_metrics.retweet_count'].isna().sum())
print("Number of missing values in 'lang':", var['lang'].isna().sum())
print("Number of missing values in 'label':", var['label'].isna().sum())


# Data Pre-processing
# Text cleaning on the tweet content.


def clean_text(text):
    text = unidecode(text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    allowed_special_chars = ['@', '#']
    words = text.split()
    words = [word for word in words if word not in stop_words and not all(char in allowed_special_chars for char in word)]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)

    return cleaned_text

var['text'] = var['text'].apply(clean_text)
print(var.text)


# Visualize the data to understand the distribution of tweets over time, 
# by newspaper, and by engagement metrics (likes, retweets).

# Convert the 'created_at' column to datetime format for time-based analysis
var['created_at'] = pd.to_datetime(var['created_at'])

# Extract year and month from the 'created_at' column
var['year'] = var['created_at'].dt.year
var['month'] = var['created_at'].dt.month

# Group the data by year and month for time-based analysis
time_grouped = var.groupby(['year', 'month']).size().reset_index(name='count')

# Group the data by newspaper for analysis by newspaper
newspaper_grouped = var['author.username'].value_counts().reset_index(name='count')

# Group the data by engagement metrics (likes and retweets)
engagement_grouped = var[['public_metrics.like_count', 'public_metrics.retweet_count']].sum().reset_index()

# Create visualizations:

# Distribution of tweets over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=time_grouped, x='year', y='count')
plt.xlabel('Year')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets Over Time')
plt.show()

# Distribution of tweets by newspaper
plt.figure(figsize=(12, 6))
sns.barplot(data=newspaper_grouped.head(10), x='author.username', y='count')
plt.xlabel('Author Username')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Tweets by Author Username (Top 10)')
plt.xticks(rotation=45)
plt.show()

# Distribution of tweets by engagement metrics
plt.figure(figsize=(10, 6))
sns.barplot(data=engagement_grouped, x='index', y='public_metrics.like_count', hue='index')
plt.xlabel('Engagement Metric')
plt.ylabel('Total Count')
plt.title('Distribution of Tweets by Engagement Metrics (Likes and Retweets)')
plt.xticks(rotation=45)
plt.legend(title='Engagement Metric')
plt.show()



# Extract hashtags from the tweet text

df = pd.DataFrame(var)

def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags

df['hashtags'] = var['text'].apply(extract_hashtags)

print(df)



# Data Analysis
# Apply Name Entity Recognition (NER) to identify key entities in the tweets.



# Perform Sentiment Analysis to evaluate newspaper sentiment towards this technologies.





# Translate non-English tweets into English.




# Utilize Zero-Shot Classification to categorize tweets into predefined or dynamically identified topics.


