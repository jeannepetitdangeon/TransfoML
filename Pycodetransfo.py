import os
import pandas as pd
import re
import nltk
import spacy
from googletrans import Translator
from tqdm import tqdm
import time
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode


###############################################################################

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

var = dat.loc[:,["id","created_at", "text", "author.id", "author.name", "author.public_metrics.followers_count","public_metrics.like_count","public_metrics.retweet_count","lang", "label"]]

###############################################################################

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

###############################################################################

# Visualize the data to understand the distribution of tweets over time, 
# by newspaper, and by engagement metrics (likes, retweets).


# Détecter les journaux les plus importants 


###############################################################################

# Extract hashtags from the tweet text

var = pd.DataFrame(var)

def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags

var['hashtags'] = var['text'].apply(extract_hashtags)

print(var)


# =============================================================================
# # Data Analysis
# =============================================================================

# Apply Name Entity Recognition (NER) to identify key entities in the tweets.

nlp = spacy.load("en_core_web_sm")
tweets = var['text']
tweet_entities = []
for tweet in tweets:
    doc = nlp(tweet)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    tweet_entities.append(entities)

# Add the extracted entities to your DataFrame
var['tweet_entities'] = tweet_entities
print(var.head())


###############################################################################

# Perform Sentiment Analysis to evaluate newspaper sentiment towards this technologies.

# Define a function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to the 'text' column
var['sentiment'] = var['text'].apply(analyze_sentiment)

###############################################################################

# Translate tweets

# Initialize the Translator object
tr = Translator()

# Creating a progression bar
progress_bar = tqdm(total=len(var), desc="Translation Progress")

for i, row in var.iterrows():
    if row["label"] != "en":
        translated_text = None
        retries = 3  # Number of times to retry the translation
        while retries > 0:
            try:
                translation = tr.translate(row["text"], dest='en')
                if translation.text is not None:
                    translated_text = translation.text
                    break  # Translation successful, exit the loop
            except Exception as e:
                print(f"Error translating row {i}: {e}")
                retries -= 1
                time.sleep(2)  # Wait for a moment before retrying

        if translated_text is not None:
            var.at[i, "text"] = translated_text

    progress_bar.update(1)

progress_bar.close()


###############################################################################


# Utilize Zero-Shot Classification to categorize tweets into predefined or 
# dynamically identified topics.

progress_bar = tqdm(total=len(var), desc="Topics Progress")

from transformers import pipeline

# Load a zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# List of topics or categories
topics = ["AI", "Robot", "VR", "5g", "IoT"]

# Create an empty list to store the predicted topics
predicted_topics = []

# Loop through each tweet
for text in var["text"]:
    # Perform zero-shot classification
    result = classifier(text, topics)

    # Extract the most likely label
    predicted_topic = result['labels'][0]
    
    # Append the predicted topic to the list
    predicted_topics.append(predicted_topic)

# Add the predicted topics to your DataFrame
var['predicted_topic'] = predicted_topics

progress_bar.update(1)

progress_bar.close()


