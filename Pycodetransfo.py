import os
import pandas as pd
import re
import nltk
import spacy
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from unidecode import unidecode
from googletrans import Translator
from tqdm import tqdm
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

###############################################################################

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
    words = [word for word in words if word not in stop_words and not 
             all(char in allowed_special_chars for char in word)]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

var['text'] = var['text'].apply(clean_text)
print(var.text)

#==============================================================================
# Extract hashtags from the tweet text
#==============================================================================

var = pd.DataFrame(var)

def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags

var['hashtags'] = var['text'].apply(extract_hashtags)

print(var)

# =============================================================================
#  Data Analysis
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

tr = Translator()

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

import pandas as pd
from transformers import pipeline
tqdm.pandas()

progress_bar = tqdm(total=len(var), desc="Classification Progress")

tweet_topics = ["AI", "Robot", "VR", "5g", "IoT"]
classifier = pipeline("zero-shot-classification")
var['classification_results'] = var['text'][:100].progress_apply(lambda text: classifier
                                                                 (text, tweet_topics)["labels"][0])

progress_bar.update(1)

progress_bar.close()




