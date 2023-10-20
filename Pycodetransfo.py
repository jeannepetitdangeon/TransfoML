import os
import pandas as pd
import re
import nltk
import spacy
import torch
import numpy as np
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
#tr = Translator()

# Creating a progression bar
# progress_bar = tqdm(total=len(var), desc="Translation Progress")

# for i, row in var.iterrows():
#     if row["label"] != "en":
#         translated_text = None
#         retries = 3  # Number of times to retry the translation
#         while retries > 0:
#             try:
#                 translation = tr.translate(row["text"], dest='en')
#                 if translation.text is not None:
#                     translated_text = translation.text
#                     break  # Translation successful, exit the loop
#             except Exception as e:
#                 print(f"Error translating row {i}: {e}")
#                 retries -= 1
#                 time.sleep(2)  # Wait for a moment before retrying

#         if translated_text is not None:
#             var.at[i, "text"] = translated_text

#     progress_bar.update(1)

# progress_bar.close()


###############################################################################


# Utilize Zero-Shot Classification to categorize tweets into predefined or 
# dynamically identified topics.

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# List of topics or categories
topics = ["AI", "Robot", "VR", "5g", "IoT"]

# Create an empty list to store the predicted topics
predicted_topics = []

# Loop through each tweet
for text in var["text"]:
    # Build the input prompt for T5
    input_prompt = f"Categorize this tweet: {text} into topics: {', '.join(topics)}"

    # Tokenize the input prompt
    input_ids = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate predictions with T5
    with torch.no_grad():
        outputs = model.generate(input_ids["input_ids"])

    # Decode the predicted categories
    predicted_category = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Append the predicted category to the list
    predicted_topics.append(predicted_category)

# Add the predicted categories to your DataFrame
var['predicted_topic'] = predicted_topics

# Display the DataFrame with predicted topics
print(var)





