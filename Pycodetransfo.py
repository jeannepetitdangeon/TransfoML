import os
import pandas as pd
import pandas as pd
import re
import nltk
import spacy
from google.cloud import translate_v2 as translate
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import matplotlib.pyplot as plt
import seaborn as sns

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

#check if there are na inside the dataset

# print("Number of missing values in 'id':", var['id'].isna().sum())
# print("Number of missing values in 'created_at':", var['created_at'].isna().sum())
# print("Number of missing values in 'text':", var['text'].isna().sum())
# print("Number of missing values in 'author.id':", var['author.id'].isna().sum())
# print("Number of missing values in 'author.name':", var['author.name'].isna().sum())
# print("Number of missing values in 'author.public_metrics.followers_count':", var['author.public_metrics.followers_count'].isna().sum())
# print("Number of missing values in 'public_metrics.like_count':", var['public_metrics.like_count'].isna().sum())
# print("Number of missing values in 'public_metrics.retweet_count':", var['public_metrics.retweet_count'].isna().sum())
# print("Number of missing values in 'lang':", var['lang'].isna().sum())
# print("Number of missing values in 'label':", var['label'].isna().sum())


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

df = pd.DataFrame(var)

def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return hashtags

df['hashtags'] = var['text'].apply(extract_hashtags)

print(df)


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

# Save the updated DataFrame to a new file
var.to_csv("tweets_with_entities.csv", index=False)  # Replace "tweets_with_entities.csv" with your desired output file

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

# Save the updated DataFrame to a new CSV file
var.to_csv("data_with_sentiment.csv", index=False)


###############################################################################


# Translate non-English tweets into English.


from googletrans import Translator
import time

def is_valid_language(source_lang, text):
    try:
        translator = Translator()
        translation = translator.translate(text, src=source_lang, dest='en')
        return True  # The language is valid
    except Exception:
        return False  # The language is not valid

def translate_to_english(text, source_lang):
    try:
        if source_lang and source_lang != 'en' and text:
            if is_valid_language(source_lang, text):
                translator = Translator()
                translation = translator.translate(text, src=source_lang, dest='en')
                # Pause for 1 second between translations
                time.sleep(1)
                return translation.text
            else:
                return text  # Handle the case where the source language is not recognized
        else:
            return text
    except Exception as e:
        print(f"Erreur lors de la traduction du texte : {e}")
        return text

# Assuming you have a DataFrame named 'var' with 'text' and 'lang' columns
var['text_english'] = var.apply(lambda row: translate_to_english(row['text'], row['lang']), axis=1)

# Enregistrer les données traduites dans un nouveau fichier CSV
var.to_csv("donnees_traduites.csv", index=False)


















# Utilize Zero-Shot Classification to categorize tweets into predefined or dynamically identified topics.










# # Translate non-English tweets into English.
# from googletrans import Translator

# #the function to do the task
# def translate_to_english(tweet):
#     translator = Translator(service_urls=['translate.google.com'])
#     try:
#         translation = translator.translate(tweet, dest='en').text
#     except:
#         # If there is an error during translation, return the original tweet
#         translation = tweet
#     return translation

# # Put the translation in a new column
# var['english_translation'] = var['text'].apply(translate_to_english)

# print(var['english_translation'])

# # Utilize Zero-Shot Classification to categorize tweets into predefined or dynamically identified topics.


