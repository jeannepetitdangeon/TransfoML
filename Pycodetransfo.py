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

###############################################################################

dat = pd.read_csv('C:/Users/Marion/OneDrive/Documents/cours/strasbourg/M2/Machine learning/transform\Assignment\data_tweet_sample_challenge.csv')    
#dat = pd.read_csv('C:/Users/epcmic/OneDrive/Documents/GitHub/Transformer/Challenge/data_tweet_sample_challenge.csv')

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

print("Number of missing values in 'id':", var['id'].isna().sum())
print("Number of missing values in 'created_at':", var['created_at'].isna().sum())
print("Number of missing values in 'text':", var['text'].isna().sum())
print("Number of missing values in 'author.id':", var['author.id'].isna().sum())
print("Number of missing values in 'author.name':", var['author.name'].isna().sum())
print("Number of missing values in 'author.public_metrics.followers_count':", var['author.public_metrics.followers_count'].isna().sum())
print("Number of missing values in 'public_metrics.like_count':", var['public_metrics.like_count'].isna().sum())
print("Number of missing values in 'public_metrics.retweet_count':", var['public_metrics.retweet_count'].isna().sum())
print("Number of missing values in 'lang':", var['lang'].isna().sum())
print("Number of missing values in 'label':", var['label'].isna().sum())


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

# Calcul des likes par follower
var['likes_per_follower'] = var['public_metrics.like_count'] / var['author.public_metrics.followers_count']
top_journals_by_country = var.groupby(['label', 'author.name'])['likes_per_follower'].mean().groupby('label', group_keys=False).nlargest(5)
print(top_journals_by_country)



# Convertissez la colonne 'created_at' en datetime dans le DataFrame var
var['created_at'] = pd.to_datetime(var['created_at'])

# Extrait l'année et le mois de la colonne 'created_at' dans le DataFrame var
var['year'] = var['created_at'].dt.year
var['month'] = var['created_at'].dt.month

# Fusionnez les DataFrames var et top_journals_by_country sur les colonnes pertinentes
merged_data = var.merge(top_journals_by_country, on=['author.name', 'label'])

# Créez une visualisation de la distribution des likes par année et mois
plt.figure(figsize=(12, 6))
sns.barplot(x='year', y='public_metrics.like_count', hue='month', data=merged_data, ci=None, palette='Set3')
plt.xlabel('Année')
plt.ylabel('Nombre de Likes')
plt.title('Distribution des Likes par Année et Mois pour les Journaux les plus Importants')
plt.xticks(rotation=45)
plt.show()
























# Extract hashtags from the tweet text

# df = pd.DataFrame(var)

# def extract_hashtags(text):
#     hashtags = re.findall(r'#\w+', text)
#     return hashtags

# df['hashtags'] = var['text'].apply(extract_hashtags)

# print(df)




# =============================================================================
# # Data Analysis
# =============================================================================

# Apply Name Entity Recognition (NER) to identify key entities in the tweets.
import spacy
from spacy import displacy
spacy.cli.download('en_core_web_sm')

nlp = spacy.load('en_core_web_sm')

# Define a function to perform NER on each tweet
def perform_ner(tweet):
    doc = nlp(tweet)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

var['entities'] = var['text'].apply(perform_ner)

# Define a function to visualize NER in each tweet
def visualize_ner(tweet):
    doc = nlp(tweet)
    displacy.render(doc, style='ent', jupyter=True)

# Apply the visualization function to any tweet of your choice
visualize_ner('I just had lunch at The Ivy in London!')

# Define a function to generate n-grams from a text
from typing import List, Tuple

def generate_ngrams(text: str, n: int) -> List[Tuple[str]]:
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Sample text for n-grams generation
sample_text = "Natural language processing is a subfield of artificial intelligence."

# Generate unigrams, bigrams, and trigrams
unigrams = generate_ngrams(sample_text, 1)
bigrams = generate_ngrams(sample_text, 2)
trigrams = generate_ngrams(sample_text, 3)
print(unigrams, bigrams, trigrams)

# Perform Sentiment Analysis to evaluate newspaper sentiment towards this technologies.
from textblob import TextBlob
from typing import Tuple

def analyze_sentiment(text: str) -> Tuple[float, float]:
    """
    Conduct sentiment analysis on the given text using TextBlob.
    
    Parameters:
    - text (str): The input text for sentiment analysis.
    
    Returns:
    - Tuple[float, float]: A tuple containing polarity and subjectivity scores.
    """
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Fetch the sentiment attributes (polarity and subjectivity)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    return polarity, subjectivity

# Analyze the sentiment of the sample texts
sample_texts = [
    "I love programming. It's amazing!",
    "I hate bugs. They are annoying.",
    "The code is okay, could be better."
]

sentiment_results = [analyze_sentiment(text) for text in sample_texts]

print(sentiment_results)

# Translate non-English tweets into English.
from googletrans import Translator

#the function to do the task
def translate_to_english(tweet):
    translator = Translator(service_urls=['translate.google.com'])
    try:
        translation = translator.translate(tweet, dest='en').text
    except:
        # If there is an error during translation, return the original tweet
        translation = tweet
    return translation

# Put the translation in a new column
var['english_translation'] = var['text'].apply(translate_to_english)

print(var['english_translation'])


# Utilize Zero-Shot Classification to categorize tweets into predefined or dynamically identified topics.


