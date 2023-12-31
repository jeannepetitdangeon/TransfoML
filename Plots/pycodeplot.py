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

dat = pd.read_csv('C:/Users/Marion/OneDrive/Documents/cours/strasbourg/M2/Machine learning/transform\Assignment\data_tweet_sample_challenge.csv')    

var = dat.loc[:,["id","created_at", "text", "author.id", "author.name", "author.public_metrics.followers_count","public_metrics.like_count","public_metrics.retweet_count","lang", "label"]]

#Visualize the data to understand the distribution of tweets over time, 
#by newspaper, and by engagement metrics (likes, retweets).


#Détecter les journaux les plus importants 

#=============================================================================
#tweets Newspapers by year
#=============================================================================
### group data by author.name and compute the number of id for every author
count_by_author = dat.groupby("author.name")["id"].count().reset_index()

# sort result by label and ascending
count_by_author_and_label = dat.groupby(["label", "author.name"])["id"].count().reset_index().sort_values(["label", "id"], ascending=[True, False])
top_5_by_label = count_by_author_and_label.groupby("label").head(5)
print(top_5_by_label)

#modelize with a plot
fig, ax = plt.subplots(figsize=(10,6))
for i, (label, grp) in enumerate(top_5_by_label.groupby('label')):
    ax.barh(y=grp['author.name'], width=grp['id'], color=plt.cm.tab10(i / len(top_5_by_label['label'].unique())), label=label)

plt.xlabel("Author")
plt.ylabel("Number of articles")
plt.title("Top 5 of author who tweet the most by country ")
plt.show()

#### The the distribution of tweets over time, by newspapers
data = dat[['author.name', 'created_at', 'public_metrics.retweet_count']]

# Count the number of tweets by newspaper
tweet_counts = data.groupby(['author.name']).agg({'public_metrics.retweet_count': 'sum'}).reset_index()

# Select the top 10 newspapers with the most tweets
top_newspapers = tweet_counts.nlargest(10, 'public_metrics.retweet_count')['author.name'].tolist()

# Filter the data to keep only the top 10 newspapers
data = data[data['author.name'].isin(top_newspapers)]

# Group the data by newspaper and tweet year, and aggregate the number of tweets
data_by_newspaper = data.groupby(['author.name', dat['created_at'].dt.year]).agg({'public_metrics.retweet_count': 'sum'}).reset_index()

# Pivot the data to have one column for each newspaper, and one row for each year
pivoted_data = data_by_newspaper.pivot(index='created_at', columns='author.name', values='public_metrics.retweet_count').fillna(0)

# Create a bar chart for each year
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("husl", len(top_newspapers)))
sns.set(rc={'figure.figsize':(12,8)})
pivoted_data.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of tweets')
plt.title('Number of tweets per year for the top 10 newspapers')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# =============================================================================
# like by followers
# =============================================================================
# Calcul des likes par follower
var['likes_per_follower'] = var['public_metrics.like_count'] / var['author.public_metrics.followers_count']
top_journals_by_country = var.groupby(['label', 'author.name'])['likes_per_follower'].mean().groupby('label', group_keys=False).nlargest(5)
print(top_journals_by_country)

dat['author.name'].unique()

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



####like by newspapers by country by newspaper
data = dat[['public_metrics.like_count', 'author.name', 'label', 'created_at']]

# Calcul du nombre de retweets par année, journal et pays
count_by_year_author_label = data.groupby(['label', 'author.name', dat['created_at'].dt.year])['public_metrics.like_count'].sum().reset_index()

# Sélection du journal qui tweet le plus par pays
top_author_per_label = count_by_year_author_label.loc[count_by_year_author_label.groupby('label')['public_metrics.retweet_count'].idxmax()]

# Fusion avec les données d'origine pour ne conserver que les données correspondant au journal sélectionné
merged_data = pd.merge(data, top_author_per_label[['author.name', 'label']], on=['author.name', 'label'], how='inner')

# Création du graphique en barres
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("husl", len(top_author_per_label['label'].unique())))
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=merged_data['created_at'].dt.year, y=merged_data['public_metrics.like_count'], hue=merged_data['author.name'], palette='husl', ci=None)
plt.xlabel('Année')
plt.ylabel('Nombre de retweets')
plt.title('Distribution des like par année pour chaque journal qui tweet le plus par pays')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


####Number of like by year and country
data = dat[['public_metrics.like_count', 'author.name', 'label', 'created_at']]

# Calcul du nombre de retweets par année, journal et pays
count_by_year_author_label = data.groupby(['label', 'author.name', dat['created_at'].dt.year])['public_metrics.like_count'].sum().reset_index()

# Sélection du journal qui tweet le plus par pays
top_author_per_label = count_by_year_author_label.loc[count_by_year_author_label.groupby('label')['public_metrics.like_count'].idxmax()]

# Fusion avec les données d'origine pour ne conserver que les données correspondant au journal sélectionné
merged_data = pd.merge(data, top_author_per_label[['author.name', 'label']], on=['author.name', 'label'], how='inner')

# Création du graphique en barres
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("husl", len(top_author_per_label['label'].unique())))
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=merged_data['created_at'].dt.year, y=merged_data['public_metrics.like_count'], hue=merged_data['author.name'], palette='husl', ci=None)
plt.xlabel('Année')
plt.ylabel('Nombre de retweets')
plt.title('Distribution des like par année pour chaque journal qui tweet le plus par pays')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# =============================================================================
# distribution of the 5 newspapers for each country by year
# =============================================================================
dat['created_at'] = pd.to_datetime(dat['created_at'])
dat['year'] = dat['created_at'].dt.year

# grouper les données par année, par auteur et par label (pays), compter le nombre d'id pour chaque auteur et trier les résultats par ordre décroissant
count_by_author_and_label_and_year = dat.groupby(['year', 'author.name', 'label'])['id'].count().reset_index().sort_values(['label', 'year', 'id'], ascending=[True, True, False])

# afficher les résultats pour les 5 premiers auteurs pour chaque label (pays) et chaque année
top_5_by_label_and_year = count_by_author_and_label_and_year.groupby(['label', 'year']).head(5)

# Créer un graphique en barre pour chaque label (pays)
for label, grp in top_5_by_label_and_year.groupby('label'):
    fig, ax = plt.subplots(figsize=(10,6))
    for i, (year, year_grp) in enumerate(grp.groupby('year')):
        ax.bar(year_grp['author.name'], year_grp['id'], color=plt.cm.tab10(i / len(grp['year'].unique())), label=year)
    plt.xlabel('Auteur')
    plt.ylabel("Nombre d'articles")
    plt.title(f"Top 5 of authors who tweets the most by country")
    plt.legend()
    plt.show()
  
####Number of retweets by year and country
data = dat[['created_at', 'public_metrics.retweet_count', 'label']]

# Conversion de la colonne 'created_at' en colonne de dates
data['created_at'] = pd.to_datetime(data['created_at'])

# Extrait l'année de la colonne 'created_at'
data['year'] = data['created_at'].dt.year

# Calcul du nombre de retweets par année et par pays
count_by_year_label = data.groupby(['label', 'year'])['public_metrics.retweet_count'].sum().reset_index()

# Création du graphique en barres
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x='year', y='public_metrics.retweet_count', hue='label', data=count_by_year_label)
plt.xlabel('Année')
plt.ylabel('Number of retweets')
plt.title('Number of retweets by year and by country')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


####Number of retweet by year by newspaper in each country 
data = dat[['public_metrics.retweet_count', 'author.name', 'label', 'created_at']]

# Calcul du nombre de retweets par année, journal et pays
count_by_year_author_label = data.groupby(['label', 'author.name', dat['created_at'].dt.year])['public_metrics.retweet_count'].sum().reset_index()

# Sélection du journal qui tweet le plus par pays
top_author_per_label = count_by_year_author_label.loc[count_by_year_author_label.groupby('label')['public_metrics.retweet_count'].idxmax()]

# Fusion avec les données d'origine pour ne conserver que les données correspondant au journal sélectionné
merged_data = pd.merge(data, top_author_per_label[['author.name', 'label']], on=['author.name', 'label'], how='inner')

# Création du graphique en barres
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("husl", len(top_author_per_label['label'].unique())))
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=merged_data['created_at'].dt.year, y=merged_data['public_metrics.retweet_count'], hue=merged_data['author.name'], palette='husl', ci=None)
plt.xlabel('Année')
plt.ylabel('Nombre de retweets')
plt.title('Distribution des retweets par année pour chaque journal qui tweet le plus par pays')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


##### graph with tweets by country by year 

var['year'] = pd.DatetimeIndex(var['created_at']).year

# Groupement des tweets par pays et par année
tweets_by_country_year = var.groupby(['year', 'label']).size().reset_index(name='count')

# Création du graphique en barre
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("pastel"))
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x="year", y="count", hue="label", data=tweets_by_country_year,
            palette=sns.color_palette("husl", len(tweets_by_country_year['label'].unique())))
plt.xlabel('Year')
plt.ylabel("Number of tweets")
plt.title("Number of tweets by country by year")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()