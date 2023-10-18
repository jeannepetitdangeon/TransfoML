import requests
import pandas as pd
import os 

#dat = pd.read_csv('C:/Users/Marion/OneDrive/Documents/cours/strasbourg/M2/Machine learning/transform\Assignment\data_tweet_sample_challenge.csv')    
dat = pd.read_csv('C:/Users/epcmic/OneDrive/Documents/GitHub/Transformer/Challenge/data_tweet_sample_challenge.csv')

# Keeping important columns, ID, time, content, author id et username, number of followers,  number of likes, number of rt, language, country code

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