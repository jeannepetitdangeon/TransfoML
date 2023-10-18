import panda as pd
import os 

dat = read.csv

dat = pd.read_csv('C:/Users/Marion/OneDrive/Documents/cours/strasbourg/M2/Machine learning/transform\Assignment\data_tweet_sample_challenge.csv')    
var = dat.loc[:,["id","created_at", "text", "author.id", "author.username", "author.public_metrics.followers_count","public_metrics.like_count","public_metrics.retweet_count","lang", "label"]]
#check if there are na inside the dataset
var.id.isna().sum()
var.created_at.isna().sum()
var.text.isna().sum()
var.author.id.isna().sum()
var.lang.isna().sum()
var.label.isna().sum()