# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:56:58 2023

@author: Yousha
"""

import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',10000)

import matplotlib.pyplot as plt

df = pd.read_csv('prepared_data.csv')

df.head()

df.plot(x='date',y='close')
df.plot(x='date',y='open')
df.plot(x='date',y='volume')
plt.xticks(rotation=90)

# Sentiment Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['headline_text'] 
    date = row['date'] 
    res[date] = sia.polarity_scores(text)
    
sa = pd.DataFrame(res).T
sa = sa.reset_index()
sa = sa.rename(columns={'index':'date'})
sa_df = sa.merge(df, on='date',how='left')

sa_df.plot.hist(y='compound')
sa_df.plot.hist(y='volume')




