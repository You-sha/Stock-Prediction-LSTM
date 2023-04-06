# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 04:03:36 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

news = pd.read_csv('news_headlines.csv')
stock = pd.read_csv('kanoria_stock.csv')

news.head()
stock.head()

# company's trades started from 2007-01-07, so I don't need news before that
news = news.loc[news.publish_date > 20070107]
news.head()

stock.Date #string
news.publish_date #int


# selecting only the dates where market was open and trading
news = news.drop_duplicates()
arr_news = np.array(news.publish_date)
arr_stocks = np.array(stock.Date.apply(lambda x: int(x.replace('-',''))))

news['date_in'] = news.publish_date.apply(lambda x: 1 if x not in arr_stocks else 0)
news = news.loc[news['date_in'] != 1].drop('date_in', axis=1)


# filtering categories and keeping only the useful ones
news['keep'] = news.headline_category.apply(lambda x: 1 if 'india' in x or 'business' in x\
                                            or 'world' in x or 'tech' in x else 0)
news.drop(news[news['keep'] != 1].index, inplace =True)


# selecting one news per day, as there are multiple
new_dates = pd.DataFrame(news['publish_date'].drop_duplicates())
news = news.reset_index()
new_dates = new_dates.reset_index()

new_dates.columns = ['index','date']

news_new = pd.merge(news, new_dates, how='left')
news_new = news_new.dropna()
new_cat = news_new.headline_category.value_counts() # checking the categories left


# meging stocks and new news
news_new = news_new.reset_index()
news_new.drop(['level_0','index'],axis=1,inplace=True)
news_new.date = stock.date

stock.columns = [i.lower() for i in stock.columns]

final_df = pd.merge(news_new, stock, how = 'left')
final_df = final_df.dropna()


# checking if dates match up
final_df['n_dates'] = final_df.date.apply(lambda x: int(x.replace('-','')))

(final_df.publish_date != final_df.n_dates).sum() #good

final_df.drop(['keep','publish_date','n_dates','headline_category','date'],axis=1,inplace=True)
final_df['dt'] = pd.to_datetime(final_df.date)

# output
final_df.to_csv('prepared_data.csv',index=None)












