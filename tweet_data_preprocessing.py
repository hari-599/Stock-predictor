# -*- coding: utf-8 -*-
"""tweet_data_Preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Mq0DA76n2bTrJUTHM8OzjDEaScorGjll
"""

import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

"""Importing tweet data from drive"""

data=pd.read_csv('/content/drive/MyDrive/Tweet.csv')
company=pd.read_csv('/content/drive/MyDrive/Company.csv')
com_tweet=pd.read_csv('/content/drive/MyDrive/Company_Tweet.csv')

"""Create a score column by adding number of comments, number of retweeets, number of likes"""

data['score']=data['comment_num']+data['retweet_num']+data['like_num']
data['post_date'] = pd.to_datetime(data['post_date'], unit='s')

data_n=pd.merge(data,com_tweet,on='tweet_id',how='right')
data_n['post_date']=pd.to_datetime(data_n['post_date'])
data_n['post_date']=data_n['post_date'].dt.date

"""Remove unneccessary columns"""

data_n=data_n.drop(['tweet_id','writer'],axis=1)

#Remove data with score less than 10
data_n=data_n[data_n['score']>=10]
data_n

data_n=data_n.rename(columns={'post_date':'Date','ticker_symbol':'Ticker'})
data_n

grouped_df = data_n.groupby(['Date', 'Ticker']).agg({
    'body': lambda x: ' '.join(x),  # Concatenate the text from the 'body' column
}).reset_index()
data_n=grouped_df.set_index(['Date','Ticker'])

"""Text cleaning and preprocessing"""

pattern = r'https?://\S+'
data_n['body'] = data_n['body'].str.replace(pattern, ' ')
pd.options.display.max_colwidth = 1000

warnings.simplefilter('ignore')
data_n['body']=data_n['body'].str.replace('(\@\w+.*?)',"")
data_n['body']=data_n['body'].str.replace('(\#w+.*?)',"")

data_n.head()

# Downloading historical stock prices with yfinance
# Configuring tickers and period
ticker_symbols = ["AAPL", "GOOG", "GOOGL", "AMZN", "TSLA", "MSFT"]
start_date = "2015-01-01"
end_date = "2019-12-31"
interval = "1d"

# Adjusting all prices to stock splits and dividend payments
auto_adjust = True

# Using yfinance package to get data from Yahoo Finance for each ticker
tickers = yf.Tickers(ticker_symbols)
tickers_df = tickers.history(start=start_date, end=end_date, interval=interval, auto_adjust=auto_adjust)

# Investigating data
print(tickers_df.shape)
tickers_df.head(20)

transformed_df = tickers_df.stack(level=1).rename_axis(['Date', 'Ticker'])
transformed_df.head(10)

from pandas.core.frame import Level
# Choosing columns to keep
cols = ['Close', 'Open', 'Volume']

# Creating a new dataFrame with selected columns
stock_df = transformed_df[cols].copy()


def calculate_growth(x):
    result = (x-x.shift(1))/x
    return result
def normalized_growth(x):
    scaler=MinMaxScaler()
    growth_series=x.dropna()
    growth_df=growth_series.to_frame()
    result= scaler.fit_transform(growth_df)
    return result


# Creating function for defining the Up (2), Stable (1), and Down (0) classes
def create_multiclass(x):
    result = 2 if x >= 0.50 else (0 if x <= 0.40 else 1)
    return result

stock_df['growth'] = stock_df.groupby(level='Ticker')['Close'].apply(calculate_growth)
no_growth_rows = stock_df[stock_df['growth'].isna()]
if len(no_growth_rows) > 0:
    stock_df = stock_df.drop(no_growth_rows.index)
stock_df['normalized_growth'] = normalized_growth(stock_df['growth'])
# stock_df['Normalized_growth']=normalized_growth(stock_df['growth'])
# Creating the multiclass target variable
# Creating function for defining the Up (2), Stable (1), and Down (0) classes
stock_df['target'] = stock_df['normalized_growth'].apply(create_multiclass)

stock_df.head(20)

stock_df['target'].value_counts()

# Choosing columns to keep
cols = ['Close', 'Open', 'Volume']

# Creating a new dataFrame with selected columns
stock_df = transformed_df[cols].copy()


def calculate_log_change(x):
    result = np.log(x) - np.log(x.shift(1))
    return result

def create_binary_variable(x):
    result = np.where(x >= 0, 1, 0)
    return result

# Creating function for defining the Up (2), Stable (1), and Down (0) classes
def create_multiclass(x):
    result = 2 if x >= 0.005 else (0 if x <= -0.005 else 1)
    return result

Creating columns for log returns and log volume change and ensuring that its calculated on individual ticker level
stock_df['log_ret'] = stock_df.groupby(level='Ticker')['Close'].apply(calculate_log_change)
stock_df['log_volume_change'] = stock_df.groupby(level='Ticker')['Volume'].apply(calculate_log_change)

# Creating columns for binary variables
# Value of 1 if equal or above 0, 0 if below
stock_df['log_ret_binary'] = stock_df['log_ret'].apply(create_binary_variable)
stock_df['log_volume_change_binary'] = stock_df['log_volume_change'].apply(create_binary_variable)

# Creating the multiclass target variable
# Creating function for defining the Up (2), Stable (1), and Down (0) classes
stock_df['target'] = stock_df['log_ret'].apply(create_multiclass)

stock_df.head(20)

stock_df['target'].value_counts()

# subset_values = price_data.iloc[1:-1, 1:].values.ravel()
# sns.displot(subset_values, kind='kde')
# plt.xlabel('Scaled Growth')
# plt.ylabel('PDF')
# plt.title('Distribution of Scaled Growth')
# lower_bound = 0.40
# upper_bound = 0.50
# # Plot vertical lines to indicate the classification boundaries
# plt.axvline(x=lower_bound, color='r', linestyle='--', label='Lower Bound')
# plt.axvline(x=upper_bound, color='g', linestyle='--', label='Upper Bound')
# plt.legend()

# plt.show()

stock_df.reset_index(inplace=True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df.set_index(['Date', 'Ticker'], inplace=True)
stock_df

date_2018 = pd.to_datetime("2018-12-31")
date_2019 = pd.to_datetime("2019-06-30")

train = data_n.loc[:date_2018]
val = data_n.loc[date_2018:date_2019]
test = data_n.loc[date_2019:]
train_data = stock_df.loc[:date_2018]
val_data = stock_df.loc[date_2018:date_2019]
test_data = stock_df.loc[date_2019:]

train_data=train.join(train_data,how='inner')
val_data=val.join(val_data,how='inner')
test_data=test.join(test_data,how='inner')

print(train_data['target'].value_counts())
print(val_data['target'].value_counts())
print(test_data['target'].value_counts())

train_data=train_data.drop(['Close','Open','Volume','log_volume_change','log_ret_binary','log_volume_change_binary','log_ret'],axis=1)
val_data=val_data.drop(['Close','Open','Volume','log_volume_change','log_ret_binary','log_volume_change_binary','log_ret'],axis=1)
test_data=test_data.drop(['Close','Open','Volume','log_volume_change','log_ret_binary','log_volume_change_binary','log_ret'],axis=1)
y_test=test_data['target']

train_data