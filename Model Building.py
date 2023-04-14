# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 01:54:35 2023

@author: Yousha
"""
import pandas as pd
import numpy as np

df = pd.read_csv('sa_dataset.csv')
df['date'] = pd.to_datetime(df['date'].apply(lambda x: x.split()[0])).dt.date
df.set_index('date',drop=True,inplace=True)
df.drop(['headline_text','compound'], axis=1, inplace=True)
df = df[['neg', 'neu', 'pos', 'open', 'low', 'close', 'adj close', 'volume', 'high']]

from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
df[df.columns] = Ms.fit_transform(df)

training_size = round(len(df) * 0.80)

train_data = df[:training_size]
test_data  = df[training_size:]

df.iloc[50,-1]

def create_sequence(dataset):
    sequences = []
    labels = []
    start_idx = 0
    
    for stop_idx in range(50, len(dataset)):
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequences), np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(9))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

model.fit(train_seq, train_label, epochs=40,validation_data=(test_seq, test_label), verbose=1)

test_predicted = model.predict(test_seq)
test_predicted[:5]

test_inverse_predicted = Ms.inverse_transform(test_predicted) # Inversing scaling on predicted data
test_inverse_predicted[:5]

train_predicted = model.predict(train_seq)
train_predicted[:5]
train_inverse_predicted = Ms.inverse_transform(train_predicted) 

prediction_df = pd.concat([df[['open','close','high']].iloc[-701:].copy(),\
                          pd.DataFrame(test_inverse_predicted[:,[3,5,-1]],columns=\
                                       ['open_predicted','close_predicted','high_predicted'],\
                                           index=df[['open','close','high']].iloc[-701:].index)], axis=1)

prediction_df[['open','close','high']] = Ms.inverse_transform(df.iloc[-701:])[:,[3,5,-1]] # Inverse scaling
    

train_prediction_df = pd.concat([df[['open','close','high']].iloc[:2953].copy(),\
                          pd.DataFrame(train_inverse_predicted[:,[3,5,-1]],columns=\
                                       ['open_predicted','close_predicted','high_predicted'],\
                                           index=df[['open','close','high']].iloc[:2953].index)], axis=1)

train_prediction_df[['open','close','high']] = Ms.inverse_transform(df.iloc[:2953])[:,[3,5,-1]] # Inverse scaling


import matplotlib.pyplot as plt

prediction_df[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


prediction_df[['close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


prediction_df[['high','high_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for highest price',size=15)
plt.show()

# High
plt.figure(figsize=(15,6))
plt.plot(train_prediction_df.index,train_prediction_df['high'], color='blue',label='Actual Price')
plt.plot(train_prediction_df.index,train_prediction_df['high_predicted'], color='red', label='Predicted Prize')
plt.plot(prediction_df.index,prediction_df['high'], color='blue')
plt.plot(prediction_df.index,prediction_df['high_predicted'], color='red')
plt.legend()
plt.xticks(rotation=45)
plt.xlabel('Date', size=12)
plt.ylabel('Stock Price', size=12)
plt.suptitle('Actual vs Predicted for highest price',size=25)
plt.savefig('High prediction vs actual.png', dpi=600)
plt.show()

from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(train_prediction_df.iloc[:,2], train_prediction_df.iloc[:,-1]))
test_rmse = np.sqrt(mean_squared_error(prediction_df.iloc[:,2], prediction_df.iloc[:,-1]))



