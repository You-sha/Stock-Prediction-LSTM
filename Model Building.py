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
df_LSTM = df[['open', 'low', 'close', 'high']]

from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
df_LSTM[df_LSTM.columns] = Ms.fit_transform(df_LSTM)

training_size = round(len(df_LSTM) * 0.80)

train_data = df_LSTM[:training_size]
test_data  = df_LSTM[training_size:]

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

model.add(Dense(4))

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

prediction_df = pd.concat([df_LSTM[['open', 'low', 'close', 'high']].iloc[-701:].copy(),\
                          pd.DataFrame(test_inverse_predicted[:,:],columns=\
                                       ['open_pred','low_pred','close_pred','high_pred'],\
                                           index=df_LSTM.iloc[-701:].index)], axis=1)

prediction_df[['open', 'low', 'close', 'high']] = Ms.inverse_transform(df_LSTM.iloc[-701:]) # Inverse scaling
    

train_prediction_df = pd.concat([df_LSTM[['open', 'low', 'close', 'high']].iloc[:2953].copy(),\
                          pd.DataFrame(train_inverse_predicted[:,:],columns=\
                                       ['open_pred','low_pred','close_pred','high_pred'],\
                                           index=df_LSTM.iloc[:2953].index)], axis=1)

train_prediction_df[['open','low','close','high']] = Ms.inverse_transform(df_LSTM.iloc[:2953]) # Inverse scaling


## Next 10 days prediction

new_df = df_LSTM.append(pd.DataFrame(
    columns=df_LSTM.columns,index=pd.date_range(start=df_LSTM.index[-1], \
                                                     periods=11, freq='D', closed='right'))
    )



upcoming_prediction = pd.DataFrame(columns=['open','low','close','high'],index=new_df.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)

curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)

upcoming_prediction[upcoming_prediction.columns] = Ms.inverse_transform(upcoming_prediction[upcoming_prediction.columns])

df_LSTM.index == ['2021-04-01']

p=0

for i in list(df_LSTM.index.astype('str')):
   p += 1
   if i == '2022-03-31':
       print(f"{p} is {i}")

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(df.iloc[3714:,4],label='Current High Price', color='blue')
plt.plot(upcoming_prediction.iloc[3714:,3],label='Upcoming High Price', color='red')
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Upcoming High price prediction',size=15)
plt.legend()
plt.savefig('Upcoming 10 days stock price.png',dpi=600,bbox_inches='tight')
plt.show()


## Actual Values

actual_upcoming = pd.read_csv('kanoria_stock_next_10.csv')
actual_upcoming['Date'] = pd.to_datetime(actual_upcoming['Date'].apply(lambda x: x.split()[0])).dt.date
actual_upcoming.set_index('Date',drop=True,inplace=True)
actual_upcoming.drop(['Adj Close','Volume'], axis=1, inplace=True)
actual_value = actual_upcoming.iloc[:7,:]

plt.figure(figsize=(15,5))
plt.plot(actual_value.iloc[:,1],label='Actual High Price', color='blue')
plt.plot(upcoming_prediction.iloc[-10:,-1],label='Predicted High Price', color='red')
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.xticks(rotation=45)
plt.title('Actual vs Predicted High price',size=15)
plt.legend()
plt.savefig('Actual vs Predicted 10 days stock price.png',dpi=600,bbox_inches='tight')
plt.show()


###

prediction_df[['open','open_pred']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


prediction_df[['close','close_pred']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


prediction_df[['high','high_pred']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for highest price',size=15)
plt.show()

# High
plt.figure(figsize=(15,5))
plt.plot(train_prediction_df.index,train_prediction_df['high'], color='blue',label='Actual Stock Price')
plt.plot(train_prediction_df.index,train_prediction_df['high_pred'], color='red', label='Predicted Stock Price')
plt.plot(prediction_df.index,prediction_df['high'], color='blue')
plt.plot(prediction_df.index,prediction_df['high_pred'], color='red')
plt.legend()
plt.xticks(rotation=45)
plt.xlabel('Date', size=14)
plt.ylabel('Stock Price', size=14)
plt.suptitle('Actual vs Predicted High',size=25)
plt.savefig('High prediction vs actual.png', dpi=600,bbox_inches='tight')
plt.show()

from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(train_prediction_df['high'], train_prediction_df['high_pred']))
test_rmse = np.sqrt(mean_squared_error(prediction_df['high'], prediction_df['high_pred']))

# Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rf = RandomForestRegressor(n_estimators = 1000, random_state=0)

X = df.drop('high',axis=1)
y = df['high'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
residuals = y_test - rf_pred

np.sqrt(mean_squared_error(y_test, rf_pred)) # much lower

plt.figure(figsize=(15,5))
plt.scatter(y_test, residuals, c=residuals, cmap='magma', edgecolors='black', linewidths=.1)
plt.colorbar(label="Error", orientation="vertical")
plt.hlines(y = 0, xmin = 15, xmax= 215, linestyle='--', colors='black')
plt.xlim(10, 220)
plt.xlabel('High'); plt.ylabel('Error')
plt.savefig('RF Residuals.png',dpi=600,bbox_inches='tight')
plt.show()


# Defining a function that takes a news headline and stock numbers as input and gives a high prediction
# for the next day

def StockModel(df_LSTM, target, lstm=False, dense_layer_lstm=0, scaling=False, columns=df_LSTM.columns):
    if lstm == True:
        from sklearn.preprocessing import MinMaxScaler
        if scaling == True:
            Ms = MinMaxScaler()
            df_LSTM[df_LSTM.columns] = Ms.fit_transform(df_LSTM)
            
            training_size = round(len(df_LSTM) * 0.80)
    
            train_data = df_LSTM[:training_size]
            test_data  = df_LSTM[training_size:]
        else:
            
            training_size = round(len(df_LSTM) * 0.80)
    
            train_data = df_LSTM[:training_size]
            test_data  = df_LSTM[training_size:]
            
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

        model.add(Dense(dense_layer_lstm))

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

        prediction_df = pd.concat([df_LSTM.iloc[-test_predicted.shape[0]:].copy(),\
                                  pd.DataFrame(test_inverse_predicted,columns=\
                                               [i + '_pred' for i in columns],\
                                                   index=df_LSTM.iloc[-test_predicted.shape[0]:].index)], axis=1)

        prediction_df.iloc[:,range(len(columns))] = Ms.inverse_transform(df_LSTM.iloc[-test_predicted.shape[0]:]) # Inverse scaling
            

        train_prediction_df = pd.concat([df_LSTM[['open', 'low', 'close', 'high']].iloc[:train_predicted.shape[0]].copy(),\
                                  pd.DataFrame(train_inverse_predicted[:,:],columns=\
                                               ['open_pred','low_pred','close_pred','high_pred'],\
                                                   index=df_LSTM.iloc[:train_predicted.shape[0]].index)], axis=1)

        train_prediction_df.iloc[:,range(len(columns))] = Ms.inverse_transform(df_LSTM.iloc[:train_predicted.shape[0]]) # Inverse scaling
        
        train_rmse = np.sqrt(mean_squared_error(train_prediction_df['high'], train_prediction_df['high_pred']))
        test_rmse = np.sqrt(mean_squared_error(prediction_df['high'], prediction_df['high_pred']))
        print(f"Train RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        rf = RandomForestRegressor(n_estimators = 1000, random_state=0)

        X = df_LSTM.drop(target,axis=1)
        y = df_LSTM[target].values

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

        rf.fit(X_train,y_train)
        rf_pred = rf.predict(X_test)

        print(f"Mean Root Squared Error: {np.sqrt(mean_squared_error(y_test, rf_pred))}")
                
    
StockModel(df_LSTM,lstm=True,scaling=True,dense_layer_lstm=4,rf_target='high')


