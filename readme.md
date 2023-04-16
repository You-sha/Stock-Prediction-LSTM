# Stock Prediction: LSTM & SA
Author: Mohammad Yousha

Predicting **Kanoria Chemicals** stock price using Long Short-Term Memory and Sentiment Analysis.

**Progress:**
- [x] Study LSTM, SA and learn their application.
- [x] Prepare data.
- [x] Build model and make predictions
- [x] Documentation.

## Resources

Multi-Variate LSTM model:  https://www.kaggle.com/code/amarsharma768/stock-price-prediction-using-lstm/notebook

## Data Preparation
There were 3 million+ datapoints for the news data, and just about 3500+ for the stock data. 

Here is what I did in this step:
* Dropped news data that was from before the company's origin. 
* Removed the data that was from days when the market was closed or the stocks weren't traded. 
* There were multiple news headlines from different papers for each day, including ones useless for this purpose (entertainment, horoscopes, sports, etc.). I kept only the useful headlines and dropped the rest.
* I randomly selected one headline for each day (since there were still multiple), and finally merged the news and stock data into one dataset.

Final dataset sample:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>headline_text</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>adj close</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-01-08</td>
      <td>ULFA strikes again in Assam; kills nine people</td>
      <td>28.666666</td>
      <td>28.666666</td>
      <td>28.666666</td>
      <td>28.666666</td>
      <td>16.249998</td>
      <td>3600.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-01-09</td>
      <td>Marry-and-dump NRIs may face Indian law</td>
      <td>28.100000</td>
      <td>28.600000</td>
      <td>28.000000</td>
      <td>28.083332</td>
      <td>15.919325</td>
      <td>2490.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-01-10</td>
      <td>Kalam sets tone for engagement of global Indians</td>
      <td>27.566666</td>
      <td>29.033333</td>
      <td>27.333332</td>
      <td>27.566666</td>
      <td>15.626451</td>
      <td>32694.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-01-11</td>
      <td>Plan panel may cut SSA budget</td>
      <td>27.700001</td>
      <td>28.416666</td>
      <td>27.666666</td>
      <td>28.000000</td>
      <td>15.872088</td>
      <td>4800.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-01-12</td>
      <td>Bangladesh president resigns as chief advisor</td>
      <td>28.299999</td>
      <td>28.600000</td>
      <td>28.116667</td>
      <td>28.433332</td>
      <td>16.117727</td>
      <td>13122.0</td>
    </tr> 
  </tbody>
</table>
<p>3754 rows × 8 columns</p>
</div>

## Model Building

I have found a notebook explaining the usage of LSTM for stocks data and have modified the code in it to fit my use case. The original code can be found [here](https://www.kaggle.com/code/amarsharma768/stock-price-prediction-using-lstm/notebook).

Here are the changes I made:
- Changed the code to fit 4 features instead of two.
- Fixed the inverse transform parts.
- Reduced the number of epochs.
- Converted it into a function and made it reproducible.

### LSTM Prediction

- **Train RMSE**: 9.31
- **Test RMSE:** 4.76

![High prediction vs actual](https://user-images.githubusercontent.com/123200960/232324264-bc4331bb-8993-4a2f-a408-7a7aeeb6b03b.png)

- Before the break around 2019 is the train set, and after that is the test set.

**Next 10 days prediction:**

![Upcoming 10 days stock price](https://user-images.githubusercontent.com/123200960/232316925-36d8dc71-97b1-4ff5-a1bf-bb954391cc5b.png)

Since the data was limited to 30 March 2022 only because of the news headlines data; I had access to the actual stock price data for the days after that, and so I decided to compare my results with the actual price.

**Next 10 days - Predicted vs Actual:**

![Actual vs Predicted 10 days stock price](https://user-images.githubusercontent.com/123200960/232316741-385109c7-e9e0-4104-8de5-e6b32b88d7c4.png)

- From these results, it can be concluded that you should not use my model for actual investment.

### Random Forest Prediction

Since my model also has to use the sentiment scoring that I performed for predictions, I have also made a Random Forest Regressor model. I have evaluated it on 3 fold cv, and have tuned it using RandomizedSearch.

I have used the features `open`, `close`, `low`, `adj close`, 'volume' of the stock data, and `neg`, `neu`, `pos` from the sentiment scoring as the independent variables and `high` as the target variable.

- **RMSE**: 30.8
- **R2 Score**: 0.69

**Test set's Actual vs Predicted:**

![Actual vs Predicted stock price (RF)](https://user-images.githubusercontent.com/123200960/232323527-4b5cc08c-6dc9-430e-8d81-fffa3327173b.png)

- The model's predictions seem to flatline around March-April 2021.

The Random Forest model does not seem to perfrom as well as the LSTM model. That may make sense as I read a scientific article stating that [LSTM is currently one of the best models for stock prediction](https://www.researchgate.net/publication/306925671_Deep_learning_for_stock_prediction_using_numerical_and_textual_information).

Still, it is best to try out different methods and find the best for yourself (as long as you have the time, of course).

## Conclusion

I have made two tools to predict stock price:
- One that uses time-series data and an LSTM model. 
- Another that uses sentiment scores from news headlines combined with the stock data and a Random Forest model.
