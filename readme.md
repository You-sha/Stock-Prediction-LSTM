# Stock Prediction: LSTM & SA
Predicting **Kanoria Chemicals** stock price using Long Short-Term Memory and Sentiment Analysis.

**Progress:**
- [x] Study LSTM, SA and learn their application.
- [x] Prepare data.
- [ ] Build model and make predictions
- [ ] Documentation.

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

*(In progress)*
