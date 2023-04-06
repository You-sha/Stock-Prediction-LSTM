# Stock Prediction: LSTM & SA
Predicting **Kanoria Chemicals** stock price using Long Short-Term Memory and Sentiment Analysis.

**Progress:**
- [x] Study LSTM, SA and learn their application.
- [x] Prepare data.
- [ ] Build model and make predictions
- [ ] Documentation.

## Data Preparation
There were 3 million+ datapoints for the news data, and just about 3500+ for the stock data. First, I dropped news data that was from before the company's origin. Then I dropped the data that was from days when the market was closed or the stocks weren't traded. Upon analyzing it, I ascertained that there were multiple news headlines from different papers for each day, including ones useless for this purpose (entertainment, horoscopes, sports, etc.). So I kept only the useful headlines and dropped the rest. Then I randomly selected one headline for each day (since there were still multiple), and finally merged the news and stock data into one dataset.

Dataset sample:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>headline_category</th>
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33110</th>
      <td>20010928</td>
      <td>unknown</td>
      <td>Watch CM sip nimbu pani at roadside stall</td>
    </tr>
    <tr>
      <th>33111</th>
      <td>20010928</td>
      <td>entertainment.hindi.bollywood</td>
      <td>Esha Deol: Prospective Queen Bee?!</td>
    </tr>
    <tr>
      <th>33112</th>
      <td>20010928</td>
      <td>city.mumbai</td>
      <td>Govt concedes NBA demands; Medha breaks fast</td>
    </tr>
    <tr>
      <th>33113</th>
      <td>20010928</td>
      <td>pune-times</td>
      <td>Too Much of Somethings</td>
    </tr>
    <tr>
      <th>33114</th>
      <td>20010928</td>
      <td>india</td>
      <td>US should have paid heed to our warnings: PM</td>
    </tr>
    <tr>
      <th>33115</th>
      <td>20010928</td>
      <td>city.hyderabad</td>
      <td>Drive to check sound pollution in cinemas</td>
    </tr>
    <tr>
      <th>33116</th>
      <td>20010928</td>
      <td>city.mumbai</td>
      <td>Handicrafts showcase Indian state of the arts</td>
    </tr>
    <tr>
      <th>33117</th>
      <td>20010928</td>
      <td>city.hyderabad</td>
      <td>Recycling of continues batteries unchecked</td>
    </tr>
    <tr>
      <th>33118</th>
      <td>20010928</td>
      <td>lucknow-times</td>
      <td>Dancing with god's grace</td>
    </tr>
    <tr>
      <th>33119</th>
      <td>20010928</td>
      <td>entertainment.hindi.bollywood</td>
      <td>Sunny Deol: Anil Sharma's favourite!</td>
    </tr>
    <tr>
      <th>33120</th>
      <td>20010928</td>
      <td>hyderabad-times</td>
      <td>Our very own Bridget Joneses</td>
    </tr>
  </tbody>
</table>
</div>



*(In progress)*
