## LSTM_STOCKS

Here, I am using LSTM model to train the stocks. I have collected the stock's data [here](https://in.finance.yahoo.com/quote/TITAN.NS/history?period1=1205557200&period2=1552626000&interval=1d&filter=history&frequency=1d).

Since we are dealing with time-series data, we can't shuffle and pick random 5% or 10% of data as a out-of-sample data. For our out-of-sample data, we will take last 5% of continuous data.

 Normalizing data: We should normalize the data between -1 and 1. This will help the model to learn quicker and accurate.

 Sequences: Sequences are created to train the model. In this example, we are having sequences in following form: Adj Close price for last 60 days followed by a prediction or target for today. This is telling the model that the particular sequence of 60 days will have the target associated with it.

 Balancing the data: Sometimes, we can have the data biased towards only one particular class. remeebr that in this example, we only have two classes to train against, either up or down, ie. 1 or 0 respectively. We need to balacne the training sample so that our model should not have a bias towards only one single taget class.

## Completed:

1. Created a target feature.
2. Normalized the data.
3. Created the sequences.
4. Balanced the data.
5. Tested the model.
6. Added the results.

**Mar 16, 2019:** Results are not looking so good. Current validation accuracy is equal to 50%, equal to throwing a rock at either up or down.

## TODO:

 * Test the model in real world trading scenario.
 * Post the results.
 * Modify the features and perform the tests.
 * Fiddle around with SEQUENCE_SIZE and FUTURE_PERIOD_PREDICT.

[Tutorial](https://www.youtube.com/watch?v=ne-dpRdNReI&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=8) 
