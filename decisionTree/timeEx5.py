import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the stock price of AMZN from the file AMZN.csv.  You could also download the data from 
# https://finance.yahoo.com/quote/AMZN/history?p=AMZN.  Perform the followings:
# 1) What is the average stock price based on the past 200 days?
# 2) Plot the graph of the historical stock price, 50 days moving average and 200 days moving average

reviews = pd.read_csv('AMZN.csv',index_col=0,
    parse_dates=True, infer_datetime_format=True)

# print(reviews['ClosingPrice'][-200:])
reviews_recent_period = reviews[-200:]

#average stock price of 200 days
print(reviews_recent_period.ClosingPrice.mean())
print(reviews_recent_period.ClosingPrice.sum())

for200Day = reviews.ClosingPrice.rolling(window=200).mean().shift(1)
for50Day = reviews.ClosingPrice.rolling(window=50).mean().shift(1)

print(reviews.ClosingPrice.head())
print()
print("rolling mean 200")
print(for200Day.head())
print()
print("rolling mean 50")
print(for50Day.head())

plt.plot(reviews.index, reviews.ClosingPrice, label='Closing Price')
plt.plot(reviews.index, for200Day, label='200 days')
plt.plot(reviews.index, for50Day, label='50 days')
plt.legend(loc='upper left')
plt.show()