import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


reviews = pd.read_csv("../../Linear_Reg_Sales.csv")
# print(reviews.head())
# print("")
# print(reviews.tail())

# print(reviews.shape)

import seaborn as sns


# plt.scatter(reviews.Advert, reviews.Sales)

# print(plt.show())


advert = reviews[['Advert']]
sales = reviews['Sales']

x_train = advert
y_train = sales

from sklearn.linear_model import LinearRegression

linReg = LinearRegression()

linReg.fit(x_train, y_train)

print(linReg.intercept_)
print(linReg.coef_)

df1 = pd.DataFrame ({
    "Advert": [11,12]
})
print(linReg.predict( df1[['Advert']]))



print(linReg.score(x_train, y_train))

# #alternative way to get r2
# from sklearn.metrics import mean_squared_error, r2_score

# y_pred = linReg.predict(x_train) 
# r2_score(y_train, y_pred) 

# plt.scatter(x_train.iloc[:,0], y_pred)  # extract only the first dimension from the x_train
# plt.plot(x_train.iloc[:,0], y_pred, color='red')
# plt.show()

# df1 = pd.DataFrame ({
#     "Advert": [1000,2000]
# })
# linReg.predict( df1[['Advert']])
