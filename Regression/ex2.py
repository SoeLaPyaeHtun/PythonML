import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import datasets

boston = datasets.load_boston()

df = pd.DataFrame(boston.data, 
                  columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])



# fig, ax = plt.subplots(figsize=(10,10)) 
# ax.scatter(df['RM'], df['RAD'])
# ax.grid()
# plt.show()


from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train = df[['RM']]
y_train = df['RAD']

print(X_train.shape)
print(y_train.shape)

lModel = linear_model.LinearRegression()

# train the model to fit the training data, finding the coef and intercept
model = lModel.fit(X_train, y_train)

print(lModel.predict([[4.67]]))

print("intercept",lModel.intercept_)
print("coef",lModel.coef_)

print(lModel.score(X_train, y_train))
y_pred = lModel.predict(X_train)
plt.scatter(X_train.iloc[:,0], y_train)
plt.plot(X_train.iloc[:,0], y_pred, color='red')
plt.show()



fig, ax = plt.subplots(figsize = (10,10))

ax.plot(X_train.iloc[:,0], y_pred, c = 'red', label='Test data')
ax.scatter(X_train.iloc[:,0], y_train, c = 'blue', label='Test points')
ax.set(xlabel='average rooms per dwelling', ylabel='median of owner home / $1000')
ax.grid()
plt.show()