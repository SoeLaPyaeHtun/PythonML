import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

reviews = pd.read_csv("K-NN_Weight.csv")
print(reviews)

df_cat = reviews

df_cat.loc[reviews['Weight']<50, 'Weight'] = 0
df_cat.loc[(reviews['Weight']>=50) & (reviews['Weight']<65), 'Weight'] = 1
df_cat.loc[reviews['Weight'] >= 65, 'Weight'] = 2

# print(df_cat.head())

# fig, ax = plt.subplots(figsize = (5,5))
# df_zero = df_cat.loc[df_cat['Weight'] == 0]
# df_one = df_cat.loc[df_cat['Weight'] == 1]
# df_two = df_cat.loc[df_cat['Weight'] == 2]
# ax.scatter(df_zero['Age'], df_zero['Height'])
# ax.scatter(df_one['Age'], df_one['Height'])
# ax.scatter(df_two['Age'], df_two['Height'])
# #plt.ylim(4, 7)
# plt.show()

from sklearn.model_selection import train_test_split


X = df_cat.iloc[:, 0:2]
y = df_cat['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


print(X_test)
print()
print(X_train.head())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

