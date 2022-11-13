import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn

from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns = ['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)'])

df['class'] = iris.target

print(df)

print(iris.target_names)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors=5)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2], df['class'], random_state = 42)
print(X_train.head())
