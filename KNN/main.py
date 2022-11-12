import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets



iris_data = pd.read_csv('../../iris-data-clean.csv')
iris_data.tail()

df = pd.DataFrame(iris_data, 
                columns=['sepal_length_cm', 'sepal width /cm', 'petal length /cm', 'petal width /cm', 'class'])
df = pd.DataFrame(iris_data)
df.tail