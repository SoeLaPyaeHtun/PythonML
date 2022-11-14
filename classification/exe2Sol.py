import pandas as pd





data = pd.read_csv("iris-data-clean.csv")

print(data)

def myfunction(x):
    if x == "Setosa":
        return 0
    elif x == "Virginica":
        return 1
    else:
        return 2

data["class"] = data["class"].apply(myfunction)

print(data)
X = data.iloc[:,0:4]
y = data["class"]

# import seaborn as sb
# import matplotlib.pyplot as plt
# sb.pairplot(data)
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train , X_test, y_train, y_test = train_test_split(X,y,random_state=42)

print(X_train)

logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial',random_state=42)

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

print(logreg.predict([[4.9, 3.5, 1.6, 0.25]]))
print(logreg.predict([[6.0, 2.9, 5.1, 1.7]]))