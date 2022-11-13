import numpy as np
import matplotlib.pyplot as plt


indepen_data = np.random.randint(0,101,(25,2)).astype(np.float32)
depen_data = np.random.randint(0,2,(25,1)).astype(np.float32)

red = indepen_data[depen_data.ravel() == 0] 
blue = indepen_data[depen_data.ravel() == 1] 


plt.scatter(red[:,0],red[:,1],80,'r','^')
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

newOne = np.random.randint(0,101,(1,2)).astype(np.float32)

plt.scatter(newOne[:,0],newOne[:,1],80,'g','o')


from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(indepen_data,depen_data)

result = knn.predict(newOne)
print(result)

plt.show()