
#  Here we are importing the iris dataset  and visualize the data perfectly



import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#ff0000", "#00FF00","#0000FF"])
from KNN_scratch import KNN

iris = datasets.load_iris()

X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state =42)

plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap =cmap, edgecolor = "k", s=20)
plt.show()


clf = KNN(k=5)
clf.fit(X_train, y_train)
predicitons = clf.predict(X_test)
print("Predicted Values are: ", predicitons)

accuracy = np.sum(predicitons==y_test) / len(y_test)
print("Accuracy is :", accuracy)