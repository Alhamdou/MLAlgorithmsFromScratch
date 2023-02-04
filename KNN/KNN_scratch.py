
"""
KNN is a machine learning Algorithm that help in the following:
    1. It help in calculating the distance from all other data points in the dataset
    2. It helps in getting the closet N-points 
    3. It helps in regression
        1. In getting the average of the values
    4. Classifier:
        1. get the label with majority of the votes
        

"""

# first lets create a fucntion for the eculidean distance as it helps in calculating the distance for us
import numpy as np

from collections import Counter
def euclidean_dis(x1, x2):
    euc_distance = np.sqrt(np.sum((x1-x2)**2))
    return euc_distance



class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit (self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    # helper function
    def _predict(self, x):
        
        # In computing the distance
        distances = [euclidean_dis(x, x_train) for x_train in self.X_train]
        
        # getting the value close to k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        
        # majorituy of the votes that are available in the dataset (as in the points) 
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]