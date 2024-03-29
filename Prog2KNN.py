import pandas as pd
import numpy as np
# Worked With Gino Costanzo 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def jaccard_distance(x1, x2):
    return 1 - np.sum(np.minimum(x1, x2)) / np.sum(np.maximum(x1, x2))

class KNN:
    def __init__(self, k=10, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        if self.distance == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance == 'jaccard':
            distances = [jaccard_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError('Invalid distance metric')
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    shuffled = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test = shuffled[:test_set_size]
    train = shuffled[test_set_size:]
    return X[train], X[test], y[train], y[test]

# Data Frame Example

data = [[2,2,0],[3,1,0],
        [4,1,0],[1,5,0],
        [-1,3,0],[-3,-2,0]]


df = pd.DataFrame( data, columns = ['Test1','Test2','Find'] )


features = df[['Test1', 'Test2']].values
target = df['Find'].values

# iterate though every point and calculate distance (norm), then take the values of the 6 smallest distances
# a way to do that might be indexing at a slice of an argsort

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, target, random_state=42)

# Initialize KNN 
k = 10
knn = KNN(k=k)

# Uses methods in KNN Class
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

knn_euclidean = KNN(k=10, distance='euclidean')
knn_euclidean.fit(X_train, Y_train)
predictions_euclidean = knn_euclidean.predict(X_test)

# Initialize KNN with Manhattan distance
knn_manhattan = KNN(k=10, distance='manhattan')
knn_manhattan.fit(X_train, Y_train)
predictions_manhattan = knn_manhattan.predict(X_test)

knn_jaccard = KNN(k=10, distance='jaccard')
knn_jaccard.fit(X_train, Y_train)
predictions_jaccard = knn_jaccard.predict(X_test)

accuracy_manhattan = np.mean(predictions_manhattan == Y_test)
print("Accuracy (Manhattan):", accuracy_manhattan)

accuracy_euclidean = np.mean(predictions_euclidean == Y_test)
print("Accuracy (Euclidean):", accuracy_euclidean)

accuracy_jaccard = np.mean(predictions_jaccard == Y_test)
print("Accuracy (Jaccard):", accuracy_jaccard)