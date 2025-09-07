import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

np.random.seed(598)

x1, x2, x3, x4 = np.random.randn(1000), np.random.randn(1000), np.random.randn(1000), np.random.randn(1000)
epsilon = np.random.randn(1000)
y = x1 + 2*x2 - x3 + epsilon

df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
x_set, y_set = df[["x1", "x2", "x3", "x4"]].values, df[["y"]].values
x_train, x_test = x_set[:500], x_set[500:]
y_train, y_test = y_set[:500], y_set[500:]


def builtin_knn(n):
    regressor = KNeighborsRegressor(n_neighbors=n)
    regressor.fit(x_train, y_train)
    return mean_squared_error(y_test, regressor.predict(x_test))

print(builtin_knn(4))
print(builtin_knn(5))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def myKNN(xtrain, ytrain, xtest, k):
    ytest = []

    for x in xtest:
        distances_arr = [euclidean_distance(x, train) for train in xtrain]
        nearest_indices = np.argsort(distances_arr)[:k]
        nearest_neighbors = [ytrain[i] for i in nearest_indices]
        ytest.append(np.mean(nearest_neighbors))

    return ytest

def mse(test, pred):
    n = len(test)
    error = 0
    for i in range(n):
        error += ((test[i]-pred[i])**2)
    
    return (float)(error / n)


ypred = myKNN(x_train, y_train, x_test, 4)
print(mse(y_test, ypred))
ypred = myKNN(x_train, y_train, x_test, 5)
print(mse(y_test, ypred))