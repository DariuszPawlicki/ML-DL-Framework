import numpy as np

def relu(X, derivative = False):
    if derivative == True:
        X[X > 0] = 1
        X[X <= 0] = 0

        return X

    return np.maximum(0, X)


def sigmoid(X, derivative = False):
    if derivative == True:
        return sigmoid(X) * (1 - sigmoid(X))

    return 1 / (1 + np.exp(-X))


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis = 1, keepdims = True)