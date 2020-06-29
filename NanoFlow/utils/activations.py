import numpy as np


def pick_activation(activation: str):
    if activation == "relu":
        return relu
    elif activation == "sigmoid":
        return sigmoid
    elif activation == "softmax":
        return softmax
    elif activation == "normalization":
        return normalization


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


def normalization(X):
    me = np.mean(X)
    variance = np.var(X)

    return (X - me) / variance