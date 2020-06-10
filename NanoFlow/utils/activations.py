from numpy import exp, maximum, sum, var, mean


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

    return maximum(0, X)


def sigmoid(X, derivative = False):
    if derivative == True:
        return sigmoid(X) * (1 - sigmoid(X))

    return 1 / (1 + exp(-X))


def softmax(X):
    return exp(X) / sum(exp(X), axis = 1, keepdims = True)


def normalization(X):
    me = mean(X)
    variance = var(X)

    return (X - me) / variance