from numpy import ndarray, sum, abs


class Regularizers():
    @staticmethod
    def l1(weights: ndarray, reg_strength = 0.01):
        return reg_strength * sum(abs(weights), axis = 1, keepdims = True)

    @staticmethod
    def l2(weights: ndarray, reg_strength = 0.01):
        return reg_strength * sum(weights**2, axis = 1 , keepdims = True)

    reg_methods = {"l1": l1,
                   "l2": l2}