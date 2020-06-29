import numpy as np


class Regularizers():

    @staticmethod
    def derive_abs(weights):
        weights[weights >= 0] = 1
        weights[weights < 0] = -1

        return weights

    @staticmethod
    def l1(weights: np.ndarray, reg_strength = 0.01):
        return reg_strength * np.sum(Regularizers.derive_abs(weights),
                                  axis = 1, keepdims = True)

    @staticmethod
    def l2(weights: np.ndarray, reg_strength = 0.01):
        return reg_strength * np.sum(2 * weights, axis = 1 , keepdims = True)

    reg_methods = {"l1": l1,
                   "l2": l2}