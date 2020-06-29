from scipy.stats import logistic
from utils.activations import *
import numpy as np
import unittest


class TestActivations(unittest.TestCase):

    def setUp(self):
        self.test_data = np.random.randn(500, 10)

    def test_relu(self):
        activations = relu(self.test_data.copy(), False)
        derivatives = relu(self.test_data.copy(), True)

        tmp = self.test_data.copy()
        tmp[tmp < 0] = 0

        self.assertEqual(np.allclose(activations, tmp), True)

        tmp = self.test_data.copy()
        tmp[tmp <= 0] = 0
        tmp[tmp > 0] = 1

        self.assertEqual(np.allclose(derivatives, tmp), True)



if __name__ == '__main__':
    unittest.main()