from scipy.special import expit, softmax as scipy_softmax
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

    def test_sigmoid(self):
        correct_activations = expit(self.test_data)
        correct_derivatives = expit(self.test_data) * (1 - expit(self.test_data))

        self.assertTrue(np.allclose(correct_activations, sigmoid(self.test_data, False)))
        self.assertTrue(np.allclose(correct_derivatives, sigmoid(self.test_data, True)))

    def test_softmax(self):
        correct_activations = scipy_softmax(self.test_data, axis = 1)

        self.assertTrue(np.allclose(correct_activations, softmax(self.test_data)))


if __name__ == '__main__':
    unittest.main()