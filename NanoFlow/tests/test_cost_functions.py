import unittest
from utils.cost_functions import *


class TestCostFunctions(unittest.TestCase):

    def setUp(self):
        self.test_labels = np.arange(0, 200, 1)
        self.test_predictions = np.arange(0, 200, 1)
        self.test_predictions_half_correct = np.concatenate((np.arange(0, 100, 1),
                                                             np.zeros(100,)))

        self.test_labels = self.test_labels.reshape(200, 1)
        self.test_predictions = self.test_predictions.reshape(200, 1)
        self.test_predictions_half_correct = self.test_predictions_half_correct.reshape(200, 1)

    def test_categorical_crossentropy(self):
        pass


if __name__ == '__main__':
    unittest.main()
