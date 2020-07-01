import unittest
from utils.metrics import *


class TestMetrics(unittest.TestCase):

    def setUp(self):

        self.test_labels = np.arange(0, 200, 1)
        self.test_predictions = np.arange(0, 200, 1)
        self.test_half_correct = np.concatenate((np.arange(0, 100, 1),
                                                 np.zeros(100,)))

        self.test_labels = self.test_labels.reshape(200, 1)
        self.test_predictions = self.test_predictions.reshape(200, 1)
        self.test_half_correct = self.test_half_correct.reshape(200, 1)

    def test_accuracy(self):

        tested_accuracy_full = np.sum((self.test_labels == self.test_predictions))
        tested_accuracy_full /= self.test_labels.shape[0]
        tested_accuracy_full *= 100

        tested_accuracy_half= np.sum((self.test_labels == self.test_half_correct))
        tested_accuracy_half /= self.test_labels.shape[0]
        tested_accuracy_half *= 100

        self.assertEqual(tested_accuracy_full, 100)
        self.assertEqual(tested_accuracy_half, 50)



if __name__ == '__main__':
    unittest.main()