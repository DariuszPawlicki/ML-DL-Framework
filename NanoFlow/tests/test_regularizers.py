import unittest
from utils.regularizers import *


class TestRegularizers(unittest.TestCase):

    def setUp(self):
        self.test_weights = np.arange(-100, 100, 1)
        self.test_weights = self.test_weights.reshape(self.test_weights.shape[0], 1)

    def test_derive_abs(self):
        correct_derivation = np.concatenate((-np.ones(100), np.ones(100)))
        correct_derivation = correct_derivation.reshape(correct_derivation.shape[0], 1)

        self.assertTrue(np.allclose(Regularizers.derive_abs(self.test_weights), correct_derivation))

    def test_l1(self):
        self.assertEqual(Regularizers.l1(self.test_weights.T), 0)

    def test_l2(self):
        self.assertEqual(Regularizers.l2(self.test_weights.T, 1),
                         2 * np.sum(self.test_weights.T, axis = 1))



if __name__ == '__main__':
    unittest.main()