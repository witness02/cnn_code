import unittest

from first_assign.cnn_with_numpy import *


class TestCnnWithNumpy(unittest.TestCase):

    def test_init(self):
        self.assertEqual('a', 'a')

    def test_conv_single_step(self):
        np.random.seed(1)
        a_slice_prev = np.random.randn(4, 4, 3)
        W = np.random.randn(4, 4, 3)
        b = np.random.randn(1, 1, 1)

        Z = conv_single_step(a_slice_prev, W, b)
        print("Z =", Z)

if __name__ == '__main_':
    unittest.main()