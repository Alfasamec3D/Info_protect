import unittest

from puf_lab.information import gaussian_information, verify


class InformationTests(unittest.TestCase):
    def test_formula_matches_covariance_matrix_including_singular_limit(self):
        self.assertEqual(len(verify()), 9)
        self.assertEqual(gaussian_information(9, .25, 1), gaussian_information(1, .25, 1))

    def test_invalid_parameters(self):
        for args in ((0,.25,0), (2.5,.25,0), (3,0,0), (3,.25,-.1), (3,.25,1.1)):
            with self.assertRaises(ValueError):
                gaussian_information(*args)
