from utils.angle_utils import angle_difference

import doctest
from math import pi
import unittest


class TestAngleUtils(unittest.TestCase):

    def test_angle_difference(self):
        self.assertAlmostEqual(-pi / 4, angle_difference(0, pi / 4))
        self.assertAlmostEqual(pi / 4, angle_difference(0, -pi / 4))
        self.assertAlmostEqual(-pi, angle_difference(0, pi))


def load_tests(loader, tests, ignore):
    """
    Function required for the unittest test discovery to find the doctests.
    """
    import utils.angle_utils
    tests.addTests(doctest.DocTestSuite(utils.angle_utils))
    return tests


if __name__ == "main":
    unittest.main()
