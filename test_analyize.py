import unittest
import types

import pytest


class TestAnalyize(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_window_size2(self):
        from analyze import window
        test_iter = iter((0, 1, 2, 3, 4))

        actual = window(test_iter, 2)
        expected = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_window_size3(self):
        from analyze import window
        test_iter = iter((0, 1, 2, 3, 4))

        actual = window(test_iter, 3)
        expected = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_point_window_iter(self):
        from analyze import point_window_iter

        test_iter = iter((0, 1, 2, 3, 0))
        actual = point_window_iter(test_iter)
        expected = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))


class TestWithinTolerance(unittest.TestCase):

    def test_within_tolerance(self):
        from analyze import within_tolerance
        actual = within_tolerance(90.001, 90, 0.001)
        self.assertTrue(actual)

    def test_within_tolerance2(self):
        from analyze import within_tolerance
        actual = within_tolerance(90.001, 90, 0.0001)
        self.assertFalse(actual)

    def test_within_tolerance3(self):
        from analyze import within_tolerance
        actual = within_tolerance(90, 90.001, 0.0001)
        self.assertFalse(actual)


class TestGetRadians(unittest.TestCase):

    def test_get_radians_0(self):
        import numpy as np
        from analyze import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((0, 1))

        actual = get_radians(a, b, c)
        expected = 0
        self.assertEqual(expected, actual)

    def test_get_radians_pi(self):
        import numpy as np
        from analyze import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((0, -1))

        actual = get_radians(a, b, c)
        expected = np.pi
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi(self):
        import numpy as np
        from analyze import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((1, 0))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi_diff_magnitude(self):
        import numpy as np
        from analyze import get_radians

        a = np.asarray((0, 3))
        b = np.asarray((0, 0))
        c = np.asarray((1.7, 0))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi_all_offset(self):
        import numpy as np
        from analyze import get_radians

        a = np.asarray((1, 3))
        b = np.asarray((1, 1))
        c = np.asarray((1.7, 1))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)
