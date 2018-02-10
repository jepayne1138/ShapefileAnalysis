import unittest
import types


class TestPointWindowIter(unittest.TestCase):

    def test_point_window_iter(self):
        from analyze import point_window_iter

        test_iter = (0, 1, 2, 3, 0)
        actual = point_window_iter(test_iter)
        expected = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))


class TestNeighborWindow(unittest.TestCase):

    def test_neighbor_window_index_1(self):
        from analyze import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 1)
        expected = (0, 1, 2)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_2(self):
        from analyze import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 2)
        expected = (1, 2, 3)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_3(self):
        from analyze import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 3)
        expected = (2, 3, 4)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_start_out_of_range(self):
        from analyze import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        with self.assertRaises(IndexError):
            neighbor_window(test_seq, 0)

    def test_neighbor_window_index_end_out_of_range(self):
        from analyze import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        with self.assertRaises(IndexError):
            neighbor_window(test_seq, 4)

    def test_neighbor_window_not_enough_elements(self):
        from analyze import neighbor_window

        test_seq = (0, 1)
        with self.assertRaises(ValueError):
            neighbor_window(test_seq, 1)


class TestPointsInline(unittest.TestCase):

    def test_points_inline(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 1))
        c = np.asarray((0, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_false(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_within_tolerance(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_negative_within_tolerance(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((-0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_has_tolerence_but_point_is_beyond(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((-0.5, 1))
        c = np.asarray((0, 2))
        tolerance = 0.3
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_line_offset(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((1, 1))
        c = np.asarray((2, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_but_out_of_segment_negative(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((.9, 1))
        c = np.asarray((2, 2))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_but_out_of_segment_positive(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((2, 2.1))
        c = np.asarray((2, 2))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_but_out_of_segment_reversed(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((2, 2))
        b = np.asarray((2, 2.1))
        c = np.asarray((1, 1))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_midpoint_on_start(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 0))
        c = np.asarray((0, 1))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_end(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 1))
        c = np.asarray((0, 1))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_start_screwed_translated(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((1, 1))
        c = np.asarray((2, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_end_screwed_translated(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((2, 2))
        c = np.asarray((2, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_reverse_direction(self):
        import numpy as np
        from analyze import points_inline

        a = np.asarray((0, 2))
        b = np.asarray((0, 1))
        c = np.asarray((0, 0))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)


class TestWithinTolerance(unittest.TestCase):

    def test_within_tolerance(self):
        from analyze import within_tolerance
        actual = within_tolerance(90.001 - 90, 0.001)
        self.assertTrue(actual)

    def test_within_tolerance2(self):
        from analyze import within_tolerance
        actual = within_tolerance(90.001 - 90, 0.0001)
        self.assertFalse(actual)

    def test_within_tolerance3(self):
        from analyze import within_tolerance
        actual = within_tolerance(90 - 90.001, 0.0001)
        self.assertFalse(actual)

    def test_within_tolerance4(self):
        from analyze import within_tolerance
        actual = within_tolerance(0.5 - 0, 0.4)
        self.assertFalse(actual)

    def test_within_tolerance5(self):
        from analyze import within_tolerance
        actual = within_tolerance(0.5 - 0, 0.49, float_tol=0)
        self.assertFalse(actual)

    def test_within_tolerance_within_negative(self):
        from analyze import within_tolerance
        with self.assertRaises(ValueError):
            within_tolerance(0.1, -0.1)


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
