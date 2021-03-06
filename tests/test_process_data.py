import unittest
import types


def assertArrayEquals(testcase, arr1, arr2):
    """Helper to assert sequence numpy array equality"""
    from itertools import zip_longest
    import numpy as np
    testcase.assertTrue(
        all([
            np.array_equal(e, a)
            for e, a
            in zip_longest(arr1, arr2)
        ])
    )


class TestLessOrClose(unittest.TestCase):

    def test_less_or_close_simple(self):
        from shapeanalysis.process_data import less_or_close
        self.assertTrue(less_or_close(0, 1))

    def test_less_or_close_greater(self):
        from shapeanalysis.process_data import less_or_close
        self.assertFalse(less_or_close(1, 0))

    def test_less_or_close_strict_equal(self):
        from shapeanalysis.process_data import less_or_close
        self.assertTrue(less_or_close(1, 1))

    def test_less_or_close_strict_float_close(self):
        from shapeanalysis.process_data import less_or_close
        # 1.01 - 1 = 0.010000000000000009
        self.assertTrue(less_or_close(1.01 - 1, 0.01))


class TestModifiedPointList(unittest.TestCase):

    def test_modified_point_list_too_small(self):
        from shapeanalysis.process_data import modified_point_list
        with self.assertRaises(ValueError):
            modified_point_list((1, 2))

    def test_modified_point_list_verify_start_and_end_same(self):
        from shapeanalysis.process_data import modified_point_list
        with self.assertRaises(ValueError):
            modified_point_list((0, 1, 2))

    def test_modified_point_list_verify_each_element_len2(self):
        from shapeanalysis.process_data import modified_point_list
        with self.assertRaises(ValueError):
            modified_point_list(((0, 0), (0, 1), (0, 2), 0))

    def test_modified_point_list_verify_second_element_wrapped(self):
        import numpy as np
        from shapeanalysis.process_data import modified_point_list
        actual = modified_point_list(
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 0))
        )
        expected = [
            np.asarray((0, 0)),
            np.asarray((0, 1)),
            np.asarray((0, 2)),
            np.asarray((0, 3)),
            np.asarray((0, 0)),
            np.asarray((0, 1)),
        ]

        # NumPy has no implicit array equality,
        # so zip to longest and check all np.array_equals to assert accuracy
        assertArrayEquals(self, expected, actual)

    def test_modified_point_list_input(self):
        import numpy as np
        from shapeanalysis.process_data import modified_point_list
        actual = modified_point_list(
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 0)]
        )
        expected = [
            np.asarray((0, 0)),
            np.asarray((0, 1)),
            np.asarray((0, 2)),
            np.asarray((0, 3)),
            np.asarray((0, 0)),
            np.asarray((0, 1)),
        ]

        # NumPy has no implicit array equality,
        # so zip to longest and check all np.array_equals to assert accuracy
        assertArrayEquals(self, expected, actual)


class TestPointWindowIter(unittest.TestCase):

    def test_point_window_iter(self):
        from shapeanalysis.process_data import point_window_iter

        test_iter = (0, 1, 2, 3, 0)
        actual = point_window_iter(test_iter)
        expected = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))


class TestNeighborWindow(unittest.TestCase):

    def test_neighbor_window_index_1(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 1)
        expected = (0, 1, 2)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_2(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 2)
        expected = (1, 2, 3)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_3(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 3)
        expected = (2, 3, 4)
        self.assertEqual(expected, actual)

    def test_neighbor_window_index_start_out_of_range(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        with self.assertRaises(IndexError):
            neighbor_window(test_seq, 0)

    def test_neighbor_window_index_end_out_of_range(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        with self.assertRaises(IndexError):
            neighbor_window(test_seq, 4)

    def test_neighbor_window_index_end_out_of_range_count2(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        with self.assertRaises(IndexError):
            neighbor_window(test_seq, 3, count=2)

    def test_neighbor_window_not_enough_elements(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1)
        with self.assertRaises(ValueError):
            neighbor_window(test_seq, 1)

    def test_neighbor_window_not_enough_elements_count2(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 3)
        with self.assertRaises(ValueError):
            neighbor_window(test_seq, 1, count=2)

    def test_neighbor_window_count2(self):
        from shapeanalysis.process_data import neighbor_window

        test_seq = (0, 1, 2, 3, 4)
        actual = neighbor_window(test_seq, 1, count=2)
        expected = (0, 1, 2, 3)
        self.assertEqual(expected, actual)


class TestPointsInline(unittest.TestCase):

    def test_points_inline(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 1))
        c = np.asarray((0, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_false(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_within_tolerance(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_negative_within_tolerance(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((-0.1, 1))
        c = np.asarray((0, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_has_tolerence_but_point_is_beyond(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((-0.5, 1))
        c = np.asarray((0, 2))
        tolerance = 0.3
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_line_offset(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((1, 1))
        c = np.asarray((2, 2))
        tolerance = 0
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_but_out_of_segment_negative(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((.9, 1))
        c = np.asarray((2, 2))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_but_out_of_segment_positive(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((2, 2.1))
        c = np.asarray((2, 2))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_but_out_of_segment_reversed(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((2, 2))
        b = np.asarray((2, 2.1))
        c = np.asarray((1, 1))
        tolerance = 0.5
        actual = points_inline(a, b, c, tolerance)
        self.assertFalse(actual)

    def test_points_inline_midpoint_on_start(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 0))
        c = np.asarray((0, 1))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_end(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 0))
        b = np.asarray((0, 1))
        c = np.asarray((0, 1))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_start_screwed_translated(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((1, 1))
        c = np.asarray((2, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_on_end_screwed_translated(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((1, 1))
        b = np.asarray((2, 2))
        c = np.asarray((2, 2))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)

    def test_points_inline_midpoint_reverse_direction(self):
        import numpy as np
        from shapeanalysis.process_data import points_inline

        a = np.asarray((0, 2))
        b = np.asarray((0, 1))
        c = np.asarray((0, 0))
        tolerance = 0.1
        actual = points_inline(a, b, c, tolerance)
        self.assertTrue(actual)


class TestMidpointProjectionOffset(unittest.TestCase):

    def test_midpoint_projection_offset_1(self):
        import numpy as np
        from shapeanalysis.process_data import midpoint_projection_offset

        a = np.asarray((0, 0))
        b = np.asarray((1, 1))
        c = np.asarray((0, 2))
        actual = midpoint_projection_offset(a, b, c)
        expected = 1
        self.assertAlmostEqual(expected, actual)

    def test_midpoint_projection_offset_2(self):
        import numpy as np
        from shapeanalysis.process_data import midpoint_projection_offset

        a = np.asarray((0, 0))
        b = np.asarray((1, 2))
        c = np.asarray((2, 0))
        actual = midpoint_projection_offset(a, b, c)
        expected = 2
        self.assertAlmostEqual(expected, actual)


class TestBetweenNeighbors(unittest.TestCase):

    def test_between_neighbors_out_of_segment_positive(self):
        import numpy as np
        from shapeanalysis.process_data import between_neighbors

        a = np.asarray((1, 1))
        b = np.asarray((2, 2.1))
        c = np.asarray((2, 2))
        actual = between_neighbors(a, b, c)
        self.assertFalse(actual)

    def test_between_neighbors_but_out_of_segment_reversed(self):
        import numpy as np
        from shapeanalysis.process_data import between_neighbors

        a = np.asarray((2, 2))
        b = np.asarray((2, 2.1))
        c = np.asarray((1, 1))
        actual = between_neighbors(a, b, c)
        self.assertFalse(actual)

    def test_between_neighbors_in_segment(self):
        import numpy as np
        from shapeanalysis.process_data import between_neighbors

        a = np.asarray((2, 2))
        b = np.asarray((1, 2))
        c = np.asarray((1, 1))
        actual = between_neighbors(a, b, c)
        self.assertTrue(actual)

    def test_between_neighbors_on_start(self):
        import numpy as np
        from shapeanalysis.process_data import between_neighbors

        a = np.asarray((2, 2))
        b = np.asarray((1, 1))
        c = np.asarray((1, 1))
        actual = between_neighbors(a, b, c)
        self.assertTrue(actual)

    def test_between_neighbors_on_end(self):
        import numpy as np
        from shapeanalysis.process_data import between_neighbors

        a = np.asarray((2, 2))
        b = np.asarray((2, 2))
        c = np.asarray((1, 1))
        actual = between_neighbors(a, b, c)
        self.assertTrue(actual)


class TestWithinTolerance(unittest.TestCase):

    def test_within_tolerance(self):
        from shapeanalysis.process_data import within_tolerance
        actual = within_tolerance(90.001 - 90, 0.001)
        self.assertTrue(actual)

    def test_within_tolerance2(self):
        from shapeanalysis.process_data import within_tolerance
        actual = within_tolerance(90.001 - 90, 0.0001)
        self.assertFalse(actual)

    def test_within_tolerance3(self):
        from shapeanalysis.process_data import within_tolerance
        actual = within_tolerance(90 - 90.001, 0.0001)
        self.assertFalse(actual)

    def test_within_tolerance4(self):
        from shapeanalysis.process_data import within_tolerance
        actual = within_tolerance(0.5 - 0, 0.4)
        self.assertFalse(actual)

    def test_within_tolerance5(self):
        from shapeanalysis.process_data import within_tolerance
        actual = within_tolerance(0.5 - 0, 0.49, float_tol=0)
        self.assertFalse(actual)

    def test_within_tolerance_within_negative(self):
        from shapeanalysis.process_data import within_tolerance
        with self.assertRaises(ValueError):
            within_tolerance(0.1, -0.1)


class TestGetRadians(unittest.TestCase):

    def test_get_radians_0(self):
        import numpy as np
        from shapeanalysis.process_data import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((0, 1))

        actual = get_radians(a, b, c)
        expected = 0
        self.assertEqual(expected, actual)

    def test_get_radians_pi(self):
        import numpy as np
        from shapeanalysis.process_data import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((0, -1))

        actual = get_radians(a, b, c)
        expected = np.pi
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi(self):
        import numpy as np
        from shapeanalysis.process_data import get_radians

        a = np.asarray((0, 1))
        b = np.asarray((0, 0))
        c = np.asarray((1, 0))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi_diff_magnitude(self):
        import numpy as np
        from shapeanalysis.process_data import get_radians

        a = np.asarray((0, 3))
        b = np.asarray((0, 0))
        c = np.asarray((1.7, 0))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)

    def test_get_radians_half_pi_all_offset(self):
        import numpy as np
        from shapeanalysis.process_data import get_radians

        a = np.asarray((1, 3))
        b = np.asarray((1, 1))
        c = np.asarray((1.7, 1))

        actual = get_radians(a, b, c)
        expected = np.pi / 2
        self.assertEqual(expected, actual)


class TestPointDataList(unittest.TestCase):

    def test_point_data_list(self):
        import numpy as np
        from shapeanalysis.process_data import PointData, point_data_list

        valid_point_list = [
            np.asarray((0, 0)),
            np.asarray((1, 1)),
            np.asarray((2, 2)),
            np.asarray((0, 3)),
            np.asarray((0, 0)),
            np.asarray((1, 1)),
        ]
        actual = list(point_data_list(valid_point_list))
        expected = [
            PointData(np.asarray((0, 0)), np.asarray((1, 1)), np.asarray((2, 2))),
            PointData(np.asarray((1, 1)), np.asarray((2, 2)), np.asarray((0, 3))),
            PointData(np.asarray((2, 2)), np.asarray((0, 3)), np.asarray((0, 0))),
            PointData(np.asarray((0, 3)), np.asarray((0, 0)), np.asarray((1, 1))),
        ]
        self.assertEqual(expected, actual)


class TestRemoveInsignificant(unittest.TestCase):

    def test_remove_insignificant(self):
        import numpy as np
        from shapeanalysis.process_data import PointData, remove_insignificant

        point_list = [
            np.asarray((0, 0)),
            np.asarray((1, 1)),
            np.asarray((2, 2)),
            np.asarray((0, 3)),
            np.asarray((0, 0)),
            np.asarray((1, 1)),
        ]
        data_list = [
            PointData(np.asarray((0, 0)), np.asarray((1, 1)), np.asarray((2, 2))),
            PointData(np.asarray((1, 1)), np.asarray((2, 2)), np.asarray((0, 3))),
            PointData(np.asarray((2, 2)), np.asarray((0, 3)), np.asarray((0, 0))),
            PointData(np.asarray((0, 3)), np.asarray((0, 0)), np.asarray((1, 1))),
        ]

        actual = remove_insignificant(point_list, data_list, 0.1)
        expected = [
            np.asarray((0, 0)),
            np.asarray((2, 2)),
            np.asarray((0, 3)),
            np.asarray((0, 0)),
            np.asarray((2, 2)),
        ]
        assertArrayEquals(self, expected, actual)

    def test_remove_insignificant_removal_order(self):
        import numpy as np
        from shapeanalysis.process_data import PointData, remove_insignificant

        point_list = [
            np.asarray((0, 0)),
            np.asarray((0.5858, 1.414)),  # removed first
            np.asarray((2, 2)),           # not removed as over tolerance after first removal
            np.asarray((3.414, 1.414)),   # not this it second to be removed
            np.asarray((4, 0)),           # no longer any points under tolerance
            np.asarray((0, 0)),
            np.asarray((0.5858, 1.414)),
        ]
        data_list = [
            PointData(np.asarray((0, 0)), np.asarray((0.5858, 1.414)), np.asarray((2, 2))),
            PointData(np.asarray((0.5858, 1.414)), np.asarray((2, 2)), np.asarray((3.414, 1.414))),
            PointData(np.asarray((2, 2)), np.asarray((3.414, 1.414)), np.asarray((4, 0))),
            PointData(np.asarray((3.414, 1.414)), np.asarray((4, 0)), np.asarray((0, 0))),
            PointData(np.asarray((4, 0)), np.asarray((0, 0)), np.asarray((0.5858, 1.414))),
        ]

        actual = remove_insignificant(point_list, data_list, 0.6)
        expected = [
            np.asarray((0, 0)),
            np.asarray((2, 2)),
            np.asarray((4, 0)),
            np.asarray((0, 0)),
            np.asarray((2, 2)),
        ]
        assertArrayEquals(self, expected, actual)

    def test_remove_insignificant_origin_removal(self):
        import numpy as np
        from shapeanalysis.process_data import PointData, remove_insignificant

        point_list = [
            np.asarray((0.5858, 1.414)),
            np.asarray((2, 2)),
            np.asarray((3.414, 1.414)),
            np.asarray((4, 0)),
            np.asarray((0, 0)),
            np.asarray((0.5858, 1.414)),
            np.asarray((2, 2)),
        ]
        data_list = [
            PointData(np.asarray((0.5858, 1.414)), np.asarray((2, 2)), np.asarray((3.414, 1.414))),
            PointData(np.asarray((2, 2)), np.asarray((3.414, 1.414)), np.asarray((4, 0))),
            PointData(np.asarray((3.414, 1.414)), np.asarray((4, 0)), np.asarray((0, 0))),
            PointData(np.asarray((4, 0)), np.asarray((0, 0)), np.asarray((0.5858, 1.414))),
            PointData(np.asarray((0, 0)), np.asarray((0.5858, 1.414)), np.asarray((2, 2))),
        ]

        actual = remove_insignificant(point_list, data_list, 0.6)
        expected = [
            np.asarray((0, 0)),
            np.asarray((2, 2)),
            np.asarray((4, 0)),
            np.asarray((0, 0)),
            np.asarray((2, 2)),
        ]
        assertArrayEquals(self, expected, actual)


class TestSignificantPoints(unittest.TestCase):

    def test_significant_points(self):
        import numpy as np
        from shapeanalysis.process_data import significant_points

        input_points = [
            (0.5858, 1.414),
            (2, 2),
            (3.414, 1.414),
            (4, 0),
            (0, 0),
            (0.5858, 1.414),
        ]
        tolerance = 0.6

        actual = significant_points(input_points, tolerance)
        expected = [
            np.asarray((0, 0)),
            np.asarray((2, 2)),
            np.asarray((4, 0)),
            np.asarray((0, 0)),
            np.asarray((2, 2)),
        ]
        assertArrayEquals(self, expected, actual)


class TestOrthogonal(unittest.TestCase):

    def test_orthogonal_equal(self):
        import numpy as np
        from shapeanalysis.process_data import orthogonal

        p1 = np.asarray((0, 0))
        p2 = np.asarray((0, 1))
        p3 = np.asarray((1, 1))
        actual = orthogonal(p1, p2, p3, 0.04)
        self.assertTrue(actual)

    def test_orthogonal_equal_reverse(self):
        import numpy as np
        from shapeanalysis.process_data import orthogonal

        p1 = np.asarray((1, 1))
        p2 = np.asarray((0, 1))
        p3 = np.asarray((0, 0))
        actual = orthogonal(p1, p2, p3, 0.04)
        self.assertTrue(actual)

    def test_orthogonal_outside_tolerance(self):
        import numpy as np
        from shapeanalysis.process_data import orthogonal

        p1 = np.asarray((0, 1))
        p2 = np.asarray((0, 0))
        p3 = np.asarray((1, 0.045))
        actual = orthogonal(p1, p2, p3, 0.04)
        self.assertFalse(actual)

    def test_orthogonal_within_tolerance1(self):
        import numpy as np
        from shapeanalysis.process_data import orthogonal

        p1 = np.asarray((0, 1))
        p2 = np.asarray((0, 0))
        p3 = np.asarray((1, 0.045))
        actual = orthogonal(p1, p2, p3, 0.05)
        self.assertTrue(actual)

    def test_orthogonal_within_tolerance2(self):
        import numpy as np
        from shapeanalysis.process_data import orthogonal

        p1 = np.asarray((0, 1))
        p2 = np.asarray((0, 0))
        p3 = np.asarray((1, 0.04))
        actual = orthogonal(p1, p2, p3, 0.04)
        self.assertTrue(actual)


class TestSameSide(unittest.TestCase):

    def test_same_side_right(self):
        import numpy as np
        from shapeanalysis.process_data import same_side

        start = np.asarray((0, 0))
        end = np.asarray((1, 1))
        p1 = np.asarray((1, 0))
        p2 = np.asarray((2, 1))

        actual = same_side(p1, start, end, p2)
        self.assertTrue(actual)

    def test_same_side_left(self):
        import numpy as np
        from shapeanalysis.process_data import same_side

        start = np.asarray((0, 0))
        end = np.asarray((1, 1))
        p1 = np.asarray((0, 1))
        p2 = np.asarray((1, 2))

        actual = same_side(p1, start, end, p2)
        self.assertTrue(actual)

    def test_same_side_different_sides(self):
        import numpy as np
        from shapeanalysis.process_data import same_side

        start = np.asarray((0, 0))
        end = np.asarray((1, 1))
        p1 = np.asarray((1, 0))
        p2 = np.asarray((1, 2))

        actual = same_side(p1, start, end, p2)
        self.assertFalse(actual)


class TestHasBox(unittest.TestCase):

    def test_has_box_non_found(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0.5858, 1.414),
            (2, 2),
            (3.414, 1.414),
            (4, 0),
            (0, 0),
            (0.5858, 1.414),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertFalse(actual)

    def test_has_box_two_found(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (0, 20),
            (10, 20),
            (20, 10),
            (20, 0),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertTrue(actual)

    def test_has_box_two_found_both_out_of_size(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (0, 85),
            (1, 85),
            (2, 1),
            (2, 0),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertFalse(actual)

    def test_has_box_two_found_low_in_change_bounds(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (0, 40),
            (5, 40),
            (6, 20),
            (6, 0),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03
        low = 5
        high = 30

        actual = has_box(input_points, tolerance, angle_tolerance, low, high)
        self.assertTrue(actual)

    def test_has_box_two_found_high_in_change_bounds(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (0, 90),
            (5, 90),
            (6, 20),
            (6, 0),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03
        low = 10
        high = 100

        actual = has_box(input_points, tolerance, angle_tolerance, low, high)
        self.assertTrue(actual)

    def test_has_box_two_found_both_out_change_bounds(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (0, 40),
            (20, 40),
            (30, 20),
            (30, 0),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03
        low = 32
        high = 38

        actual = has_box(input_points, tolerance, angle_tolerance, low, high)
        self.assertFalse(actual)

    def test_has_box_one_right_angle_found(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (15, 0),
            (0, 15),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertFalse(actual)

    def test_has_box_multiple_found(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (15, 0),
            (15, 15),
            (0, 15),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertTrue(actual)

    def test_has_box_semicircle_single(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 0),
            (15, 0),
            (15, 15),
            (0, 15),
            (0, 0),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertTrue(actual)

    def test_has_box_parallelogram(self):
        from shapeanalysis.process_data import has_box

        input_points = [
            (0, 1),
            (15, 0),
            (30, 1),
            (15, 2),
            (0, 1),
        ]
        tolerance = 0.6
        angle_tolerance = 0.03

        actual = has_box(input_points, tolerance, angle_tolerance)
        self.assertFalse(actual)


class TestDistance(unittest.TestCase):

    def test_distance1(self):
        import numpy as np
        from shapeanalysis.process_data import distance

        p1 = np.asarray((0, 0))
        p2 = np.asarray((0, 1))

        actual = distance(p1, p2)
        expected = 1
        self.assertAlmostEqual(expected, actual)

    def test_distance2(self):
        import numpy as np
        from shapeanalysis.process_data import distance

        p1 = np.asarray((0, 0))
        p2 = np.asarray((0, 2))

        actual = distance(p1, p2)
        expected = 2
        self.assertAlmostEqual(expected, actual)

    def test_distance3(self):
        import numpy as np
        from shapeanalysis.process_data import distance

        p1 = np.asarray((0, 0))
        p2 = np.asarray((1, 1.7320508076))

        actual = distance(p1, p2)
        expected = 2
        self.assertAlmostEqual(expected, actual)


class TestCentroid(unittest.TestCase):

    def test_centroid_square(self):
        import numpy as np
        from shapeanalysis.process_data import centroid

        points = np.asarray([
            np.asarray((0, 0)),
            np.asarray((0, 2)),
            np.asarray((2, 2)),
            np.asarray((2, 0)),
        ])
        actual = centroid(points)
        expected = np.asarray((1, 1))
        self.assertTrue(np.allclose(expected, actual))

    def test_centroid_square_with_repeat(self):
        import numpy as np
        from shapeanalysis.process_data import centroid

        points = np.asarray([
            np.asarray((0, 0)),
            np.asarray((0, 2)),
            np.asarray((2, 2)),
            np.asarray((2, 0)),
            np.asarray((0, 0)),
        ])
        actual = centroid(points)
        expected = np.asarray((1, 1))
        self.assertTrue(np.allclose(expected, actual))

    def test_centroid_triangle(self):
        import numpy as np
        from shapeanalysis.process_data import centroid

        points = np.asarray([
            np.asarray((0, 0)),
            np.asarray((2, 2)),
            np.asarray((4, 0)),
        ])
        actual = centroid(points)
        expected = np.asarray((2, 2 / 3))
        self.assertTrue(np.allclose(expected, actual))


class TestSplitList(unittest.TestCase):

    def test_split_list_empty(self):
        from shapeanalysis.process_data import split_list

        original = []
        split_indexes = []
        actual = split_list(original, split_indexes)
        expected = [[]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_only_zero(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3]
        split_indexes = [0]
        actual = split_list(original, split_indexes)
        expected = [[1, 2, 3]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_only_empty(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3]
        split_indexes = []
        actual = split_list(original, split_indexes)
        expected = [[1, 2, 3]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_none(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3]
        split_indexes = None
        actual = split_list(original, split_indexes)
        expected = [[1, 2, 3]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_split1(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3, 4, 5]
        split_indexes = [0, 2]
        actual = split_list(original, split_indexes)
        expected = [[1, 2], [3, 4, 5]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_split2(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3, 4, 5]
        split_indexes = [0, 2, 4]
        actual = split_list(original, split_indexes)
        expected = [[1, 2], [3, 4], [5]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))

    def test_split_list_split2_no_leading(self):
        from shapeanalysis.process_data import split_list

        original = [1, 2, 3, 4, 5]
        split_indexes = [2, 4]
        actual = split_list(original, split_indexes)
        expected = [[1, 2], [3, 4], [5]]

        self.assertIsInstance(actual, types.GeneratorType)
        self.assertEqual(expected, list(actual))


class TestNearestDistances(unittest.TestCase):

    def assertDictArrayValuesEqual(self, expected, actual):
        import numpy as np
        try:
            if set(expected.keys()) == set(actual.keys()):
                for x in expected.keys():
                    if not np.allclose(expected[x], actual[x]):
                        raise AssertionError(f'{expected[x]} != {actual[x]}')
        except AttributeError:
            raise AssertionError('dictionary keys do not match')

    def test_nearest_distances_1_nearest(self):
        import numpy as np
        from shapeanalysis.process_data import nearest_distances

        points = [
            np.asarray((0, 0)),
            np.asarray((0, 1)),
            np.asarray((3, 0)),
            np.asarray((0, 2.5)),
        ]

        actual = nearest_distances(points, num_nearest=1)
        expected = {
            np.array((0, 0), dtype=np.float).tobytes(): np.array((1,), dtype=np.float),
            np.array((0, 1), dtype=np.float).tobytes(): np.array((1,), dtype=np.float),
            np.array((3, 0), dtype=np.float).tobytes(): np.array((3,), dtype=np.float),
            np.array((0, 2.5), dtype=np.float).tobytes(): np.array((1.5,), dtype=np.float),
        }

        self.assertDictArrayValuesEqual(expected, actual)

    def test_nearest_distances_2_nearest(self):
        import numpy as np
        from shapeanalysis.process_data import nearest_distances

        points = [
            np.asarray((0, 0)),
            np.asarray((0, 1)),
            np.asarray((3, 0)),
            np.asarray((0, 2.5)),
        ]

        actual = nearest_distances(points, num_nearest=2)
        expected = {
            np.array((0, 0), dtype=np.float).tobytes(): np.array((1, 2.5), dtype=np.float),
            np.array((0, 1), dtype=np.float).tobytes(): np.array((1, 1.5), dtype=np.float),
            np.array((3, 0), dtype=np.float).tobytes(): np.array((3, 3.16227766), dtype=np.float),
            np.array((0, 2.5), dtype=np.float).tobytes(): np.array((1.5, 2.5), dtype=np.float),
        }

        self.assertDictArrayValuesEqual(expected, actual)
        self.assertTrue(actual)


class TestGetPointIndexByValue(unittest.TestCase):

    def test_get_point_index_by_value(self):
        import numpy as np
        from shapeanalysis.process_data import get_point_index_by_value

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((0, 1)),
        ]
        search_point = np.array((2, 1))
        actual = get_point_index_by_value(points, search_point)

        expected = 2
        self.assertEqual(expected, actual)

    def test_get_point_index_by_value_duplicate(self):
        import numpy as np
        from shapeanalysis.process_data import get_point_index_by_value

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((1, 2)),
            np.array((0, 1)),
        ]
        search_point = np.array((1, 2))
        actual = get_point_index_by_value(points, search_point)
        expected = 1
        self.assertEqual(expected, actual)


class TestGetTopPoint(unittest.TestCase):

    def test_get_top_point(self):
        import numpy as np
        from shapeanalysis.process_data import get_top_point

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((0, 1)),
        ]
        actual = get_top_point(points)

        expected = np.array((1, 2))
        self.assertTrue(np.array_equal(expected, actual))

    def test_get_top_point_two_equal_returns_right(self):
        import numpy as np
        from shapeanalysis.process_data import get_top_point

        points = [
            np.array((0, 0)),
            np.array((0, 1)),
            np.array((1, 1)),
            np.array((1, 0)),
            np.array((0, 0)),
        ]
        actual = get_top_point(points)

        expected = np.array((1, 1))
        self.assertTrue(np.array_equal(expected, actual))


class TestRemoveArrayWrap(unittest.TestCase):

    def test_remove_array_wrap(self):
        import numpy as np
        from shapeanalysis.process_data import remove_array_wrap

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
        ]
        actual = remove_array_wrap(points)

        expected = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
        ]
        self.assertTrue(np.array_equal(expected, actual))

    def test_remove_array_wrap_first(self):
        import numpy as np
        from shapeanalysis.process_data import remove_array_wrap

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((0, 1)),
        ]
        actual = remove_array_wrap(points)

        expected = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
        ]
        self.assertTrue(np.array_equal(expected, actual))

    def test_remove_array_wrap_first_2(self):
        import numpy as np
        from shapeanalysis.process_data import remove_array_wrap

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((0, 1)),
            np.array((1, 2)),
        ]
        actual = remove_array_wrap(points)

        expected = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
        ]
        self.assertTrue(np.array_equal(expected, actual))

    def test_remove_array_wrap_first_2(self):
        import numpy as np
        from shapeanalysis.process_data import remove_array_wrap

        points = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
            np.array((0, 1)),
            np.array((1, 2)),
        ]
        actual = remove_array_wrap(points)

        expected = [
            np.array((0, 1)),
            np.array((1, 2)),
            np.array((2, 1)),
            np.array((1, 0)),
        ]
        self.assertTrue(np.array_equal(expected, actual))


class TestMidLineRotation(unittest.TestCase):

    def test_mid_line_rotation(self):
        import numpy as np
        from shapeanalysis.process_data import mid_line_rotation

        left_point = np.array((1, 2))
        right_point = np.array((2, 1))
        actual = mid_line_rotation(left_point, right_point)

        expected = np.pi / 4
        self.assertAlmostEqual(expected, actual)

    def test_mid_line_rotation_horizontal(self):
        import numpy as np
        from shapeanalysis.process_data import mid_line_rotation

        left_point = np.array((0, 0))
        right_point = np.array((0, 1))
        actual = mid_line_rotation(left_point, right_point)

        expected = 0
        self.assertAlmostEqual(expected, actual)

    def test_mid_line_rotation_angle_small(self):
        import numpy as np
        from shapeanalysis.process_data import mid_line_rotation

        left_point = np.array((0, 0))
        right_point = np.array((1, 0.41421356237))
        actual = mid_line_rotation(left_point, right_point)

        expected = np.pi / 8
        self.assertAlmostEqual(expected, actual)

    def test_mid_line_rotation_angle_small_reflected(self):
        import numpy as np
        from shapeanalysis.process_data import mid_line_rotation

        left_point = np.array((0, 0))
        right_point = np.array((1, -0.41421356237))
        actual = mid_line_rotation(left_point, right_point)

        expected = np.pi / 8
        self.assertAlmostEqual(expected, actual)
