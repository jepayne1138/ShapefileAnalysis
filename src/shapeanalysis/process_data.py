from math import isclose

import numpy as np
import scipy.spatial


class PointData:

    def __init__(self, left, point, right):
        self.left = left
        self.point = point
        self.right = right
        self.between = between_neighbors(left, point, right)
        self.offset = midpoint_projection_offset(left, point, right)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([
                np.array_equal(self.left, other.left),
                np.array_equal(self.point, other.point),
                np.array_equal(self.right, other.right),
                (self.between == other.between),
                isclose(self.offset, other.offset),
            ])
        return NotImplemented


def less_or_close(a, b, *args, **kwargs):
    # Use isclose for handling effective equivalence
    return a < b or isclose(a, b, *args, **kwargs)


def neighbor_window(seq, index, count=1):
    if len(seq) < (count + 2):
        raise ValueError("seq must have at least 3 elements to have neighbors")
    if index < 1 or index > (len(seq) - (count + 1)):
        raise IndexError(f"Index must fall between 1 and len(seq) - 2 to have neighbors: (index={index}, seq={seq})")
    return seq[index - 1:index + count + 1]


def wrap_to(seq, number):
    # Undo any existing wrap
    wrap_length = wrap_len(seq)
    if wrap_length > 0:
        seq = seq[:-wrap_length]

    # Wrap to the given number
    return np.append(seq, seq[:number], axis=0)


def wrap_len(seq):
    if len(seq) <= 1:
        return 0
    end_index = len(seq) - 1
    first_match_index = get_point_index_by_value(seq, seq[-1])
    if first_match_index != end_index:
        return first_match_index + 1
    return 0


def modified_point_list(seq):
    if len(seq) < 3:
        raise ValueError("seq must have at least 3 elements to have neighbors")
    if not np.array_equal(seq[0], seq[-1]):
        seq = np.append(seq, [seq[0]], axis=0)
    return_seq = []
    for pnt in tuple(seq) + (seq[1],):
        try:
            if len(pnt) == 2:
                return_seq.append(np.asarray(pnt))
                continue
        except TypeError:
            raise ValueError("each element in seq must have len(2)")
    return return_seq


def point_window_iter(seq, window_count=1):
    # Iterates over groups of three points, where the input seq
    # has first and last the same, then add a final group with the
    # first/last element in the middle
    if np.array_equal(seq[0], seq[-1]):
        elem_wrapped_seq = tuple(seq) + (seq[1],)
    else:
        elem_wrapped_seq = seq
    for i in range(1, len(elem_wrapped_seq) - 1):
        yield neighbor_window(elem_wrapped_seq, i, count=window_count)


def within_tolerance(value, within, float_tol=1e-9):
    if (within < 0):
        raise ValueError('Argument "within" cannot be negative')
    abs_value = abs(value)
    return less_or_close(abs_value, within, rel_tol=float_tol)


def midpoint_projection_offset(pnt1, pnt2, pnt3):
    outer_vec = pnt3 - pnt1
    norm_outer = np.linalg.norm(outer_vec)
    return abs(np.cross(outer_vec, pnt1 - pnt2) / norm_outer)


def between_neighbors(pnt1, pnt2, pnt3):
    """Midpoint projected onto neighboring points line is contained in segment"""
    # Make sure the projection of the midpoint lies between the outer points
    outer_vec = pnt3 - pnt1
    norm_outer = np.linalg.norm(outer_vec)
    scalar_proj = np.dot(pnt2 - pnt1, outer_vec / norm_outer)
    return (
        less_or_close(0, scalar_proj) and less_or_close(scalar_proj, norm_outer)
    )


def points_inline(pnt1, pnt2, pnt3, tolerance, float_tol=1e-9):
    """Check if the middle point lies on the line between 1 and 2 withing tolerance"""
    mid_offset = midpoint_projection_offset(pnt1, pnt2, pnt3)

    # First check point is inline within tolerence
    is_inline = within_tolerance(mid_offset, tolerance, float_tol)

    # Make sure the projection of the midpoint lies between the outer points
    is_between = between_neighbors(pnt1, pnt2, pnt3)

    return is_inline and is_between


def get_radians(pnt1, pnt2, pnt3):
    v1 = pnt1 - pnt2
    v2 = pnt3 - pnt2
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def orthogonal(pnt1, pnt2, pnt3, tolerance):
    rad = get_radians(pnt1, pnt2, pnt3)
    return within_tolerance(rad - (np.pi / 2), tolerance)


def same_side(pnt1, line_start, line_end, pnt2):
    to_base = get_radians(pnt1, line_start, line_end)
    to_pnt2 = get_radians(pnt1, line_start, pnt2)
    return to_base > to_pnt2


def point_data_list(point_seq):
    for i in range(1, len(point_seq) - 1):
        p1, p2, p3 = neighbor_window(point_seq, i)
        yield PointData(p1, p2, p3)


def remove_insignificant(point_iter, data_iter, tolerance):
    data_seq = list(data_iter)
    sig_points = list(point_iter)
    while True:
        rem_values = [x.offset for x in data_seq if x.between and less_or_close(x.offset, tolerance)]
        if rem_values:
            next_rmv = min(rem_values)
            for index, data in enumerate(data_seq):
                if data.between and isclose(data.offset, next_rmv):
                    break
            # Remove then recalculate neighbors
            del sig_points[index + 1]
            del data_seq[index]
            if index == 0:
                # Replace last point with new following point
                sig_points[-1] = sig_points[1]
            if index == len(data_seq):
                sig_points[0] = sig_points[index]
            if index > 0:
                data_seq[index - 1] = PointData(*neighbor_window(sig_points, index))
            if index < len(data_seq):
                data_seq[index] = PointData(*neighbor_window(sig_points, index + 1))
            if index == len(data_seq):
                data_seq[index - 1] = PointData(*neighbor_window(sig_points, index))
        else:
            break
    return sig_points


def significant_points(points, tolerance):
    point_seq = modified_point_list(points)
    data_seq = point_data_list(point_seq)
    return remove_insignificant(point_seq, data_seq, tolerance)


def has_box(points, tolerance, angle_tolerance, min_len=10, max_len=80):
    sig_points = significant_points(points, tolerance)

    # Under 5 and the box is not possible
    if len(sig_points) < 5:
        return []

    for i in range(1, len(sig_points) - 2):
        p1, p2, p3, p4 = neighbor_window(sig_points, i, count=2)

        mid_dist = distance(p2, p3)
        if (orthogonal(p1, p2, p3, angle_tolerance) and
                orthogonal(p2, p3, p4, angle_tolerance) and
                same_side(p1, p2, p3, p4) and
                less_or_close(mid_dist, max_len) and
                less_or_close(min_len, mid_dist)):
            return [p1, p2, p3, p4]
    return []


def distance(pnt1, pnt2):
    return np.linalg.norm(np.asarray(pnt2) - np.asarray(pnt1))


def centroid(points):
    arr = np.asarray(points)
    if np.array_equal(arr[0], arr[-1]):
        arr = arr[:-1]
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return np.asarray((sum_x / length, sum_y / length))


def nearest_distances(points, num_nearest=1):
    if num_nearest < 1:
        ValueError("num_nearest must be at least 1")
    if len(points) <= num_nearest:
        ValueError("num_nearest cannot be larges than len(points) - 1")

    arr = np.array(points)
    tree = scipy.spatial.KDTree(arr)
    res = tree.query(tree.data, num_nearest + 1)
    # Return {
    #   [p_x, p_y].tobytes() : [dist_first_nearest, dist_sec_nearest, ..., dist_nth_nearest]
    # }  tobytes used as bytestring is hashable
    return {
        point.astype(np.float).tobytes(): dist[1:]
        for point, dist in zip(tree.data, res[0])
    }


def split_list(original, split_indexes):
    if not split_indexes:
        split_indexes = [0]
    if split_indexes[0] != 0:
        split_indexes.insert(0, 0)
    split_indexes.append(len(original))
    for i in range(1, len(split_indexes)):
        s, e = neighbor_window(split_indexes, i, 0)
        yield original[s:e]


def get_point_index_by_value(search_list, search_elem):
    if (len(np.shape(search_list)) - 1 != len(np.shape(search_elem))):
        raise ValueError("Search element not correct dimensions for search list")
    # https://stackoverflow.com/a/18927811
    search = np.asarray(search_list)
    elem = np.asarray(search_elem)
    match_arr = search == elem
    if len(np.shape(search)) > 1:
        match_arr = np.all(match_arr, axis=1)
    return np.where(match_arr)[0][0]


def get_top_point(points):
    arr = np.asarray(points)
    return arr[np.lexsort((arr[:,0], arr[:,1]))][::-1][0]


def remove_array_wrap(points):
    arr_last_index = len(points) - 1
    first_index = get_point_index_by_value(points, points[arr_last_index])
    if first_index != arr_last_index:
        arr_start = points[:first_index + 1]
        arr_end = points[-(first_index + 1):]
        if np.allclose(arr_start, arr_end):
            return points[:-(first_index + 1)]
    return points


def mid_line_rotation(left_point, right_point):
    horiz_point = left_point[:] + [1, 0]
    raw_angle = get_radians(horiz_point, left_point, right_point)
    return raw_angle % (np.pi / 2)


def is_rectangle(points, line_tolerance, angle_tolerance):
    sig = significant_points(points, line_tolerance)
    arr = remove_array_wrap(sig)
    if len(arr) == 4:
        np.append(arr, arr[0])
        for (p1, p2, p3) in point_window_iter(wrap_to(sig, 2)):
            if not orthogonal(p1, p2, p3, angle_tolerance):
                return []
        return wrap_to(arr, 1)
    return []


def area(points):
    arr = np.asarray(points)
    return (1 / 2) * abs(
        np.dot(arr[:,0], np.roll(arr[:,1], 1)) -
        np.dot(np.roll(arr[:,0], 1), arr[:,1])
    )
