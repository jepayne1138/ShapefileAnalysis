import argparse
from math import isclose
import sys

import numpy as np
import shapefile


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


def neighbor_window(seq, index):
    if len(seq) < 3:
        raise ValueError("seq must have at least 3 elements to have neighbors")
    if index < 1 or index > (len(seq) - 2):
        raise IndexError("Index must fall between 1 and len(seq) - 2 to have neighbors")
    return seq[index - 1:index + 2]


def modified_point_list(seq):
    if len(seq) < 3:
        raise ValueError("seq must have at least 3 elements to have neighbors")
    if seq[0] != seq[-1]:
        raise ValueError("First and last element must match")
    return_seq = []
    for pnt in tuple(seq) + (seq[1],):
        try:
            if len(pnt) == 2:
                return_seq.append(np.asarray(pnt))
                continue
        except TypeError:
            raise ValueError("each element in seq must have len(2)")
    return return_seq


def point_window_iter(seq):
    # Iterates over groups of three points, where the input seq
    # has first and last the same, then add a final group with the
    # first/last element in the middle
    elem_wrapped_seq = seq + (seq[1],)
    for i in range(1, len(elem_wrapped_seq) - 1):
        yield neighbor_window(elem_wrapped_seq, i)


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


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Analyzes a tax parcel shapefile')
    parser.add_argument('shapefile', type=str, help='Path to the .shp file')
    return parser.parse_args(sys.argv[1:])


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


def main():
    args = parse_arguments(sys.argv[1:])

    # sf = shapefile.Reader(args.shapefile)
    # shapeRecs = sf.shapeRecords()


if __name__ == '__main__':
    main()
