import argparse
from itertools import islice
from math import isclose
import sys

import numpy as np
import shapefile


def neighbor_window(seq, index):
    if len(seq) < 3:
        raise ValueError("seq must have at least 3 elements to have neighbors")
    if index < 1 or index > (len(seq) - 2):
        raise IndexError("Index must fall between 1 and len(seq) - 2 to have neighbors")
    return seq[index - 1:index + 2]


def window(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def point_window_iter(seq):
    # Iterates over groups of three points, where the input seq
    # has first and last the same, then add a final group with the
    # first/last element in the middle
    first = None
    for i, item in enumerate(window(seq, 3)):
        if i == 0:
            first = item
        yield item
    yield item[1:] + (first[1],)


def within_tolerance(value, within, float_tol=1e-9):
    if (within < 0):
        raise ValueError('Argument "within" cannot be negative')
    abs_value = abs(value)
    # Use isclose for handling effective equivalence
    return abs_value < within or isclose(abs_value, within, rel_tol=float_tol)


def points_inline(pnt1, pnt2, pnt3, tolerance, float_tol=1e-9):
    """Check if the middle point lies on the line between 1 and 2 withing tolerance"""
    outer_vec = pnt3 - pnt1
    norm_outer = np.linalg.norm(outer_vec)
    min_offset = np.cross(outer_vec, pnt1 - pnt2) / norm_outer

    # First check point is inline within tolerence
    is_inline = within_tolerance(min_offset, tolerance, float_tol)

    # Make sure the projection of the midpoint lies between the outer points

    scalar_proj = np.dot(pnt2 - pnt1, outer_vec / norm_outer)
    is_between = (
        (scalar_proj > 0 or isclose(scalar_proj, 0)) and
        (scalar_proj < norm_outer or isclose(scalar_proj, norm_outer))
    )
    return is_inline and is_between


def get_radians(pnt1, pnt2, pnt3):
    v1 = pnt1 - pnt2
    v2 = pnt3 - pnt2
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Analyzes a tax parcel shapefile')
    parser.add_argument('shapefile', type=str, help='Path to the .shp file')
    return parser.parse_args(sys.argv[1:])


def main():
    args = parse_arguments(sys.argv[1:])

    # sf = shapefile.Reader(args.shapefile)
    # shapeRecs = sf.shapeRecords()


if __name__ == '__main__':
    main()
