import argparse
from itertools import islice
from math import isclose
import sys

import numpy as np
import shapefile


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


def within_tolerance(actual, target, tolerance):
    diff = abs(target - actual)
    # Use isclose for handling effective equivalence
    return diff < tolerance or isclose(diff, tolerance)


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
