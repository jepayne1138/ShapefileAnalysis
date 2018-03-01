import argparse
import itertools
import logging
import sys

import numpy as np
import shapefile

from shapeanalysis import database
from shapeanalysis.process_data import (
    split_list,
    point_window_iter,
    distance,
    area,
    is_rectangle,
    centroid,
    nearest_distances,
    wrap_to,
    get_radians,
    mid_line_rotation,
    has_box,
    remove_array_wrap,
    significant_points,
)

# Global logger instance
logger = logging.getLogger()


def configure_logger():
    # Set up logger
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Analyzes a tax parcel shapefile')
    parser.add_argument('shapefile', type=str, help='Path to the .shp file')
    parser.add_argument('output', type=str, help='Path to the output file')
    parser.add_argument('-i', '--inline-tolerance', type=float, default=0.6, help='Tolerance for determining insignificant point (feet): default=0.6')
    parser.add_argument('-a', '--angle-tolerance', type=float, default=0.03, help='Tolerance for measuring angles (radians): default=0.03')
    return parser.parse_args(sys.argv[1:])


def main():
    args = parse_arguments(sys.argv[1:])
    configure_logger()

    # Read in the given shapefile
    logger.info('Reading file...')
    sf = shapefile.Reader(args.shapefile)
    shapeRecs = sf.shapeRecords()

    # Process the shape records
    logger.info('Processing...')
    # Get individual shape points separated, as each shape can include multiple parts

    logger.info('Getting separated shapes...')
    shape_list = [
        (points, rec)
        for rec in shapeRecs
        for points in split_list(rec.shape.points, rec.shape.parts)
    ]
    # [:1000]

    # total = len(point_list)
    # matches = []
    # for i, (points, rec) in enumerate(point_list):
    #     logger.debug(f'Processing: {i}/{total}')
    #     # Check if the shape matches the search criteria
    #     sig_points = significant_points(points, args.inline_tolerance)
    #     if has_box(points, args.inline_tolerance, args.angle_tolerance):
    #         matches.append((rec, sig_points))

    # Get center points
    logger.info('Getting nearest distances...')
    nearest_list = []
    centroid_points = [centroid(shape) for (shape, _) in shape_list]
    distances = nearest_distances(centroid_points, 2)
    for i in range(len(shape_list)):
        num_sig_points = len(remove_array_wrap(significant_points(shape_list[i][0], 0.6)))
        shapeRec = shape_list[i][1]
        pid = shapeRec.record[0]
        near_dists = distances[centroid_points[i].astype(np.float).tobytes()]
        nearest_list.append((pid, near_dists[0], near_dists[1], num_sig_points))

    # Get rectangles
    logger.info('Getting rectangle matches...')
    rectangle_list = []
    for i in range(len(shape_list)):
        shape_points = is_rectangle(shape_list[i][0], 0.6, 0.03)
        if len(shape_points) > 0:
            shapeRec = shape_list[i][1]
            pid = shapeRec.record[0]

            sides = [
                distance(*points)
                for points
                in point_window_iter(shape_points, 0)
            ]
            angles = [
                get_radians(*points)
                for points
                in point_window_iter(wrap_to(shape_points, 2), 1)
            ]
            side_angles = list(
                itertools.chain.from_iterable(zip(sides, angles))
            )

            # Calculate smallest and largest ratio
            ordered_sides = sorted(sides)
            min_ratio = ordered_sides[0] / ordered_sides[3]
            max_ratio = ordered_sides[1] / ordered_sides[2]

            shape_area = area(shape_points)
            rec_list_add = (
                [pid] +
                side_angles +
                [
                    min_ratio,
                    max_ratio,
                    shape_area,
                ]
            )
            rectangle_list.append(tuple(rec_list_add))

    # Get boxlike
    boxlike_list = []
    for i in range(len(shape_list)):
        boxlike_points = has_box(shape_list[i][0], 0.6, 0.03, min_len=10, max_len=80)
        if len(boxlike_points) > 0:
            shapeRec = shape_list[i][1]
            pid = shapeRec.record[0]
            p1, p2, p3, p4 = boxlike_points
            boxlike_list_add = [
                pid,
                mid_line_rotation(p2, p3),
                distance(p1, p2),
                get_radians(p1, p2, p3),
                distance(p2, p3),
                get_radians(p2, p3, p4),
                distance(p3, p4),
            ]
            boxlike_list.append(tuple(boxlike_list_add))

    with database.connection(args.output) as conn:
        logger.info('Rebuilding database...')
        database.create_database(conn)
        logger.info('Writing nearest points to database...')
        database.insert_main(conn, nearest_list)
        logger.info('Writing rectangles to database...')
        database.insert_rectangle(conn, rectangle_list)
        logger.info('Writing boxlike to database...')
        database.insert_boxlike(conn, boxlike_list)


if __name__ == '__main__':
    main()
