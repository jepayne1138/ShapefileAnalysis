import argparse
import logging
import sys

import numpy as np
import shapefile

import shapeanalysis.database as database
from shapeanalysis.process_data import (
    split_list,
    significant_points,
    has_box,
    centroid,
    nearest_distances,
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
    point_list = [(points, rec) for rec in shapeRecs for points in split_list(rec.shape.points, rec.shape.parts)][:100]
    total = len(point_list)
    matches = []
    for i, (points, rec) in enumerate(point_list):
        logger.debug(f'Processing: {i}/{total}')
        # Check if the shape matches the search criteria
        sig_points = significant_points(points, args.inline_tolerance)
        if has_box(points, args.inline_tolerance, args.angle_tolerance):
            matches.append((rec, sig_points))

    rec_data = []
    centroid_points = []
    for match_rec, points in matches:
        centroid_point = centroid(points)
        centroid_points.append(centroid_point)
        rec_data.append([match_rec, centroid_point])

    distances = nearest_distances(centroid_points, 2)

    main = []
    for i in range(len(rec_data)):
        pid = rec_data[i][0].record[0]
        near_dists = distances[rec_data[i][1].astype(np.float).tobytes()]
        main.append((pid, near_dists[0], near_dists[1]))

    # for _, c_point in rec_data:
    #     rec_data
    # for i in range(len(rec_data)):
    #     rec_data[i] += list(distances[rec_data[i][1].astype(np.float).tobytes()])

    # import csv
    # with open('output.csv', 'w', newline='') as outputfile:
    #     csv_writer = csv.writer(outputfile)
    #     for x in rec_data:
    #         csv_writer.writerow([x[0].record[0]] + [str(tuple(x[1]))] + x[2:])

    with database.connection(args.output) as conn:
        database.create_database(conn)
        database.insert_main(conn, main)


if __name__ == '__main__':
    main()
