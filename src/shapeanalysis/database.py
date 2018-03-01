import sqlite3


def create_database(conn):
    c = conn.cursor()

    # Table "main"
    c.execute("""drop table if exists main""")
    c.execute("""create table main (pid, nearest1, nearest2, numpoints)""")

    # Table "rectangle"
    c.execute("""drop table if exists rectangle""")
    c.execute("""create table rectangle (pid, side1, angle12, side2, angle23, side3, angle34, side4, angle41, minratio, maxratio, area)""")

    # Table "boxlike"
    c.execute("""drop table if exists boxlike""")
    c.execute("""create table boxlike (pid, hangle, left, langle, mid, rangle, right)""")


def insert_main(conn, data):
    c = conn.cursor()

    c.executemany(
        """
        insert into main
        (pid, nearest1, nearest2, numpoints)
        values (?, ?, ?, ?)
        """,
        data
    )


def insert_boxlike(conn, data):
    c = conn.cursor()

    c.executemany(
        """
        insert into boxlike
        (pid, hangle, left, langle, mid, rangle, right)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        data
    )


def insert_rectangle(conn, data):
    c = conn.cursor()

    c.executemany(
        """
        insert into rectangle
        (pid, side1, angle12, side2, angle23, side3, angle34, side4, angle41, minratio, maxratio, area)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        data
    )


def connection(filename):
    return sqlite3.connect(filename)
