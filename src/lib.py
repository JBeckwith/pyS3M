"""
    picasso/lib
    ~~~~~~~~~~~~~~~~~~~~

    Handy functions and classes

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

import numba
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from numpy.lib.recfunctions import append_fields, drop_fields


def append_to_rec(rec_array, data, name):
    """Appends a new column to the existing np.recarray.

    Parameters
    ----------
    rec_array : np.rec.array
        Recarray to which the new column is appended.
    data : np.1darray
        Data to be appended.
    name : str
        Name of the new column.

    Returns
    -------
    rec_array : np.rec.array
        Recarray with the new column.
    """

    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
    rec_array = append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array



def is_loc_at(x, y, locs, r):
    """Checks if localizations are at position (x, y) within radius r.

    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    is_picked : np.ndarray
        Boolean array indicating if localization is at position.
    """

    dx = locs.xc - x
    dy = locs.yc - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked


def locs_at(x, y, locs, r):
    """Returns localizations at position (x, y) within radius r.

    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    picked_locs : np.rec.array
        Localizations at position.
    """

    is_picked = is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    return picked_locs


@numba.jit(nopython=True)
def check_if_in_polygon(x, y, X, Y):
    """Checks if points (x, y) are in polygon defined by corners (X, Y).
    Uses the ray casting algorithm, see check_if_in_rectangle for
    details.

    Parameters
    ----------
    x : numpy.1darray
        x-coordinates of points.
    y : numpy.1darray
        y-coordinates of points.
    X : numpy.1darray
        x-coordinates of polygon corners.
    Y : numpy.1darray
        y-coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : numpy.ndarray
        Boolean array indicating if point is in polygon.
    """

    n_locs = len(x)
    n_polygon = len(X)
    is_in_polygon = np.zeros(n_locs, dtype=np.bool_)

    for i in range(n_locs):
        count = 0
        for j in range(n_polygon):
            j_next = (j + 1) % n_polygon
            if ((Y[j] > y[i]) != (Y[j_next] > y[i])) and (
                x[i] < X[j] + (X[j_next] - X[j]) * (y[i] - Y[j]) / (Y[j_next] - Y[j])
            ):
                count += 1
        if count % 2 == 1:
            is_in_polygon[i] = True

    return is_in_polygon


def locs_in_polygon(locs, X, Y):
    """Returns localizations in polygon defined by corners (X, Y).

    Parameters
    ----------
    locs : numpy.recarray
        Localizations.
    X : list
        x-coordinates of polygon corners.
    Y : list
        y-coordinates of polygon corners.

    Returns
    -------
    picked_locs : numpy.recarray
        Localizations in polygon.
    """

    is_in_polygon = check_if_in_polygon(locs.xc, locs.yc, np.array(X), np.array(Y))
    return locs[is_in_polygon]


@numba.jit(nopython=True)
def check_if_in_rectangle(x, y, X, Y):
    """
    Checks if locs with coordinates (x, y) are in rectangle with corners (X, Y)
    by counting the number of rectangle sides which are hit by a ray
    originating from each loc to the right. If the number of hit rectangle
    sides is odd, then the loc is in the rectangle

    Parameters
    ----------
    x : numpy.1darray
        x-coordinates of points.
    y : numpy.1darray
        y-coordinates of points.
    X : numpy.1darray
        x-coordinates of polygon corners.
    Y : numpy.1darray
        y-coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : numpy.ndarray
        Boolean array indicating if point is in polygon.
    """

    n_locs = len(x)
    ray_hits_rectangle_side = np.zeros((n_locs, 4))
    for i in range(4):
        # get two y coordinates of corner points forming one rectangle side
        y_corner_1 = Y[i]
        # take the first if we're at the last side:
        y_corner_2 = Y[0] if i == 3 else Y[i + 1]
        y_corners_min = min(y_corner_1, y_corner_2)
        y_corners_max = max(y_corner_1, y_corner_2)
        for j in range(n_locs):
            y_loc = y[j]
            # only if loc is on level of rectangle side, its ray can hit:
            if y_corners_min <= y_loc <= y_corners_max:
                x_corner_1 = X[i]
                # take the first if we're at the last side:
                x_corner_2 = X[0] if i == 3 else X[i + 1]
                # calculate intersection point of ray and side:
                m_inv = (x_corner_2 - x_corner_1) / (y_corner_2 - y_corner_1)
                x_intersect = m_inv * (y_loc - y_corner_1) + x_corner_1
                x_loc = x[j]
                if x_intersect >= x_loc:
                    # ray hits rectangle side on the right side
                    ray_hits_rectangle_side[j, i] = 1
    n_sides_hit = np.sum(ray_hits_rectangle_side, axis=1)
    is_in_rectangle = n_sides_hit % 2 == 1
    return is_in_rectangle


def locs_in_rectangle(locs, X, Y):
    """Returns localizations in rectangle defined by corners (X, Y).

    Parameters
    ----------
    locs : numpy.recarray
        Localizations list.
    X : list
        x-coordinates of rectangle corners.
    Y : list
        y-coordinates of rectangle corners.

    Returns
    -------
    picked_locs : numpy.recarray
        Localizations in rectangle.
    """

    is_in_rectangle = check_if_in_rectangle(locs.xc, locs.yc, np.array(X), np.array(Y))
    picked_locs = locs[is_in_rectangle]
    return picked_locs


def remove_from_rec(rec_array, name):
    """Removes a column from the existing np.recarray.

    Parameters
    ----------
    rec_array : np.rec.array
        Recarray from which the column is removed.
    name : str
        Name of the column to be removed.

    Returns
    -------
    rec_array : np.rec.array
        Recarray without the column.
    """

    rec_array = drop_fields(rec_array, name, usemask=False, asrecarray=True)
    return rec_array


def get_pick_polygon_corners(pick):
    """Returns X and Y coordinates of a pick polygon.

    Returns None, None if the pick is not a closed polygon."""

    if len(pick) < 3 or pick[0] != pick[-1]:
        return None, None
    else:
        X = [_[0] for _ in pick]
        Y = [_[1] for _ in pick]
        return X, Y


def get_pick_rectangle_corners(start_x, start_y, end_x, end_y, width):
    """Finds the positions of corners of a rectangular pick.
    Rectangular pick is defined by:
        [(start_x, start_y), (end_x, end_y)]
    and its width. (all values in camera pixels)

    Returns
    -------
    corners : tuple
        Contains corners' x and y coordinates in two lists
    """

    if end_x == start_x:
        alpha = np.pi / 2
    else:
        alpha = np.arctan((end_y - start_y) / (end_x - start_x))
    dx = width * np.sin(alpha) / 2
    dy = width * np.cos(alpha) / 2
    x1 = float(start_x - dx)
    x2 = float(start_x + dx)
    x4 = float(end_x - dx)
    x3 = float(end_x + dx)
    y1 = float(start_y + dy)
    y2 = float(start_y - dy)
    y4 = float(end_y + dy)
    y3 = float(end_y - dy)
    corners = ([x1, x2, x3, x4], [y1, y2, y3, y4])
    return corners

