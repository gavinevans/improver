# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
""" Provides support utilities for checking cubes."""

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np


def check_cube_datatypes(cube, fix=False):
    """Check whether the data within a cube is float64.
    Coordinates are also checked to ensure that they have the desired datatype.
    The cube can be modified in place, if the fix keyword is
    specified to be True.
    Prior to converting datatypes, the following unit conversions occur to
    ensure that any datatype conversion, and any associated rounding, uses the
    most appropriate units:
        * time converted to "seconds since 1970-01-01 00:00:00"
        * forecast_reference_time converted to
          "seconds since 1970-01-01 00:00:00"
        * forecast_period converted to seconds

    Args:
        cube (iris.cube.Cube):
            The input cube that will be checked for whether it is float64.

    Keyword Args:
        fix (bool):
            If fix is True, then the cube is amended to not include erroneous
            datatypes, otherwise, an error will be raised.

    Raises:
        TypeError : Raised if the cube is not of the datatype specified, and
            'fix' is not set to True.

    """
    if cube.dtype == np.float64:
        if fix:
            cube.data = cube.data.astype(np.float32)
        else:
            msg = ("The {} cube is expected to be of float32 datatype. "
                   "The cube provided was of float64 datatype.".format(
                       cube.name()))
            raise TypeError(msg)
    for coord in cube.coords():
        if coord.name() in ["time", "forecast_reference_time"]:
            coord.convert_units("seconds since 1970-01-01 00:00:00")
            check_coord_datatypes(coord, np.int64, fix=fix, rounding=True)
        elif coord.name() in ["forecast_period", "realization", "spot_index",
                              "neighbour_selection_method", "grid_attributes",
                              "wmo_id", "model_id"]:
            if coord.name() in ["forecast_period"]:
                coord.convert_units("seconds")
            check_coord_datatypes(coord, np.int32, fix=fix)
        elif coord.name() in ["neighbour_selection_method_name",
                              "grid_attributes_key", "model_configuration"]:
            check_coord_datatypes(coord, np.unicode_, fix=fix)
        else:
            check_coord_datatypes(coord, np.float32, fix=fix)


def check_coord_datatypes(coord, datatype, fix=False, rounding=False):
    """Check that the coordinate is of the datatype specified. The coordinate
    will be modified in place, if the fix keyword is set to True. This
    function is intended to be called from check_cube_datatypes, so that the
    appropriate unit conversions have already occurred.

    Args:
        coord (iris.coord.DimCoord, iris.coord.AuxCoord):
            The coordinate that will be checked for its datatype.
        datatype (type):
            The datatype that is expected for the coordinate.
            For example, np.int32, np.float32, np.float64, etc.

    Keyword Args:
       fix (bool):
           If fix is True, then the cube is amended to not include datatype
           specified, otherwise, an error will be raised.
       rounding (bool):
           If fix is True, then the rounding keyword can also be enabled,
           that will round the coordinate points and bounds prior to changing
           the datatype.

    Raises:
        TypeError: The coordinate points are not of the expected datatype.
        TypeError: The coordinate bounds are not of the expected datatype.
    """
    # Check that the units of the coordinate are appropriate for
    # datatype conversion.
    _check_coord_units_for_datatype_conversion

    # Check to ensure that the dtype of the coordinate points is
    # mismatched with the specifed datatype, and ensure that the dtype
    # of the coordinate points is not a subtype of the specified datatype.
    # The np.issubdtype check is primarily used for compared unicode dtypes
    # with np.unicode_.
    if (coord.points.dtype != datatype and
            not np.issubdtype(coord.points.dtype, datatype)):
        if fix:
            if rounding:
                coord.points = np.around(coord.points)
            coord.points = coord.points.astype(datatype)
        else:
            msg = ("The {} coordinate points {} are expected "
                   "to be of {} datatype. "
                   "The coordinate points provided were "
                   "of {} datatype.").format(
                       coord.name(), coord.points, datatype, coord.dtype)
            raise TypeError(msg)
    if (hasattr(coord, "bounds") and coord.bounds is not None and
            coord.bounds.dtype != datatype and
            not np.issubdtype(coord.bounds.dtype, datatype)):
        if fix:
            if rounding:
                coord.bounds = np.around(coord.bounds)
            coord.bounds = coord.bounds.astype(datatype)
        else:
            msg = ("The {} coordinate bounds {} are expected "
                   "to be of {} datatype. "
                   "The coordinate bounds provided were "
                   "of {} datatype.").format(
                       coord.name(), coord.bounds, datatype, coord.dtype)
            raise TypeError(msg)


def _check_coord_units_for_datatype_conversion(coord):
    """Check that the units of the coordinate that is supplied is appropriate
       for datatype conversion.

    Args:
        coord (iris.coord.DimCoord, iris.coord.AuxCoord):
            The coordinate that will be checked for its datatype.

    Raises:
        ValueError: The coordinate units are not appropriate for datatype
            conversion.
        ValueError: The coordinate units are not appropriate for datatype
            conversion.
    """
    if coord.name() in ["time", "forecast_reference_time"]:
        if coord.units != "seconds since 1970-01-01 00:00:00":
            msg = ("For datatype conversion, the {} coordinate "
                   "must have units of seconds since 1970-01-01 00:00:00. "
                   "The units supplied were {}".format(
                        coord.name(), coord.units))
            raise ValueError(msg)
    if coord.name() in ["forecast_period"]:
        if coord.units != "seconds":
            msg = ("For datatype conversion, the {} coordinate "
                   "must have units of seconds."
                   "The units supplied were {}".format(
                        coord.name(), coord.units))
            raise ValueError(msg)


def check_for_x_and_y_axes(cube, require_dim_coords=False):
    """
    Check whether the cube has an x and y axis, otherwise raise an error.

    Args:
        cube (Iris.cube.Cube):
            Cube to be checked for x and y axes.
        require_dim_coords (bool):
            If true the x and y coordinates must be dimension coordinates.

    Raises:
        ValueError : Raise an error if non-uniform increments exist between
                      grid points.
    """
    for axis in ["x", "y"]:
        if require_dim_coords:
            coord = cube.coords(axis=axis, dim_coords=True)
        else:
            coord = cube.coords(axis=axis)

        if coord:
            pass
        else:
            msg = ("The cube does not contain the expected {}"
                   "coordinates.".format(axis))
            raise ValueError(msg)


def check_cube_coordinates(cube, new_cube, exception_coordinates=None):
    """Find and promote to dimension coordinates any scalar coordinates in
    new_cube that were originally dimension coordinates in the progenitor
    cube. If coordinate is in new_cube that is not in the old cube, keep
    coordinate in its current position.

    Args:
        cube (iris.cube.Cube):
            The input cube that will be checked to identify the preferred
            coordinate order for the output cube.
        new_cube (iris.cube.Cube):
            The cube that must be checked and adjusted using the coordinate
            order from the original cube.
        exception_coordinates (List of strings or None):
            The names of the coordinates that are permitted to be within the
            new_cube but are not available within the original cube.

    Returns:
        new_cube (iris.cube.Cube):
            Modified cube with relevant scalar coordinates promoted to
            dimension coordinates with the dimension coordinates re-ordered,
            as best as can be done based on the original cube.

    Raises:
        CoordinateNotFoundError : Raised if the final dimension
            coordinates of the returned cube do not match the input cube.
        CoordinateNotFoundError : If a coordinate is within in the permitted
            exceptions but is not in the new_cube.
    """
    if exception_coordinates is None:
        exception_coordinates = []

    # Promote available and relevant scalar coordinates
    cube_dim_names = [coord.name() for coord in cube.dim_coords]
    for coord in new_cube.aux_coords[::-1]:
        if coord.name() in cube_dim_names:
            new_cube = iris.util.new_axis(new_cube, coord)
    new_cube_dim_names = [coord.name() for coord in new_cube.dim_coords]
    # If we have the wrong number of dimensions then raise an error.
    if (len(cube.dim_coords)+len(exception_coordinates) !=
            len(new_cube.dim_coords)):

        msg = ('The number of dimension coordinates within the new cube '
               'do not match the number of dimension coordinates within the '
               'original cube plus the number of exception coordinates. '
               '\n input cube dimensions {}, new cube dimensions {}'.format(
                   cube_dim_names, new_cube_dim_names))
        raise CoordinateNotFoundError(msg)

    # Ensure dimension order matches
    new_cube_dimension_order = {coord.name(): new_cube.coord_dims(
        coord.name())[0] for coord in new_cube.dim_coords}
    correct_order = []
    new_cube_only_dims = []
    for coord_name in cube_dim_names:
        correct_order.append(new_cube_dimension_order[coord_name])
    for coord_name in exception_coordinates:
        try:
            new_coord_dim = new_cube.coord_dims(coord_name)[0]
            new_cube_only_dims.append(new_coord_dim)
        except CoordinateNotFoundError:
            msg = ("All permitted exception_coordinates must be on the"
                   " new_cube. In this case, coordinate {0} within the list "
                   "of permitted exception_coordinates ({1}) is not available"
                   " on the new_cube.").format(
                        coord_name, exception_coordinates)
            raise CoordinateNotFoundError(msg)

    correct_order = np.array(correct_order)
    for dim in new_cube_only_dims:
        correct_order = np.insert(correct_order, dim, dim)

    new_cube.transpose(correct_order)

    return new_cube


def find_dimension_coordinate_mismatch(
        first_cube, second_cube, two_way_mismatch=True):
    """Determine if there is a mismatch between the dimension coordinates in
    two cubes.

    Args:
        first_cube (Iris.cube.Cube):
            First cube to compare.
        second_cube (Iris.cube.Cube):
            Second cube to compare.
        two_way_mismatch (Logical):
            If True, a two way mismatch is calculated e.g.
                second_cube - first_cube AND
                first_cube - second_cube
            If False, a one way mismatch is calculated e.g.
                second_cube - first_cube

    Returns:
        result (List):
            List of the dimension coordinates that are only present in
            one out of the two cubes.

    """
    first_dim_names = [coord.name() for coord in first_cube.dim_coords]
    second_dim_names = [coord.name() for coord in second_cube.dim_coords]
    if two_way_mismatch:
        mismatch = (list(set(second_dim_names) - set(first_dim_names)) +
                    list(set(first_dim_names) - set(second_dim_names)))
    else:
        mismatch = list(set(second_dim_names) - set(first_dim_names))
    return mismatch


def spatial_coords_match(first_cube, second_cube):
    """
    Determine if the x and y coords in the two cubes are the same.

    Args:
        first_cube (Iris.cube.Cube):
            First cube to compare.
        second_cube (Iris.cube.Cube):
            Second cube to compare.

    Returns:
        result (bool):
            True if the x and y coords are the exactly the same to the
            precision of the floating-point values (this should be true for
            any cubes derived using cube.regrid()), otherwise False.
    """
    return (first_cube.coord(axis='x') == second_cube.coord(axis='x') and
            first_cube.coord(axis='y') == second_cube.coord(axis='y'))


def find_percentile_coordinate(cube):
    """Find percentile coord in cube.

    Args:
        cube (iris.cube.Cube):
            Cube contain one or more percentiles.
    Returns:
        perc_coord(iris.coords.Coord) :
            Percentile coordinate.
    Raises:
        TypeError : If cube is not of type iris.cube.Cube.
        CoordinateNotFoundError : If no percentile coordinate is found in cube.
        ValueError : If there is more than one percentile coords in the cube.
    """
    if not isinstance(cube, iris.cube.Cube):
        msg = ('Expecting data to be an instance of '
               'iris.cube.Cube but is {0}.'.format(type(cube)))
        raise TypeError(msg)
    standard_name = cube.name()
    perc_coord = None
    perc_found = 0
    for coord in cube.coords():
        if coord.name().find('percentile') >= 0:
            perc_found += 1
            perc_coord = coord
    if perc_found != 1:
        if perc_found == 0:
            msg = ('No percentile coord found on {0:s} data'.format(
                standard_name))
            raise CoordinateNotFoundError(msg)
        else:
            msg = ('Too many percentile coords found on {0:s} data'.format(
                standard_name))
            raise ValueError(msg)
    return perc_coord
