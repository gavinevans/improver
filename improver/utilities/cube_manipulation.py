# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
""" Provides support utilities for cube manipulation."""

import warnings

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import CoordinateNotFoundError

from improver import BasePlugin
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_checker import check_cube_coordinates


def collapsed(cube, *args, **kwargs):
    """Collapses the cube with given arguments.

    The cell methods of the output cube will match the cell methods
    from the input cube. Any cell methods generated by the iris
    collapsed method will not be retained.

    Args:
        cube (iris.cube.Cube):
            A Cube to be collapsed.

    Returns:
        iris.cube.Cube:
            A collapsed cube where the cell methods match the input cube.
    """
    original_methods = cube.cell_methods
    new_cube = cube.collapsed(*args, **kwargs)
    new_cube.cell_methods = original_methods
    return new_cube


def collapse_realizations(cube):
    """Collapses the realization coord of a cube and strips the coord from the cube.

    Args:
        cube (iris.cube.Cube):
            Input cube

    Returns:
        iris.cube.Cube:
            Cube with realization coord collapsed and removed.
    """
    returned_cube = collapsed(cube, "realization", iris.analysis.MEAN)
    returned_cube.remove_coord("realization")
    return returned_cube


def get_dim_coord_names(cube):
    """
    Returns an ordered list of dimension coordinate names on the cube

    Args:
        cube (iris.cube.Cube)

    Returns:
        list of str
    """
    return [coord.name() for coord in cube.coords(dim_coords=True)]


def equalise_cube_attributes(cubes, silent=None):
    """
    Function to remove attributes that do not match between all cubes in the
    list.  Cubes are modified in place.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to check the attributes and revise.
        silent (list or None):
            List of attributes to remove silently if unmatched.

    Warns:
        UserWarning:
            If an unmatched attribute is not in the "silent" list,
            a warning will be raised.

    NOTE 16/05/19: iris.experimental now has an equalise_attributes function,
    which removes any unmatched attributes without raising a warning.

    TODO replace this function with the iris version once it is promoted into
    the standard iris package.  At that time, the silent_attributes member of
    MergeCubes becomes obsolete and should be removed.
    """
    if silent is None:
        silent = []
    unmatched = compare_attributes(cubes)
    warning_msg = "Deleting unmatched attribute {}, value {}"
    if len(unmatched) > 0:
        for i, cube in enumerate(cubes):
            for attr in unmatched[i]:
                if attr not in silent:
                    warnings.warn(warning_msg.format(attr, cube.attributes[attr]))
                cube.attributes.pop(attr)


def strip_var_names(cubes):
    """
    Strips var_name from the cube and from all coordinates except where
    required to support probabilistic metadata.  Inputs are modified in place.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube)

    Returns:
        iris.cube.CubeList
    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    for cube in cubes:
        cube.var_name = None
        for coord in cube.coords():
            # retain var name required for threshold coordinate
            if coord.var_name != "threshold":
                coord.var_name = None
    return cubes


class MergeCubes(BasePlugin):
    """
    Class adding functionality to iris.merge_cubes()

    Accounts for differences in attributes, cell methods and bounds ranges to
    avoid merge failures and anonymous dimensions.
    """

    def __init__(self):
        """Initialise constants"""
        # List of attributes to remove silently if unmatched
        self.silent_attributes = ["history", "title", "mosg__grid_version"]

    @staticmethod
    def _equalise_cell_methods(cubelist):
        """
        Function to equalise cell methods that do not match.  Modifies cubes
        in place.

        Args:
            cubelist (iris.cube.CubeList):
                List of cubes to check the cell methods and revise.
        """
        cell_methods = cubelist[0].cell_methods
        for cube in cubelist[1:]:
            cell_methods = list(set(cell_methods) & set(cube.cell_methods))
        for cube in cubelist:
            cube.cell_methods = tuple(cell_methods)

    @staticmethod
    def _check_time_bounds_ranges(cube):
        """
        Check the bounds on any dimensional time coordinates after merging.
        For example, to check time and forecast period ranges for accumulations
        to avoid blending 1 hr with 3 hr accumulations.  If points on the
        coordinate are not compatible, raise an error.

        Args:
            cube (iris.cube.Cube):
                Merged cube
        """
        for name in ["time", "forecast_period"]:
            try:
                coord = cube.coord(name)
            except CoordinateNotFoundError:
                continue

            if coord.bounds is None:
                continue
            if len(coord.points) == 1:
                continue

            bounds_ranges = np.abs(np.diff(coord.bounds))
            reference_range = bounds_ranges[0]
            if not np.all(np.isclose(bounds_ranges, reference_range)):
                msg = (
                    "Cube with mismatching {} bounds ranges "
                    "cannot be blended".format(name)
                )
                raise ValueError(msg)

    def process(
        self, cubes_in, check_time_bounds_ranges=False, slice_over_realization=False
    ):
        """
        Function to merge cubes, accounting for differences in attributes,
        coordinates and cell methods.  Note that cubes with different sets
        of coordinates (as opposed to cubes with the same coordinates with
        different values) cannot be merged.

        If the input is a single Cube, this is returned unmodified.  A
        CubeList of length 1 is checked for mismatched time bounds before
        returning the single Cube (since a CubeList of this form may be the
        result of premature iris merging on load).

        Args:
            cubes_in (iris.cube.CubeList or iris.cube.Cube):
                Cubes to be merged.
            check_time_bounds_ranges (bool):
                Flag to check whether scalar time bounds ranges match.
                This is for when we are expecting to create a new "time" axis
                through merging for eg precipitation accumulations, where we
                want to make sure that the bounds match so that we are not eg
                combining 1 hour with 3 hour accumulations.
            slice_over_realization (bool):
                Options to combine cubes with different realization dimensions.
                These cannot always be concatenated directly as this can create a
                non-monotonic realization coordinate.

        Returns:
            iris.cube.Cube:
                Merged cube.
        """
        # if input is already a single cube, return unchanged
        if isinstance(cubes_in, iris.cube.Cube):
            return cubes_in

        if len(cubes_in) == 1:
            # iris merges cubelist into shortest list possible on load
            # - may already have collapsed across invalid time bounds
            if check_time_bounds_ranges:
                self._check_time_bounds_ranges(cubes_in[0])
            return cubes_in[0]

        # create copies of input cubes so as not to modify in place
        cubelist = iris.cube.CubeList([])
        for cube in cubes_in:
            if slice_over_realization:
                for real_slice in cube.slices_over("realization"):
                    cubelist.append(real_slice.copy())
            else:
                cubelist.append(cube.copy())

        # equalise cube attributes, cell methods and coordinate names
        equalise_cube_attributes(cubelist, silent=self.silent_attributes)
        strip_var_names(cubelist)
        self._equalise_cell_methods(cubelist)

        # merge resulting cubelist
        result = cubelist.merge_cube()

        # check time bounds if required
        if check_time_bounds_ranges:
            self._check_time_bounds_ranges(result)

        return result


def get_filtered_attributes(cube, attribute_filter=None):
    """
    Build dictionary of attributes that match the attribute_filter. If the
    attribute_filter is None, return all attributes.

    Args:
        cube (iris.cube.Cube):
            A cube from which attributes partially matching the
            attribute_filter will be returned.
        attribute_filter (str or None):
            A string to match, or partially match, against attributes to build
            a filtered attribute dictionary. If None, all attributes are
            returned.
    Returns:
        dict:
            A dictionary of attributes partially matching the attribute_filter
            that were found on the input cube.
    """
    attributes = cube.attributes
    if attribute_filter is not None:
        attributes = {k: v for (k, v) in attributes.items() if attribute_filter in k}
    return attributes


def compare_attributes(cubes, attribute_filter=None):
    """
    Function to compare attributes of cubes

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to compare (must be more than 1)
        attribute_filter (str or None):
            A string to filter which attributes are actually compared. If None
            all attributes are compared.
    Returns:
        list of dict:
            List of dictionaries of unmatching attributes
    Warns:
        Warning: If only a single cube is supplied
    """
    unmatching_attributes = []
    if len(cubes) == 1:
        msg = "Only a single cube so no differences will be found "
        warnings.warn(msg)
    else:
        reference_attributes = get_filtered_attributes(
            cubes[0], attribute_filter=attribute_filter
        )

        common_keys = reference_attributes.keys()
        for cube in cubes[1:]:
            cube_attributes = get_filtered_attributes(
                cube, attribute_filter=attribute_filter
            )
            common_keys = {
                key
                for key in cube_attributes.keys()
                if key in common_keys
                and np.all(cube_attributes[key] == reference_attributes[key])
            }

        for cube in cubes:
            cube_attributes = get_filtered_attributes(
                cube, attribute_filter=attribute_filter
            )
            unique_attributes = {
                key: value
                for (key, value) in cube_attributes.items()
                if key not in common_keys
            }
            unmatching_attributes.append(unique_attributes)

    return unmatching_attributes


def compare_coords(cubes):
    """
    Function to compare the coordinates of the cubes

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to compare (must be more than 1)

    Returns:
        list of dict:
            List of dictionaries of unmatching coordinates
            Number of dictionaries equals number of cubes
            unless cubes is a single cube in which case
            unmatching_coords returns an empty list.

    Warns:
        Warning: If only a single cube is supplied
    """
    unmatching_coords = []
    if len(cubes) == 1:
        msg = "Only a single cube so no differences will be found "
        warnings.warn(msg)
    else:
        common_coords = cubes[0].coords()
        for cube in cubes[1:]:
            cube_coords = cube.coords()
            common_coords = [
                coord
                for coord in common_coords
                if (
                    coord in cube_coords
                    and np.all(cube.coords(coord) == cubes[0].coords(coord))
                )
            ]

        for i, cube in enumerate(cubes):
            unmatching_coords.append(dict())
            for coord in cube.coords():
                if coord not in common_coords:
                    dim_coords = cube.dim_coords
                    if coord in dim_coords:
                        dim_val = dim_coords.index(coord)
                    else:
                        dim_val = None
                    aux_val = None
                    if dim_val is None and len(cube.coord_dims(coord)) > 0:
                        aux_val = cube.coord_dims(coord)[0]
                    unmatching_coords[i].update(
                        {
                            coord.name(): {
                                "data_dims": dim_val,
                                "aux_dims": aux_val,
                                "coord": coord,
                            }
                        }
                    )

    return unmatching_coords


def sort_coord_in_cube(cube, coord, descending=False):
    """Sort a cube based on the ordering within the chosen coordinate.
    Sorting can either be in ascending or descending order.
    This code is based upon https://gist.github.com/pelson/9763057.

    Args:
        cube (iris.cube.Cube):
            The input cube to be sorted.
        coord (str):
            Name of the coordinate to be sorted.
        descending (bool):
            If True it will be sorted in descending order.

    Returns:
        iris.cube.Cube:
            Cube where the chosen coordinate has been sorted into either
            ascending or descending order.

    Warns:
        Warning if the coordinate being processed is a circular coordinate.

    """
    coord_to_sort = cube.coord(coord)
    if isinstance(coord_to_sort, DimCoord):
        if coord_to_sort.circular:
            msg = (
                "The {} coordinate is circular. If the values in the "
                "coordinate span a boundary then the sorting may return "
                "an undesirable result.".format(coord_to_sort.name())
            )
            warnings.warn(msg)
    (dim,) = cube.coord_dims(coord_to_sort)
    index = [slice(None)] * cube.ndim
    index[dim] = np.argsort(coord_to_sort.points)
    if descending:
        index[dim] = index[dim][::-1]
    return cube[tuple(index)]


def enforce_coordinate_ordering(cube, coord_names, anchor_start=True):
    """
    Function to reorder dimensions within a cube.
    Note that the input cube is modified in place.

    Args:
        cube (iris.cube.Cube):
            Cube where the ordering will be enforced to match the order within
            the coord_names. This input cube will be modified as part of this
            function.
        coord_names (list or str):
            List of the names of the coordinates to order. If a string is
            passed in, only the single specified coordinate is reordered.
        anchor_start (bool):
            Define whether the specified coordinates should be moved to the
            start (True) or end (False) of the list of dimensions. If True, the
            coordinates are inserted as the first dimensions in the order in
            which they are provided. If False, the coordinates are moved to the
            end. For example, if the specified coordinate names are
            ["time", "realization"] then "realization" will be the last
            coordinate within the cube, whilst "time" will be the last but one.
    """
    if isinstance(coord_names, str):
        coord_names = [coord_names]

    # construct a list of dimensions on the cube to be reordered
    dim_coord_names = get_dim_coord_names(cube)
    coords_to_reorder = []
    for coord in coord_names:
        if coord == "threshold":
            try:
                coord = find_threshold_coordinate(cube).name()
            except CoordinateNotFoundError:
                continue
        if coord in dim_coord_names:
            coords_to_reorder.append(coord)

    original_coords = cube.coords(dim_coords=True)
    coord_dims = cube.coord_dims

    # construct list of reordered dimensions assuming start anchor
    new_dims = [coord_dims(coord)[0] for coord in coords_to_reorder]
    new_dims.extend(
        [
            coord_dims(coord)[0]
            for coord in original_coords
            if coord_dims(coord)[0] not in new_dims
        ]
    )

    # if anchor is end, reshuffle the list
    if not anchor_start:
        new_dims_end = new_dims[len(coords_to_reorder) :]
        new_dims_end.extend(new_dims[: len(coords_to_reorder)])
        new_dims = new_dims_end

    # transpose cube using new coordinate order
    if new_dims != sorted(new_dims):
        cube.transpose(new_dims)
    return


def clip_cube_data(cube, minimum_value, maximum_value):
    """Apply np.clip to data in a cube to ensure that the limits do not go
    beyond the provided minimum and maximum values.

    Args:
        cube (iris.cube.Cube):
            The cube that has been processed and contains data that is to be
            clipped.
        minimum_value (int or float):
            The minimum value, with data in the cube that falls below this
            threshold set to it.
        maximum_value (int or float):
            The maximum value, with data in the cube that falls above this
            threshold set to it.
    Returns:
        iris.cube.Cube:
            The processed cube with the data clipped to the limits of the
            original preprocessed cube.
    """
    original_attributes = cube.attributes
    original_methods = cube.cell_methods

    result = iris.cube.CubeList()
    for cube_slice in cube.slices([cube.coord(axis="y"), cube.coord(axis="x")]):
        cube_slice.data = np.clip(cube_slice.data, minimum_value, maximum_value)
        result.append(cube_slice)

    result = result.merge_cube()
    result.cell_methods = original_methods
    result.attributes = original_attributes
    result = check_cube_coordinates(cube, result)
    return result


def expand_bounds(result_cube, cubelist, coord_names, use_midpoint=False):
    """Alter a coordinate on result_cube such that bounds are expanded to cover
    the entire range of the input cubes (cubelist).  The input result_cube is
    modified in place and returned.

    For example, in the case of time cubes if the input cubes have
    bounds of [0000Z, 0100Z] & [0100Z, 0200Z] then the output cube will
    have bounds of [0000Z,0200Z]

    Args:
        result_cube (iris.cube.Cube):
            Cube with coords requiring expansion
        cubelist (iris.cube.CubeList):
            List of input cubes with source coords
        coord_names (list of str):
            Coordinates which should be expanded
        use_midpoint (bool):
            If True, coordinate points returned are halfway between the
            expanded bounds.  If False (default), the upper bound is used.
            Note if the midpoint is used then python will convert
            result.coord('coord').points[0] to a float UNLESS the coord
            units contain 'seconds'.  This is to ensure that midpoints are
            not rounded down, for example when times are in hours.

    Returns:
        iris.cube.Cube:
            Cube with coords expanded.
    """
    for coord in coord_names:

        if len(result_cube.coord(coord).points) != 1:
            emsg = (
                "the expand bounds function should only be used on a"
                'coordinate with a single point. The coordinate "{}" '
                "has {} points."
            )
            raise ValueError(emsg.format(coord, len(result_cube.coord(coord).points)))

        bounds = [cube.coord(coord).bounds for cube in cubelist]
        if any(b is None for b in bounds):
            if not all(b is None for b in bounds):
                raise ValueError(
                    "cannot expand bounds for a mixture of "
                    "bounded / unbounded coordinates"
                )
            points = [cube.coord(coord).points for cube in cubelist]
            new_low_bound = np.min(points)
            new_top_bound = np.max(points)
        else:
            new_low_bound = np.min(bounds)
            new_top_bound = np.max(bounds)
        result_coord = result_cube.coord(coord)
        result_coord.bounds = np.array([[new_low_bound, new_top_bound]])
        if result_coord.bounds.dtype == np.float64:
            result_coord.bounds = result_coord.bounds.astype(np.float32)

        if use_midpoint:
            if "seconds" in str(result_coord.units):
                # integer division of seconds required to retain precision
                dtype_orig = result_coord.dtype
                result_coord.points = [
                    (new_top_bound - new_low_bound) // 2 + new_low_bound
                ]
                # re-cast to original precision to avoid escalating int32s
                result_coord.points = result_coord.points.astype(dtype_orig)
            else:
                # float division of hours required for accuracy
                result_coord.points = [
                    (new_top_bound - new_low_bound) / 2.0 + new_low_bound
                ]
        else:
            result_coord.points = [new_top_bound]

        if result_coord.points.dtype == np.float64:
            result_coord.points = result_coord.points.astype(np.float32)

    return result_cube
