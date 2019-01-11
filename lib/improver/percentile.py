# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Module containing percentiling classes."""


import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.constants import DEFAULT_PERCENTILES


class PercentileConverter(object):

    """Plugin for converting from a set of values to a PDF.

    Generate percentiles together with min, max, mean, stdev.

    """

    def __init__(self, collapse_coord, percentiles=None,
                 fast_percentile_method=True):
        """
        Create a PDF plugin with a given source plugin.

        Args:
            collapse_coord (str or list of str):
                The name of the coordinate(s) to collapse over. This
                coordinate(s) will no longer be present on the output cube, as
                it will have been replaced by the percentile coordinate.

            percentiles (Iterable list of floats or None):
                Percentile values at which to calculate; if not provided uses
                DEFAULT_PERCENTILES. (optional)

        Raises:
            TypeError: If collapse_coord is not a string.

        """
        if not isinstance(collapse_coord, list):
            collapse_coord = [collapse_coord]
        if not all([isinstance(test_coord, str)
                    for test_coord in collapse_coord]):
            raise TypeError('collapse_coord is {!r}, which is not a string '
                            'as is expected.'.format(collapse_coord))

        if percentiles is not None:
            self.percentiles = [np.float32(value) for value in percentiles]
        else:
            self.percentiles = [
                np.float32(value) for value in DEFAULT_PERCENTILES]

        # Collapsing multiple coordinates results in a new percentile
        # coordinate, its name suffixed by the original coordinate names. Such
        # a collapse is cummutative (i.e. coordinate order doesn't matter).
        # However the coordinates are sorted here such that the resulting
        # percentile coordinate has a consistent name regardless of the order
        # in which the user provides the original coordinate names.
        self.collapse_coord = sorted(collapse_coord)
        self.fast_percentile_method = fast_percentile_method

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<PercentileConverter: collapse_coord={}, percentiles={}'
                .format(self.collapse_coord, self.percentiles))
        return desc

    def process(self, cube):
        """
        Create a cube containing the percentiles as a new dimension.

        What's generated by default is:
            * 15 percentiles - (0%, 5%, 10%, 20%, 25%, 30%, 40%, 50%, 60%,
              70%, 75%, 80%, 90%, 95%, 100%)

        Args:
            cube (iris.cube.Cube):
                Given the collapse coordinate, convert the set of values
                along that coordinate into a PDF and extract percentiles.

        Returns:
            cube (iris.cube.Cube):
                A single merged cube of all the cubes produced by each
                percentile collapse.

        """
        # Store data type and enforce the same type on return.
        data_type = cube.dtype
        # Test that collapse coords are present in cube before proceding.
        n_collapse_coords = len(self.collapse_coord)
        n_valid_coords = sum([test_coord == coord.name()
                              for coord in cube.coords()
                              for test_coord in self.collapse_coord])

        if n_valid_coords == n_collapse_coords:
            result = cube.collapsed(
                self.collapse_coord, iris.analysis.PERCENTILE,
                percent=self.percentiles,
                fast_percentile_method=self.fast_percentile_method)
            result.data = result.data.astype(data_type)
            for coord in self.collapse_coord:
                result.remove_coord(coord)
            return result

        raise CoordinateNotFoundError(
            "Coordinate '{}' not found in cube passed to {}.".format(
                self.collapse_coord, self.__class__.__name__))
