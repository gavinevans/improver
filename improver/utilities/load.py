# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Module for loading cubes."""

from typing import List, Optional, Tuple, Union

import iris
import pandas as pd
from iris import Constraint
from iris.cube import Cube, CubeList

from improver.utilities.cube_manipulation import (
    MergeCubes,
    enforce_coordinate_ordering,
    strip_var_names,
)


def load_cubelist(
    filepath: Union[str, List[str]],
    constraints: Optional[Union[Constraint, str]] = None,
    no_lazy_load: bool = False,
) -> CubeList:
    """Load cubes from filepath(s) into a cubelist. Strips off all
    var names except for "threshold"-type coordinates, where this is different
    from the standard or long name.

    Args:
        filepath:
            Filepath(s) that will be loaded.
        constraints:
            Constraint to be applied when loading from the input filepath.
            This can be in the form of an iris.Constraint or could be a string
            that is intended to match the name of the cube.
            The default is None.
        no_lazy_load:
            If True, bypass cube deferred (lazy) loading and load the whole
            cube into memory. This can increase performance at the cost of
            memory. If False (default) then lazy load.

    Returns:
        CubeList that has been created from the input filepath given the
        constraints provided.
    """
    # Remove legacy metadata prefix cube if present
    constraints = (
        iris.Constraint(cube_func=lambda cube: cube.long_name != "prefixes")
        & constraints
    )

    # Load each file individually to avoid partial merging (not used
    # iris.load_raw() due to issues with time representation)
    if isinstance(filepath, str):
        cubes = iris.load(filepath, constraints=constraints)
    else:
        cubes = iris.cube.CubeList([])
        for item in filepath:
            cubes.extend(iris.load(item, constraints=constraints))

    if not cubes:
        message = "No cubes found using constraints {}".format(constraints)
        raise ValueError(message)

    # Remove var_name from cubes and coordinates (except where needed to
    # describe probabilistic data)
    cubes = strip_var_names(cubes)

    for cube in cubes:

        # Remove metadata attributes pointing to legacy prefix cube
        cube.attributes.pop("bald__isPrefixedBy", None)

        # Ensure the probabilistic coordinates are the first coordinates within
        # a cube and are in the specified order.
        enforce_coordinate_ordering(cube, ["realization", "percentile", "threshold"])
        # Ensure the y and x dimensions are the last within the cube.
        y_name = cube.coord(axis="y").name()
        x_name = cube.coord(axis="x").name()
        enforce_coordinate_ordering(cube, [y_name, x_name], anchor_start=False)
        if no_lazy_load:
            # Force cube's data into memory by touching the .data attribute.
            cube.data

    return cubes


def load_cube(
    filepath: Union[str, List[str]],
    constraints: Optional[Union[Constraint, str]] = None,
    no_lazy_load: bool = False,
) -> Cube:
    """Load the filepath provided using Iris into a cube. Strips off all
    var names except for "threshold"-type coordinates, where this is different
    from the standard or long name.

    Args:
        filepath:
            Filepath that will be loaded or list of filepaths that can be
            merged into a single cube.
        constraints:
            Constraint to be applied when loading from the input filepath.
            This can be in the form of an iris.Constraint or could be a string
            that is intended to match the name of the cube.
            The default is None.
        no_lazy_load:
            If True, bypass cube deferred (lazy) loading and load the whole
            cube into memory. This can increase performance at the cost of
            memory. If False (default) then lazy load.

    Returns:
        Cube that has been loaded from the input filepath given the
        constraints provided.
    """
    cubes = load_cubelist(filepath, constraints, no_lazy_load)
    # Merge loaded cubes
    if len(cubes) == 1:
        cube = cubes[0]
    else:
        cube = MergeCubes()(cubes)
    return cube


def load_parquet(
    filepath: str, filters: Optional[List[Tuple[str, str, str]]] = None
) -> pd.DataFrame:
    """Load the filepath provided to a parquet file into a pandas DataFrame.

    Args:
        filepath:
            Filepath that will be loaded into a single pandas DataFrame.
        filters:
            Filter to restrict the contents of parquet file loaded.
            For example: [('diagnostic', '==', 'wind_speed_at_10m')].

    Returns:
        Pandas DataFrame that has been loaded from the input filepath given
        the filters provided.
    """
    if filters is None:
        filters = []
    df = pd.read_parquet(filepath, filters=filters)
    if df.empty:
        msg = (
            f"The requested filepath {filepath} does not contain the "
            f"requested contents: {filters}"
        )
        raise IOError(msg)
    return df
