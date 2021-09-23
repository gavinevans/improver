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
"""init for calibration"""

from collections import OrderedDict
from datetime import timedelta
from improver.utilities.temporal import cycletime_to_datetime
from typing import List, Optional, Tuple

import iris
import numpy as np
import pandas as pd
from iris.cube import Cube

from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.cube_manipulation import MergeCubes


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str
) -> Tuple[Cube, Cube, Optional[Cube]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths,
    and in some instances a land-sea mask is also required.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """
    grouped_cubes = {}
    for cube in cubes:
        try:
            cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
        except ValueError:
            cube_name = cube.name()
        grouped_cubes.setdefault(cube_name, []).append(cube)
    if len(grouped_cubes) == 1:
        # Only one group - all forecast/truth cubes
        land_sea_mask = None
        diag_name = list(grouped_cubes.keys())[0]
    elif len(grouped_cubes) == 2:
        # Two groups - the one with exactly one cube matching a name should
        # be the land_sea_mask, since we require more than 2 cubes in
        # the forecast/truth group
        grouped_cubes = OrderedDict(
            sorted(grouped_cubes.items(), key=lambda kv: len(kv[1]))
        )
        # landsea name should be the key with the lowest number of cubes (1)
        landsea_name, diag_name = list(grouped_cubes.keys())
        land_sea_mask = grouped_cubes[landsea_name][0]
        if len(grouped_cubes[landsea_name]) != 1:
            raise IOError("Expected one cube for land-sea mask.")
    else:
        raise ValueError("Must have cubes with 1 or 2 distinct names.")

    # split non-land_sea_mask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split("=")
    input_cubes = grouped_cubes[diag_name]
    grouped_cubes = {"truth": [], "historical forecast": []}
    for cube in input_cubes:
        if cube.attributes.get(truth_key) == truth_value:
            grouped_cubes["truth"].append(cube)
        else:
            grouped_cubes["historical forecast"].append(cube)

    missing_inputs = " and ".join(k for k, v in grouped_cubes.items() if not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(grouped_cubes["truth"])
    forecast = MergeCubes()(grouped_cubes["historical forecast"])

    return forecast, truth, land_sea_mask


def load_parquet(filepath, filters=None) -> pd.DataFrame:
    df = pd.read_parquet(filepath, filters=filters)
    if df.empty:
        msg = (f"The requested filepath {filepath} does not contain the "
               f"requested contents: {filters}")
        raise IOError(msg)
    return df


def forecast_table_to_cube(filepath, diagnostic, cycletime: str,
        forecast_period: int, training_length: int) -> Cube:
    """Convert a forecast table into an iris Cube.

    Args:
        filepath:
            Path to a parquet file containing forecasts.
        diagnostic:
            The name of the diagnostic. This diagnostic must be available
            within the diag column within the forecast table.

    Returns:

    """
    table = load_parquet(filepath, filters=[("diag", "==", diagnostic)])
    forecast_period_td = timedelta(forecast_period)
    validity_time = cycletime_to_datetime(cycletime) + forecast_period_td

    date_range = pd.date_range(end=validity_time-timedelta(day=1), periods=training_length, freq="D")

    #cube = pd.to_xarray(table).to_iris()

    for adate in date_range:
        table = table.loc[(table["time"] == adate) &
                          (table["forecast_period"] == forecast_period_td)]


        # Filter WMO IDs as only want IDs in truth table.
        time_coord = iris.coords.DimCoord(
            table["time"].unique(),
            "time",
            bounds=table["time_bounds"],
            units=TIME_COORDS["time"].units,
        )
        fp_coord = iris.coords.AuxCoord(
            table["forecast_period"].unique(),
            "forecast_period",
            bounds=table["forecast_period_bounds"],
            units=TIME_COORDS["time"].units,
        )
        frt_coord = iris.coords.AuxCoord(
            table["forecast_reference_time"],
            "forecast_reference_time",
            bounds=table["forecast_reference_time_bounds"],
            units=TIME_COORDS["time"].units,
        )
        for percentile in table["percentile"].unique():
            perc_coord = iris.coords.DimCoord(
                percentile, "percentile", units=1
            )

        forecast_cube = build_spotdata_cube(
            table["fc"].astype(np.float32),  # data
            table["cf_name"].unique(),
            table["units"].unique(),
            table["altitude"].astype(np.float32),  # altitude
            table["latitude"].astype(np.float32),  # latitude
            table["longitude"].astype(np.float32),  # longitude
            table["wmo_id"].unique(),
            additional_dims=time_coord + perc_coord,
        )
        forecast_cube.add_aux_coord(
            (frt_coord, forecast_cube.coord("time").ndim),
            (fp_coord, forecast_cube.coord("time").ndim),
        )

    forecast_cube = RebadgePercentilesAsRealizations()(forecast_cube)

    plugin = EstimateCoefficientsForEnsembleCalibration(
        distribution,
        point_by_point=point_by_point,
        use_default_initial_guess=use_default_initial_guess,
        desired_units=units,
        predictor=predictor,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return plugin(forecast, truth, landsea_mask=land_sea_mask)


def truth_table_to_cube(filepath, diagnostic):

    truth_table = pd.read_parquet(truth, filters=[("diag", "==", {diagnostic})])
