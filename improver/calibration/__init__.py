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
from typing import List, Optional, Tuple

from iris.cube import Cube, CubeList

from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.utilities.cube_manipulation import MergeCubes


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str, land_sea_mask_name: bool
) -> Tuple[Cube, Cube, Optional[CubeList], Optional[Cube]]:
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
    cubes_dict = {"truths": [], "land_sea_mask": [], "other": []}
    # split non-land_sea_mask cubes on forecast vs truth
    truth_key, truth_value = truth_attribute.split("=")
    for cube in cubes:
        try:
            cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
        except ValueError:
            cube_name = cube.name()
        if cube.attributes.get(truth_key) == truth_value:
            cubes_dict["truth"].setdefault(cube_name, []).append(cube)
        elif cube_name == land_sea_mask_name:
            cubes_dict["land_sea_mask"].setdefault(cube_name, []).append(cube)
        else:
            blend_time_list = [c for c in cube.coords() if c.name() == "blend_time"]
            if len(blend_time_list):
                cube.remove_coord("blend_time")
            cube.coord("forecast_period").attributes = {}
            cube.coord("forecast_reference_time").attributes = {}
            cubes_dict["other"].setdefault(cube_name, []).append(cube)

    if len(cubes_dict["truths"]) > 1:
        msg = (f"Truth supplied for multiple diagnostics {list(cubes_dict['truth'].keys())}. "
               "The truth should only exist for one diagnostic.")
        raise ValueError(msg)

    if land_sea_mask_name and not cubes_dict["land_sea_mask"]:
        raise IOError("Expected one cube for land-sea mask with "
                      f"the name {land_sea_mask_name}.")

    diag_name = cubes_dict["truths"].keys()[0]
    cubes_dict["historic_forecasts"] = cubes_dict["other"][diag_name]
    for k, v in cubes_dict["other"].items():
        if k != diag_name:
            cubes_dict["additional_fields"].set_default(k, []).append(v)

    missing_inputs = " and ".join(k for k, v in cubes_dict.items() if k in ["truth", "historic_forecasts"] and not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(cubes_dict["truths"][diag_name])
    forecast = MergeCubes()(cubes_dict["historic_forecasts"])
    additional_fields = CubeList([MergeCubes()(cubes_dict["additional_fields"][k]) for k in cubes_dict["additional_fields"]])
    return forecast, truth, additional_fields, cubes_dict["land_sea_mask"]

    # if len(grouped_cubes) == 1:
    #     # Only one group - all forecast/truth cubes
    #     land_sea_mask = None
    #     diag_name = list(grouped_cubes.keys())[0]
    # elif len(grouped_cubes) == 2:
    #     # Two groups - the one with exactly one cube matching a name should
    #     # be the land_sea_mask, since we require more than 2 cubes in
    #     # the forecast/truth group
    #     grouped_cubes = OrderedDict(
    #         sorted(grouped_cubes.items(), key=lambda kv: len(kv[1]))
    #     )
    #     # landsea name should be the key with the lowest number of cubes (1)
    #     landsea_name, diag_name = list(grouped_cubes.keys())
    #     land_sea_mask = grouped_cubes[landsea_name][0]
    #     if len(grouped_cubes[landsea_name]) != 1:
    #         raise IOError("Expected one cube for land-sea mask.")
    # else:
    #     raise ValueError("Must have cubes with 1 or 2 distinct names.")

    # split non-land_sea_mask cubes on forecast vs truth
    # truth_key, truth_value = truth_attribute.split("=")
    # grouped_cubes = {"truth": [], "historical forecast": []}
    # for cube in cubes_dict:
    #     if cube.attributes.get(truth_key) == truth_value:
    #         grouped_cubes["truth"].append(cube)
    #     else:
    #         blend_time_list = [c for c in cube.coords() if c.name() == "blend_time"]
    #         if len(blend_time_list):
    #             cube.remove_coord("blend_time")
    #         cube.coord("forecast_period").attributes = {}
    #         cube.coord("forecast_reference_time").attributes = {}
    #         grouped_cubes["historical forecast"].append(cube)

    missing_inputs = " and ".join(k for k, v in grouped_cubes.items() if not v)
    if missing_inputs:
        raise IOError(f"Missing {missing_inputs} input.")

    truth = MergeCubes()(grouped_cubes["truth"])
    forecast = MergeCubes()(grouped_cubes["historical forecast"])
    return forecast, truth, land_sea_mask
