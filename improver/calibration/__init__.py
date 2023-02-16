# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""init for calibration that contains functionality to split forecast, truth
and coefficient inputs.
"""

from typing import Dict, List, Optional, Tuple

from iris.cube import Cube, CubeList

from improver.metadata.probabilistic import (
    get_diagnostic_cube_name_from_probability_name,
)
from improver.utilities.cube_manipulation import MergeCubes


def split_forecasts_and_truth(
    cubes: List[Cube], truth_attribute: str, land_sea_mask_name: str = None,
) -> Tuple[Cube, Cube, Optional[Cube], Optional[List[Cube]]]:
    """
    A common utility for splitting the various inputs cubes required for
    calibration CLIs. These are generally the forecast cubes, historic truths. In some
    instances a land-sea mask and/or cubes containing static additional predictors
    can be used.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            These include the historical forecasts, in the format supported by
            the calibration CLIs, and the truth cubes. Can optionally also contain a
            land-sea mask cube and cubes containing additional predictors.
        truth_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.
        land_sea_mask_name (str):
            Name of the land-sea mask cube. This must be provided if a
            land-sea mask is provided within the list of input cubes, in
            order to identify the land-sea mask.

    Returns:
        - A cube containing all the historic forecasts.
        - A cube containing all the truth data.
        - If found within the input cubes list a land-sea mask will be
          returned, else None is returned.
        - If found within the input cubes list a list of cubes of static additional
          predictors will be returned, else None is returned.

    Raises:
        ValueError:
            An unexpected number of distinct cube names were passed in.
        IOError:
            More than one cube was identified as a land-sea mask.
        IOError:
            Missing truth or historical forecast in input cubes.
    """
    grouped_cubes = {}
    land_sea_mask = []
    static_additional_predictors = []
    for cube in cubes:
        if land_sea_mask_name and cube.name() == land_sea_mask_name:
            land_sea_mask.append(cube)
        elif "time" not in [c.name() for c in cube.coords()]:
            static_additional_predictors.append(cube)
        else:
            try:
                cube_name = get_diagnostic_cube_name_from_probability_name(cube.name())
            except ValueError:
                cube_name = cube.name()
            grouped_cubes.setdefault(cube_name, []).append(cube)
    if len(land_sea_mask) == 0:
        land_sea_mask = None
    elif len(land_sea_mask) != 1:
        raise IOError("Expected at most one cube for land-sea mask.")
    else:
        land_sea_mask = land_sea_mask[0]
    if len(static_additional_predictors) == 0:
        static_additional_predictors = None
    if len(grouped_cubes) == 1:
        # Only one group - all forecast/truth cubes
        diag_name = list(grouped_cubes.keys())[0]
    else:
        diag_names = list(grouped_cubes.keys())
        raise ValueError(
            f"Only forecasts for one diagnostic can be input. The following diagnostics"
            f" were found: {diag_names}"
        )

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

    return forecast, truth, land_sea_mask, static_additional_predictors


def split_forecasts_and_coeffs(
    cubes: CubeList, land_sea_mask_name: Optional[str] = None,
):
    """Split the input forecast, coefficients, static additional predictors,
    land sea-mask and probability template, if provided. The coefficients
    cubes and land-sea mask are identified based on their name. The
    static additional predictors are identified as not have a time
    coordinate. The current forecast and probability template are then split.

    Args:
        cubes:
            A list of input cubes which will be split into relevant groups.
            This includes the forecast, coefficients, static additional
            predictors, land-sea mask and probability template.
        land_sea_mask_name:
            Name of the land-sea mask cube to help identification.

    Returns:
        - A cube containing the current forecast.
        - If found, a cubelist containing the coefficients else None.
        - If found, a cubelist containing the static additional predictor else None.
        - If found, a land-sea mask will be returned, else None.
        - If found, a probability template will be returned, else None.

    Raises:
        ValueError: If multiple items provided, when only one is expected.
        ValueError: If no forecast is found.
    """
    coefficients = CubeList()
    land_sea_mask = None
    grouped_cubes: Dict[str, List[Cube]] = {}
    static_additional_predictors = CubeList()

    for cubelist in cubes:
        for cube in cubelist:
            if "emos_coefficient" in cube.name():
                coefficients.append(cube)
            elif land_sea_mask_name and cube.name() == land_sea_mask_name:
                land_sea_mask = cube
            elif "time" not in [c.name() for c in cube.coords()]:
                static_additional_predictors.append(cube)
            else:
                if "probability" in cube.name() and any(
                    "probability" in k for k in grouped_cubes
                ):
                    msg = (
                        "Providing multiple probability cubes is "
                        "not supported. A probability cube can "
                        "either be provided as the forecast or "
                        "the probability template, but not both. "
                        f"Cubes provided: {grouped_cubes.keys()} "
                        f"and {cube.name()}."
                    )
                    raise ValueError(msg)
                elif cube.name() in grouped_cubes:
                    msg = (
                        "Multiple items have been provided with the "
                        f"name {cube.name()}. Only one item is expected."
                    )
                    raise ValueError(msg)
                grouped_cubes.setdefault(cube.name(), []).append(cube)

    prob_template = None
    # Split the forecast and the probability template.
    if len(grouped_cubes) == 0:
        msg = "No forecast is present. A forecast cube is required."
        raise ValueError(msg)
    elif len(grouped_cubes) == 1:
        (current_forecast,) = list(grouped_cubes.values())[0]
    elif len(grouped_cubes) == 2:
        for key in grouped_cubes.keys():
            if "probability" in key:
                (prob_template,) = grouped_cubes[key]
            else:
                (current_forecast,) = grouped_cubes[key]

    coefficients = coefficients if coefficients else None
    static_additional_predictors = (
        static_additional_predictors if static_additional_predictors else None
    )
    return (
        current_forecast,
        coefficients,
        static_additional_predictors,
        land_sea_mask,
        prob_template,
    )


def validity_time_check(forecast: Cube, validity_times: List[str]) -> bool:
    """Check the validity time of the forecast matches the accepted validity times
    within the validity times list.

    Args:
        forecast:
            Cube containing the forecast to be calibrated.
        validity_times:
            Times at which the forecast must be valid. This must be provided
            as a four digit string (HHMM) where the first two digits represent the hour
            and the last two digits represent the minutes e.g. 0300 or 0315. If the
            forecast provided is at a different validity time then no coefficients
            will be applied.

    Returns:
        If the validity time within the cube matches a validity time within the
        validity time list, then True is returned. Otherwise, False is returned.
    """
    point = forecast.coord("time").cell(0).point
    if f"{point.hour:02}{point.minute:02}" not in validity_times:
        return False
    return True


def add_warning_comment(forecast: Cube) -> Cube:
    """Add a comment to warn that calibration has not been applied.

    Args:
        forecast: The forecast to which a comment will be added.

    Returns:
        Forecast with an additional comment.
    """
    if forecast.attributes.get("comment", None):
        forecast.attributes["comment"] = forecast.attributes["comment"] + (
            "\nWarning: Calibration of this forecast has been attempted, "
            "however, no calibration has been applied."
        )
    else:
        forecast.attributes["comment"] = (
            "Warning: Calibration of this forecast has been attempted, "
            "however, no calibration has been applied."
        )
    return forecast
