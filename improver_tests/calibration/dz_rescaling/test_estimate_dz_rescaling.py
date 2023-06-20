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
"""Unit tests for the EstimateDzRescaling."""

import numpy as np
import pandas as pd
import pytest

import iris
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.calibration.dz_rescaling import EstimateDzRescaling
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.neighbour_finding import NeighbourSelection


def _create_forecasts(forecast_reference_time, validity_time, forecast_period):
    """Create site forecasts for testing."""
    data = np.array([0, 20, 10, 15])
    altitude = np.array([0, 100, 20, 50])
    latitude = np.array([0, 1, 2, 3])
    longitude = np.array([0, 1, 2, 3])
    wmo_id = ["00001", "00002", "00003", "00004"]

    perc_coord = AuxCoord(
        np.array(50, dtype=np.float32), long_name="percentile", units="%",
    )
    fp_coord = AuxCoord(
        np.array(forecast_period * 3600, dtype=TIME_COORDS["forecast_period"].dtype,),
        "forecast_period",
        units=TIME_COORDS["forecast_period"].units,
    )
    time_coord = AuxCoord(
        np.array(
            pd.Timestamp(validity_time).timestamp(), dtype=TIME_COORDS["time"].dtype,
        ),
        "time",
        units=TIME_COORDS["time"].units,
    )
    frt_coord = AuxCoord(
        np.array(
            pd.Timestamp(forecast_reference_time).timestamp(),
            dtype=TIME_COORDS["forecast_reference_time"].dtype,
        ),
        "forecast_reference_time",
        units=TIME_COORDS["forecast_reference_time"].units,
    )
    cube = build_spotdata_cube(
        data,
        "wind_speed_at_10m",
        "m s-1",
        altitude,
        latitude,
        longitude,
        wmo_id,
        scalar_coords=[fp_coord, time_coord, frt_coord, perc_coord],
    )
    return cube


def _create_truths(validity_time):
    """Create site truths for testing. The truth data here shows an example where the
    wind speed is slightly greater at the sites with higher altitude."""
    data = np.array([0, 20.2, 10, 15.1])
    altitude = np.array([0, 100, 20, 50])
    latitude = np.array([0, 1, 2, 3])
    longitude = np.array([0, 1, 2, 3])
    wmo_id = ["00001", "00002", "00003", "00004"]
    time_coord = AuxCoord(
        np.array(
            pd.Timestamp(validity_time).timestamp(), dtype=TIME_COORDS["time"].dtype,
        ),
        "time",
        units=TIME_COORDS["time"].units,
    )
    cube = build_spotdata_cube(
        data,
        "wind_speed_at_10m",
        "m s-1",
        altitude,
        latitude,
        longitude,
        wmo_id,
        scalar_coords=[time_coord],
    )
    return cube


def _create_neighbour_cube():
    """Use the NeighbourSelection functionality to create a cube containing the
    most appropriate neighbouring grid point for a particular site."""
    land_data = np.zeros((9, 9))
    land_data[0:2, 4] = 1
    land_data[4, 4] = 1
    orography_data = np.zeros((9, 9))
    orography_data[0, 4] = 1
    orography_data[1, 4] = 5

    # Global coordinates and cubes
    projection = iris.coord_systems.GeogCS(6371229.0)
    xcoord = iris.coords.DimCoord(
        np.linspace(-4, 4, 9),
        standard_name="longitude",
        units="degrees",
        coord_system=projection,
        circular=True,
    )
    xcoord.guess_bounds()
    ycoord = iris.coords.DimCoord(
        np.linspace(-4, 4, 9),
        standard_name="latitude",
        units="degrees",
        coord_system=projection,
        circular=False,
    )
    ycoord.guess_bounds()

    global_land_mask = iris.cube.Cube(
        land_data,
        standard_name="land_binary_mask",
        units=1,
        dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)],
    )
    global_orography = iris.cube.Cube(
        orography_data,
        standard_name="surface_altitude",
        units="m",
        dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)],
    )
    global_sites = [
        {"altitude": 2.0, "latitude": 0.0, "longitude": 0.1, "wmo_id": 1},
        {"altitude": 10.0, "latitude": 1, "longitude": 1.1, "wmo_id": 2},
        {"altitude": 100.0, "latitude": 2, "longitude": 2.1, "wmo_id": 3},
        {"altitude": 30.0, "latitude": 3, "longitude": 3.1, "wmo_id": 4},
    ]

    plugin = NeighbourSelection()
    cube = plugin.process(global_sites, global_orography, global_land_mask)
    return cube


@pytest.mark.parametrize(
    "forecast_periods,dz_lower_bound,dz_upper_bound,truth_adjustment,expected_data",
    [
        ([6], -200, 200, 0, [0.9998, 0.9989, 0.9894, 0.9968]),
        # ([6], -75, 75, 0, [0.9997, 0.9983, 0.9877, 0.9951]),
        # ([6], -200, 200, -1, [0.9986, 0.9931, 0.9331, 0.9794]),
        # ([6], -75, 75, -1, [0.9979, 0.9895, 0.9241, 0.9689]),
        # ([6], -200, 200, 1, [1.0008, 1.0040, 1.0404, 1.0119]),
        # ([6], -75, 75, 1, [1.0013, 1.0063, 1.0480, 1.0189]),
        # ([6], None, None, 0, [0.9998, 0.9989, 0.9894, 0.9968]),
        # ([6, 12], -200, 200, 0, [0.9998, 0.9989, 0.9894, 0.9968],),
        # ([6, 12], -75, 75, 0, [0.9997, 0.9983, 0.9877, 0.9951],),
    ],
)
def test_estimate_dz_rescaling(
    forecast_periods, dz_lower_bound, dz_upper_bound, truth_adjustment, expected_data
):
    """Test the EstimateDzRescaling plugin including where multiple forecast periods
    are supplied, and where the lower and upper bounds for the difference in altitude
    are specified. An adjustment to truth is also included, which highlights the
    variation in the resulting values that will be used in scaling the forecast
    provided. These variations are large because of the very small number of points used
    in this example."""
    forecast_reference_time = "20170101T0000Z"
    forecast_cubelist = iris.cube.CubeList()
    truth_cubelist = iris.cube.CubeList()
    for forecast_period in forecast_periods:
        validity_time = (
            pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
        ).strftime("%Y%m%dT%H%MZ")
        forecast_cubelist.append(
            _create_forecasts(forecast_reference_time, validity_time, forecast_period)
        )
        truth_cubelist.append(_create_truths(validity_time))

    forecasts = forecast_cubelist.merge_cube()
    truths = truth_cubelist.merge_cube()

    neighbour_cube = _create_neighbour_cube()
    truths.data = np.clip(truths.data + truth_adjustment, 0, None)

    result = EstimateDzRescaling(
        dz_lower_bound=dz_lower_bound, dz_upper_bound=dz_upper_bound,
    )(forecasts, truths, neighbour_cube)
    assert isinstance(result, Cube)

    expected_data = np.squeeze(np.tile(expected_data, (len(forecast_periods), 1)))

    assert result.name() == "scaled_vertical_displacement"
    assert result.units == "1"
    np.testing.assert_allclose(result.data, expected_data, atol=1e-4, rtol=1e-4)

