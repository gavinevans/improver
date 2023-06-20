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
"""Estimate the the appropriate rescaling based on the difference in altitude between
the grid point and the site."""

from typing import Dict, Optional, Union

import iris
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial as poly1d
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import PostProcessingPlugin
from improver.calibration.utilities import filter_non_matching_cubes
from improver.spotdata.neighbour_finding import NeighbourSelection


class EstimateDzRescaling(PostProcessingPlugin):

    """Estimate a rescaling of the input forecasts based on the difference in
    altitude between the grid point and the site."""

    def __init__(
        self,
        dz_lower_bound: Union[str, float] = None,
        dz_upper_bound: Union[str, float] = None,
        land_constraint: bool = False,
        similar_altitude: bool = False,
    ):
        """Initialise class.

        Args:
            dz_lower_bound: The lowest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a lower
                (or more negative) difference in altitude will be excluded.
                Defaults to None.
            dz_upper_bound: The highest acceptable value for the difference in
                altitude between the grid point and the site. Sites with a larger
                positive difference in altitude will be excluded. Defaults to None.
            polyfit_deg: The degree chosen for the fitting polynomial. This is
                set to 1. Please see numpy.polynomial.polynomial.Polynomial.fit
                for further information.
        """
        if dz_lower_bound is None:
            self.dz_lower_bound = -np.inf
        else:
            self.dz_lower_bound = np.float32(dz_lower_bound)
        if dz_upper_bound is None:
            self.dz_upper_bound = np.inf
        else:
            self.dz_upper_bound = np.float32(dz_upper_bound)
        self.polyfit_deg = 1

        self.neighbour_selection_method = NeighbourSelection(
            land_constraint=land_constraint, minimum_dz=similar_altitude
        ).neighbour_finding_method_name()

    def _fit_polynomial(self, forecasts: Cube, truths: Cube, dz: Cube) -> float:
        """Create a polynomial fit between the log of the ratio of forecasts and truths,
        and the difference in altitude between the grid point and the site.

        Args:
            forecasts: Forecast cubes.
            truths: Truth cubes.
            dz: Difference in altitude between the grid point and the site location.

        Returns:
            A scale factor deduced from a polynomial fit.
        """
        data_filter = (
            (forecasts.data != 0)
            & (truths.data != 0)
            & (dz.data > self.dz_lower_bound)
            & (dz.data < self.dz_upper_bound)
        )

        forecasts_data = forecasts.data.flatten()
        truths_data = truths.data.flatten()
        dz_data = np.broadcast_to(dz.data, forecasts.shape).flatten()
        data_filter = data_filter.flatten()

        log_error_ratio = np.log(forecasts_data[data_filter] / truths_data[data_filter])

        scale_factor = poly1d(
            polyfit(dz_data[data_filter], log_error_ratio, self.polyfit_deg)
        ).coef

        # import pdb
        # pdb.set_trace()
        # import matplotlib.pyplot as plt
        # t = np.arange(1, 100)
        # plt.scatter(dz_data[data_filter], log_error_ratio)
        # plt.plot(t, scale_factor[0] + scale_factor[1] * t, color="red")
        # plt.show()

        # Only retain the multiplicative coefficient as the scale factor.
        # This helps conceptually with the difference in altitude rescaling
        # where if the dz of the grid point and the site are the same, then no
        # adjustment will be made.
        scale_factor = scale_factor[1]
        return scale_factor

    def _compute_scaled_dz(self, scale_factor: float, dz: Cube) -> Cube:
        """Compute the scaled difference in altitude.

        Args:
            scale_factor: A scale factor deduced from a polynomial fit.
            dz: The difference in altitude between the grid point and the site.

        Returns:
            Scaled difference in altitude.
        """
        scaled_dz = dz.copy()
        scaled_dz.rename("scaled_vertical_displacement")
        scaled_dz.units = "1"

        # Multiplication by -1 using negative exponent rule, so that this term can
        # be multiplied by the forecast during the application step.
        scaled_dz.data = np.exp(-1.0 * scale_factor * dz.data)

        # Compute lower and upper bounds for the scaled dz.
        # dz_lower_bound may not result in the lower bound for the scaled dz depending
        # upon the sign of the scale_factor term.
        scaled_dz_a = np.exp(-1.0 * scale_factor * self.dz_lower_bound)
        scaled_dz_b = np.exp(-1.0 * scale_factor * self.dz_upper_bound)
        scaled_dz_lower = np.amin([scaled_dz_a, scaled_dz_b])
        scaled_dz_upper = np.amax([scaled_dz_a, scaled_dz_b])

        scaled_dz.data[(scaled_dz.data < scaled_dz_lower)] = scaled_dz_lower
        scaled_dz.data[(scaled_dz.data > scaled_dz_upper)] = scaled_dz_upper

        return scaled_dz

    def process(self, forecasts: Cube, truths: Cube, neighbour_cube: Cube) -> Cube:
        """Fit a polynomial using the forecasts and truths to compute a scaled
        version of the difference of altitude between the grid point and the
        site location.

        A mathematical summary of the steps within this plugin are:
        1. Estimate a scale factor for the relationship between the difference in
        altitude between the grid point and the site, and the natural log of the
        forecast divided by the truth.

        .. math::
            dz = \ln(forecast / truth) * scale_factor

        2. Rearranging this equation gives:

        .. math::
            truth = forecast / \exp(s*dz)

        or alternatively:

        .. math::
            truth = forecast * \exp(-s*dz)

        This plugin is aiming the estimate the :math:`\exp(-s*dz)` component, which
        can later be used for multiplying by a forecast to estimate the truth.

        Args:
            forecasts: Forecast cubes.
            truths: Truth cubes.
            dz: Difference in altitude between the grid point and the site location.

        Returns:
            A scaled version difference of altitude between the grid point and the
            site location.
        """
        method = iris.Constraint(
            neighbour_selection_method_name=self.neighbour_selection_method
        )
        index_constraint = iris.Constraint(
            grid_attributes_key=["vertical_displacement"]
        )
        dz_cube = neighbour_cube.extract(method & index_constraint)

        sites = list(
            set(forecasts.coord("wmo_id").points)
            & set(truths.coord("wmo_id").points)
            & set(dz_cube.coord("wmo_id").points)
        )
        constr = iris.Constraint(wmo_id=sites)
        training_forecasts = forecasts.extract(constr)
        training_truths = truths.extract(constr)
        dz_training_cube = dz_cube.extract(constr)

        sites = list(
            set(forecasts.coord("wmo_id").points) & set(dz_cube.coord("wmo_id").points)
        )
        constr = iris.Constraint(wmo_id=sites)
        dz_apply_cube = dz_cube.extract(constr)

        scaled_dz_slices = iris.cube.CubeList([])
        for forecast_slice in training_forecasts.slices_over("forecast_period"):
            forecast_slice, truth_slice = filter_non_matching_cubes(
                forecast_slice.copy(), training_truths
            )

            constr = iris.Constraint(percentile=50.0)
            forecast_slice = forecast_slice.extract(constr)

            scale_factor = self._fit_polynomial(
                forecast_slice, truth_slice, dz_training_cube
            )
            scaled_dz_slice = self._compute_scaled_dz(scale_factor, dz_apply_cube)
            scaled_dz_slice.add_aux_coord(forecast_slice.coord("forecast_period"))
            scaled_dz_slices.append(scaled_dz_slice)
        return scaled_dz_slices.merge_cube()


class ApplyDzRescaling(PostProcessingPlugin):

    """Apply rescaling of the forecast using the difference in altitude between the
    grid point and the site."""

    def __init__(self):
        """Initialise class."""
        pass

    def process(self, forecast, scaled_dz):
        """Apply rescaling of the forecast to account for differences in the altitude
        between the grid point and the site, as assessed using a training dataset.
        The most appropriate scaled dz is selected by choosing the forecast period
        that is greater than or equal to the forecast period of the forecast.

        Args:
            forecast: Forecast to be adjusted using dz rescaling.
            scaled_dz: A scaled version of the difference in altitude between the
                grid point and the site.

        Returns:
            A forecast has been rescaled to account for differences between historic
            forecasts and truths that can be accounted for based on the difference
            in altitude between the grid point and the site.
        """
        import pdb

        pdb.set_trace()
        fp_diff = (
            scaled_dz.coord("forecast_period").points
            - forecast.coord("forecast_period").points
        )
        value = np.argmin(fp_diff[fp_diff >= 0])
        value = scaled_dz.coord("forecast_period").points[fp_diff >= 0][value]

        constr = iris.Constraint(forecast_period=value)

        forecast.data = forecast.data * scaled_dz.extract(constr).data
        return forecast
